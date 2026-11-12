"""
LLM integration module for Ollama and optional OpenAI-compatible fallback.
Implements retry logic, configurable timeout, health checks, and detailed telemetry.
"""

import time
import json
import logging
from typing import Optional, Dict, Any, List
import requests

from faq_assistant.config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    DEFAULT_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    MAX_RETRIES,
    RETRY_BACKOFF_FACTOR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FAQAssistant.LLM")


class LLMResult:
    """Encapsulates the response from the LLM including telemetry."""
    def __init__(
        self,
        content: str,
        latency_seconds: float,
        model_name: str,
        success: bool = True,
        error_message: Optional[str] = None,
        source: str = "ollama"
    ):
        self.content = content.strip()
        self.latency_seconds = latency_seconds
        self.model_name = model_name
        self.success = success
        self.error_message = error_message
        self.source = source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "latency_seconds": round(self.latency_seconds, 3),
            "model_name": self.model_name,
            "success": self.success,
            "error_message": self.error_message,
            "source": self.source
        }


class OllamaClient:
    """
    Robust wrapper for Ollama local LLM server.
    Features exponential backoff retry, strict timeouts, and fallback handling.
    """

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model: str = OLLAMA_MODEL,
        timeout: int = LLM_TIMEOUT_SECONDS
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """
        Verify if the Ollama daemon is reachable and whether the configured model exists.
        """
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                model_found = any(self.model in name for name in model_names)
                return {
                    "server_online": True,
                    "model_available": model_found,
                    "installed_models": model_names,
                    "target_model": self.model
                }
            return {
                "server_online": False,
                "model_available": False,
                "error": f"HTTP status {resp.status_code}"
            }
        except Exception as e:
            return {
                "server_online": False,
                "model_available": False,
                "error": str(e)
            }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = 512,
        stop_sequences: Optional[List[str]] = None
    ) -> LLMResult:
        """
        Send a generation request to Ollama with retry and timeout.
        If Ollama fails and OpenAI fallback is configured, delegates to OpenAI.
        """
        start_time = time.time()
        last_error = None

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        if system_prompt:
            payload["system"] = system_prompt
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        url = f"{self.host}/api/generate"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )
                if resp.status_code == 200:
                    data = resp.json()
                    elapsed = time.time() - start_time
                    content = data.get("response", "")
                    return LLMResult(
                        content=content,
                        latency_seconds=elapsed,
                        model_name=self.model,
                        success=True,
                        source="ollama"
                    )
                else:
                    last_error = f"Ollama returned HTTP {resp.status_code}: {resp.text}"
                    logger.warning(f"Ollama attempt {attempt}/{MAX_RETRIES} failed: {last_error}")

            except requests.exceptions.Timeout:
                last_error = f"Ollama request timed out after {self.timeout}s (attempt {attempt}/{MAX_RETRIES})"
                logger.warning(last_error)
            except requests.exceptions.ConnectionError:
                last_error = f"Cannot connect to Ollama at {self.host} (attempt {attempt}/{MAX_RETRIES})"
                logger.warning(last_error)
            except Exception as e:
                last_error = f"Unexpected error during Ollama call: {str(e)}"
                logger.warning(last_error)

            if attempt < MAX_RETRIES:
                sleep_time = RETRY_BACKOFF_FACTOR ** attempt
                time.sleep(sleep_time)

        # Try optional OpenAI-compatible API fallback if enabled
        if OPENAI_API_KEY:
            logger.info("Attempting OpenAI-compatible fallback...")
            return self._call_openai_fallback(prompt, system_prompt, temperature, max_tokens, start_time)

        elapsed = time.time() - start_time
        return LLMResult(
            content="Error: The language model did not respond in time. Please check your Ollama service.",
            latency_seconds=elapsed,
            model_name=self.model,
            success=False,
            error_message=last_error,
            source="ollama-failed"
        )

    def _call_openai_fallback(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        start_time: float
    ) -> LLMResult:
        """Fallback to OpenAI-compatible endpoint when local LLM is offline or failing."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            url = f"{OPENAI_API_BASE.rstrip('/')}/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            body = {
                "model": OPENAI_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                elapsed = time.time() - start_time
                return LLMResult(
                    content=content,
                    latency_seconds=elapsed,
                    model_name=OPENAI_MODEL,
                    success=True,
                    source="openai-fallback"
                )
            else:
                elapsed = time.time() - start_time
                return LLMResult(
                    content="Fallback API failed to generate a response.",
                    latency_seconds=elapsed,
                    model_name=OPENAI_MODEL,
                    success=False,
                    error_message=f"HTTP {resp.status_code}: {resp.text}",
                    source="openai-fallback"
                )
        except Exception as e:
            elapsed = time.time() - start_time
            return LLMResult(
                content="Fallback API encounter an error.",
                latency_seconds=elapsed,
                model_name=OPENAI_MODEL,
                success=False,
                error_message=str(e),
                source="openai-fallback"
            )


# Default shared client instance
default_client = OllamaClient()
