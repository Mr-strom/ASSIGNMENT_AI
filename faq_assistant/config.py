"""
Configuration settings for the Course & College FAQ Assistant.
Handles hardware checks, Ollama settings, retrieval thresholds, and file paths.
"""

import os
import platform
import psutil
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if available
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

# ---------------------------------------------------------------------------
# Hardware Detection Helpers
# ---------------------------------------------------------------------------
def detect_hardware():
    """
    Detect system CPU, total RAM, available RAM, and GPU info.
    Returns a dictionary of hardware specs.
    """
    specs = {
        "os": f"{platform.system()} {platform.release()} ({platform.architecture()[0]})",
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores": psutil.cpu_count(logical=False) or 4,
        "logical_cpus": psutil.cpu_count(logical=True) or 8,
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "available_ram_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
        "gpu": "Integrated / CPU inference"
    }

    # Detect GPU if available on Windows / Linux
    try:
        if platform.system() == "Windows":
            import subprocess
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                capture_output=True, text=True, timeout=5
            )
            gpus = [line.strip() for line in out.stdout.splitlines() if line.strip()]
            if gpus:
                specs["gpu"] = ", ".join(gpus)
    except Exception:
        pass

    return specs


def recommend_model(available_ram_gb: float) -> str:
    """
    Select recommended model based on available RAM (leaving ~30% headroom):
      - <6 GB: llama3.2:1b or qwen2.5:0.5b
      - 6-8 GB: llama3.2:3b or qwen2.5:3b
      - 8-16 GB: llama3.1:8b or qwen2.5:7b
      - >16 GB: qwen2.5:14b or llama3.1:8b
    """
    if available_ram_gb < 6.0:
        return "llama3.2:1b"
    elif available_ram_gb < 8.0:
        return "llama3.2:3b"
    elif available_ram_gb <= 16.0:
        return "llama3.2:3b"  # 3b is optimal on CPU for low latency & reliability
    else:
        return "llama3.1:8b"


# Run hardware detection at startup
SYSTEM_SPECS = detect_hardware()
RECOMMENDED_MODEL = recommend_model(SYSTEM_SPECS["available_ram_gb"])

# ---------------------------------------------------------------------------
# LLM and Service Configuration
# ---------------------------------------------------------------------------
# Local Ollama settings (default)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")

# Optional OpenAI-compatible fallback (can be set via env var)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# LLM Generation parameters
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.2"))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF_FACTOR = float(os.getenv("RETRY_BACKOFF_FACTOR", "1.5"))

# Retrieval parameters
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.26"))
TOP_K_FAQS = int(os.getenv("TOP_K_FAQS", "3"))

# Memory / History parameters
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5"))

# Paths
DATA_PATH = BASE_DIR / "faq_data.json"
RESULTS_DIR = BASE_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
