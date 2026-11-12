# Course & College FAQ Assistant — Local LLM Edition

A Streamlit and CLI academic FAQ assistant using **Ollama locally** (no OpenAI or Gemini API key required).
It answers student questions from a local college FAQ knowledge base (`faq_assistant/faq_data.json`) using
TF-IDF + keyword retrieval, rule/LLM intent routing, conversation memory, and three prompt engineering
strategies: **Zero-Shot**, **Few-Shot**, and **Role Prompting**.

## Requirements
- Windows 10/11 (also works on Linux/macOS)
- Python 3.10+ (tested on Python 3.13)
- Ollama for Windows (`https://ollama.com/download`)
- 4 GB+ free RAM for `qwen2.5:0.5b` (8 GB+ recommended if switching to `llama3.2:3b` or `qwen2.5:3b`)

## 1. Install Ollama and download a model

Install Ollama from https://ollama.com/download

In PowerShell:

```powershell
ollama pull qwen2.5:0.5b
ollama run qwen2.5:0.5b
```

Try a test question, then type `/bye`. If your machine has more RAM and you want larger responses,
pull `llama3.2:3b` or `qwen2.5:3b` and update `OLLAMA_MODEL` in `.env`.

## 2. Set up the Python project

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

No API key is required. `.env` defaults to local Ollama (an optional OpenAI-compatible fallback is
supported if `OPENAI_API_KEY` is set).

## 3. Run the app

Make sure Ollama is running (`ollama serve` if not already running in the background):

### Streamlit Web UI
```powershell
python run.py --ui
# or directly:
streamlit run faq_assistant/app.py
```

### Interactive Terminal CLI
```powershell
python run.py --cli
```

### Side-by-Side Prompt Comparison
```powershell
python run.py --compare -q "What is the penalty for late homework?"
```

### Evaluation Suite & Unit Tests
```powershell
python run.py --eval --runs 3
python -m pytest
```

## Project Structure

```text
.
├── run.py                          # Launcher (--ui, --cli, --compare, --eval)
├── REPORT.md                       # Project report and strategy analysis
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── faq_assistant/
│   ├── app.py                      # Streamlit web UI (chat, compare, FAQs, eval)
│   ├── cli.py                      # Rich terminal chat interface
│   ├── config.py                   # Hardware checks, Ollama config, paths
│   ├── engine.py                   # Main assistant pipeline
│   ├── llm.py                      # Ollama client with retries and timeout
│   ├── prompts.py                  # Zero-Shot, Few-Shot, Role, Intent prompts
│   ├── intent.py                   # Rule-based + LLM intent classifier
│   ├── memory.py                   # Multi-turn history and follow-up resolution
│   ├── retrieval.py                # TF-IDF + keyword boost FAQ retriever
│   ├── compare.py                  # Side-by-side strategy comparison helper
│   ├── evaluate.py                 # 12-query evaluation script
│   ├── faq_data.json               # 14 sample college/course FAQs
│   └── tests/                      # Unit tests (pytest)
└── results/
    ├── evaluation_results.csv      # Saved per-query evaluation runs
    └── evaluation_summary.md       # Saved summary table
```

## Architecture
Streamlit / CLI -> Engine -> Intent Classifier & Memory -> TF-IDF + Keyword Retriever ->
Zero-Shot / Few-Shot / Role Prompt -> Local Ollama -> Evaluation Report.

## Evaluation Summary (12 Queries, 3 Runs Each)

| Strategy | Correctness | Relevance | Consistency | Quality | Overall | Intent Acc. | Unknown Declined | Avg Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot** | 4.79 | 4.93 | 4.44 | 4.60 | **4.69 / 5.0** | 100.0% | 100.0% | 4.91s |
| **Few-Shot** | 4.88 | 4.93 | 4.25 | 4.50 | **4.64 / 5.0** | 100.0% | 100.0% | 4.96s |
| **Role** | 4.54 | 4.77 | 3.72 | 4.75 | **4.44 / 5.0** | 100.0% | 100.0% | 5.36s |

See [REPORT.md](REPORT.md) and [results/evaluation_summary.md](results/evaluation_summary.md) for the
per-query breakdown and strategy comparison notes.

## Notes
- Runs locally on Ollama (`qwen2.5:0.5b` by default); no external API calls are made unless `OPENAI_API_KEY` is configured.
- Retrieval uses TF-IDF n-grams with keyword boosting and a `0.26` similarity threshold so it works cleanly on CPU-only laptops without a heavy vector database.
- Questions below the retrieval threshold trigger a fallback refusal instead of letting the model guess campus policies.
- The sample FAQs in `faq_assistant/faq_data.json` use a fictional department ("Evergreen State University") and can be edited as needed.
- Evaluation scores in `results/` are heuristic (keyword coverage, refusal checks, and cross-run token overlap), not human-graded benchmarks.
