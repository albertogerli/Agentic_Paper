<div align="center">

# 🧑‍🔬 Agentic_Paper

**A multi-agent LLM orchestrator for academic peer-review.**

[![PyPI version](https://img.shields.io/pypi/v/agentic-paper.svg)](https://pypi.org/project/agentic-paper/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/albertogerli/Agentic_Paper/actions/workflows/ci.yml/badge.svg)](https://github.com/albertogerli/Agentic_Paper/actions/workflows/ci.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-46aef7.svg)](https://github.com/astral-sh/ruff)

*Built for students, PhDs, and researchers who want a transparent, reproducible second opinion on a manuscript — not another opaque chatbot.*

</div>

---

## Why Agentic_Paper?

**This is not another ChatGPT wrapper.**

A single LLM, given a paper and the prompt *"please review this"*, gives you the average of the internet. `Agentic_Paper` does something genuinely different:

- 🧠 **12 specialised reviewer agents** run **in parallel**, each with its own role, prompt, and base complexity — Methodology, Results, Literature, Structure, Impact, Contradiction, Ethics, AI-Origin, Hallucination, Citation Validator, Statcheck Validator, Revision Assessor.
- 🧑‍⚖️ A **Coordinator** synthesises their structured verdicts, names disagreements, and orders revision priorities.
- ✉️ An **Editor** + an **Author/Editor Summary** agent produce a journal-style decision letter and the confidential note to the editor — separately.
- 📜 **Every single LLM call is audited** — token counts, latency, cost estimate, prompt hash, thinking-mode flag, seed — written to `audit.jsonl` so you can prove what was asked and answered. No hallucination hides in the dark.
- 🔎 **Citations are validated against [OpenAlex](https://openalex.org)** (~250M open scholarly records, no API key needed). Fabricated references get flagged automatically.
- 🧮 **Reported p-values are recomputed** via the R `statcheck` package — if a paper says `t(28) = 2.3, p = .01` and the math says `p ≈ 0.029`, you'll see it.
- 🔌 **Multi-provider, pluggable**: OpenAI, Anthropic Claude, Google Gemini, **and any OpenAI-compatible local endpoint** — see [§ Local & Free Models](#-local--free-models-with-ollama).
- 🎛️ **Typed everything**: reviewers don't return free-form prose, they return validated `pydantic` models. Downstream agents consume structure, not substrings.

Outputs: a Markdown report, a stand-alone HTML dashboard, a structured JSON, and a `run_id`-scoped folder you can hand off when a journal asks *"how was this assessment produced?"*.

---

## Installation

```bash
pip install agentic-paper
```

That's it. Pure-Python; works on macOS, Linux, and Windows with **Python 3.10+**.

For the optional web UI (FastAPI + HTMX live demo):

```bash
pip install "agentic-paper[web]"
```

For statistical sanity checking (recommended for empirical papers), also install [R](https://www.r-project.org/) and the `statcheck` + `jsonlite` packages:

```r
install.packages(c("statcheck", "jsonlite"))
```

If R isn't available, the rest of the pipeline still runs — the Statcheck Validator simply reports *"not available"* in the final report.

---

## Quickstart

### 1. Set a provider key

```bash
export OPENAI_API_KEY="sk-..."
# Optional, for multi-provider routing:
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

> 💡 **No budget?** Skip this step and jump to [Local & Free Models](#-local--free-models-with-ollama).

### 2. Review a paper from the terminal

```bash
agentic-paper paper.pdf --seed 42
```

Outputs land under `output_paper_review/<run_id>/` — open `dashboard_*.html` for a styled report, or read `review_report_*.md` directly.

### 3. Or use the web UI

```bash
agentic-paper-web --port 8000
# → http://127.0.0.1:8000/
```

A clean drop-zone page: drag a PDF in, watch the 12 agents think live (real `thinking_delta` stream when the provider supports it), then read the report inline. Optional **Bring-Your-Own-Key** form for sharing the demo with colleagues without exposing your account — keys are held in the worker stack frame, never logged, never written to disk.

```
┌─────────────────────────────────────────────┐
│  drop a PDF here  →  watch the agents work  │
│  ⠋ methodology   reading…                   │
│  ✓ results       done (4.2 s, $0.018)       │
│  ⠴ literature    thinking…                  │
│  …                                          │
└─────────────────────────────────────────────┘
```

### ⚡ Auto-Mode: never fail because of a missing key

The web UI's routing profiles (`max` / `std` / `quick`) deliberately spread agents across **multiple vendors** to play to each model's strengths — for example `std` sends High-tier reasoning to Claude, Standard tier to GPT, Basic tier to Gemini. If you only paste **one** API key in the BYOK form, naïve routing would 404 on the other two providers and tank the run.

**Auto-Mode fixes this transparently.** When the BYOK form is submitted:

1. Each tier is checked against the keys you actually provided.
2. Tiers pointing to an unavailable provider are remapped to an equivalent model on a provider you *do* have (e.g. `tier_high: anthropic/claude-opus-4-7` → `google/gemini-3-pro`).
3. `thinking_budget` and the tier's role intensity are preserved — Auto-Mode picks the flagship reasoning model of the fallback vendor for `tier_high`, the mid-tier for `tier_standard`, and the cheapest for `tier_basic`.
4. A **yellow banner** at the top of the run page lists every remap with the original vs. new (provider, model) so you know exactly what changed.

The run proceeds end-to-end with a single key, with no manual config edits. Auto-Mode only kicks in when at least one BYOK key is supplied — runs that use the server-side config are left alone.

---

## 🦙 Local & Free Models (with Ollama)

**You don't need a credit card to use Agentic_Paper.** The `ProviderRegistry` accepts any OpenAI-compatible endpoint, which means you can run the entire pipeline against [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai), [vLLM](https://github.com/vllm-project/vllm), or any local server you control. **Free peer-review, fully private, all on your laptop.**

### Step-by-step: Ollama + Llama 3

```bash
# 1. Install Ollama from https://ollama.com (one-line installer)
# 2. Pull a model — Llama 3.1 8B fits on a laptop with 16 GB RAM
ollama pull llama3.1

# 3. Start Ollama in the background (it auto-serves an OpenAI-compatible API on :11434)
ollama serve &

# 4. Point Agentic_Paper at it — two env vars is all it takes
export OPENAI_API_KEY="ollama"                          # any non-empty string
export OPENAI_API_BASE="http://localhost:11434/v1"      # Ollama's OpenAI-compat endpoint

# 5. Run the review using your local model
agentic-paper paper.pdf --config config.local.yaml
```

Minimal `config.local.yaml` to wire every tier to the local model:

```yaml
output_dir: output_paper_review
routing:
  tier_high:     { provider: openai, model: llama3.1 }
  tier_standard: { provider: openai, model: llama3.1 }
  tier_basic:    { provider: openai, model: llama3.1 }
providers:
  openai:
    api_key_env: OPENAI_API_KEY
    base_url: http://localhost:11434/v1
```

### Recommended local model tiers

| Hardware | Suggested model | Notes |
|---|---|---|
| Laptop, 16 GB RAM | `llama3.1` (8B) | Solid baseline. Reviews are slower but coherent. |
| Workstation, 32 GB+ | `llama3.1:70b` or `qwen2.5:32b` | Closer to GPT-4o quality on reasoning. |
| GPU box, 24 GB+ VRAM | `deepseek-r1` via vLLM | Excellent for the Methodology / Contradiction reviewers. |
| Mac Studio (M2 Ultra+) | `llama3.1:70b` MLX | Apple-silicon native; faster than CUDA at comparable mem. |

### Caveats with local models

- **Structured outputs**: small open-weight models occasionally violate the JSON schema. Agentic_Paper retries with `tenacity` and falls back to `response_format: json_object`. Larger models (≥ 30B) are noticeably more reliable.
- **Quality**: a 7-8B local model will not match Claude Opus 4.7 — but for a first pass on a draft (catching contradictions, missing citations, structural issues), it's more than enough.
- **Privacy**: nothing leaves your machine. Perfect for unpublished manuscripts under embargo or NDA.
- **Cost**: literally zero (modulo electricity).

### Mixed routing: free local + paid top-tier

You can also keep the cheap agents local and route only the heavy reasoning to a paid provider:

```yaml
routing:
  tier_high:     { provider: anthropic, model: claude-opus-4-7, thinking_budget: auto }
  tier_standard: { provider: openai,    model: gpt-5.4-mini }
  tier_basic:    { provider: ollama_local, model: llama3.1 }
providers:
  ollama_local:
    api_key_env: OPENAI_API_KEY
    base_url: http://localhost:11434/v1
```

The framework treats any custom provider name with a `base_url` as OpenAI-compatible.

---

## Architecture (in 30 seconds)

```
        PDF ──▶ PaperExtractor ──▶ paper.txt + complexity score
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │ ConcurrentAgentRunner  │
                              │   (asyncio.gather)     │
                              └──────────┬─────────────┘
                                         │ 12 reviewers in parallel
                                         ▼
                              Coordinator ─▶ Author/Editor Summary
                                         │
                                         ▼
                                      Editor
                                         │
                                         ▼
                          Markdown · JSON · HTML · audit.jsonl
                          (all under output/<run_id>/)
```

The codebase is deliberately small and modular:

- **`orchestrator.py`** — coordinates the pipeline; doesn't know about concurrency.
- **`agent_runner.py`** — `ConcurrentAgentRunner` owns the `asyncio` machinery. Swappable for Celery / Ray / Dask without touching the orchestrator.
- **`storage.py`** — `StorageProvider` ABC + `LocalFileStorage`. Implement `S3Storage` or `PostgresStorage` once; everything else keeps working.
- **`providers/`** — one module per vendor (`OpenAI`, `Anthropic`, `Google`, OpenAI-compat). Each implements a uniform `LLMProvider` interface.
- **`agents/`** — one file per role. Each defines `KEY`, `NAME`, `INSTRUCTIONS`, `SCHEMA`, `base_complexity`. Adding a 13th reviewer is a 30-line file.
- **`schemas.py`** — `pydantic` models. Every LLM call returns a validated instance, not a parsed string.
- **`external/`** — OpenAlex (citations), statcheck (R subprocess).

If you read one file to understand the project, read [`agentic_paper/orchestrator.py`](agentic_paper/orchestrator.py). It's ~570 lines and reads like the table of contents of this README.

---

## What's in the run directory

After `agentic-paper paper.pdf` finishes, `output_paper_review/<run_id>/` contains:

```
audit.jsonl              ← one JSON row per LLM call (12 fields)
paper.txt                ← extracted text (kept for retry-failed-agents)
paper_info.json          ← title / authors / abstract / detected sections
review_<agent>.txt       ← every reviewer's validated, structured verdict
review_report_*.md       ← the human-readable report
review_results_*.json    ← machine-readable bundle (incl. routing + audit summary)
executive_summary_*.md   ← one-page TL;DR
dashboard_*.html         ← stand-alone styled report (no server needed)
prompts/<agent>.txt      ← exact prompt sent — full prompt + context dump
responses/<agent>.json   ← raw response payload from the provider
paper_review_system.log  ← debug log of the whole run
```

This is the reproducibility bundle. Hand it off when a journal asks *"how was this assessment produced?"* and the answer is *one tarball*.

---

## Reproducibility & determinism

```bash
agentic-paper paper.pdf --seed 42
```

The seed is forwarded to every provider that supports it:
- **OpenAI** — `seed=N` on Responses + Chat Completions.
- **Google Gemini** — `GenerateContentConfig.seed=N`.
- **Anthropic** — recorded in audit but not propagated (the Messages API doesn't expose a seed yet); pair with `temperature: 0` for maximal stability.

Cost, latency, and token counts for every call are queryable from `audit.jsonl` with one `jq` command — no separate observability stack required.

---

## Limitations (honest)

Things `Agentic_Paper` does **not** do:

- **Substitute for human peer review.** It surfaces mechanical issues — internal inconsistencies, citation gaps, statistical misreporting — faster than a tired human reviewer. It does not have taste, domain depth in *your* niche, or knowledge of journal-specific norms.
- **Inspect figures, tables, or equations rendered as images.** Only text is parsed (pdfplumber + heuristics).
- **Fact-check beyond citations.** No PubMed / arXiv / Semantic Scholar grounding — only OpenAlex resolution of explicit references.
- **Multi-paper synthesis.** One paper per run; use a shell loop for batch.
- **Translate.** Non-English papers technically work but the reviewer prompts assume an English peer-review register.

---

## Development

```bash
git clone https://github.com/albertogerli/Agentic_Paper.git
cd Agentic_Paper
pip install -e ".[dev,web]"
pytest -q --cov=agentic_paper --cov-fail-under=60
```

224 tests, ~74 % line coverage, CI on Python 3.10 / 3.11 / 3.12.

PRs welcome — especially: new local-model recipes, new reviewer roles, S3/Postgres `StorageProvider` implementations, non-English prompt packs.

---

## Citing

If `Agentic_Paper` contributes to research output, please cite:

```bibtex
@software{gerli_agentic_paper_2026,
  author    = {Gerli, Alberto G.},
  title     = {Agentic\_Paper: A multi-agent, multi-provider, structured-output
               peer-review pipeline for scientific manuscripts},
  year      = {2026},
  url       = {https://github.com/albertogerli/Agentic_Paper},
  version   = {2.0.0}
}
```

---

## License

[MIT](LICENSE). Use it, fork it, ship it.

## Contact

- **Issues / PRs**: <https://github.com/albertogerli/Agentic_Paper/issues>
- **Email**: <alberto@albertogerli.it>
- **Workshop**: Physalia 2026 — *Agentic Workflows for Scientific Reviewing*
