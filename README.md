````markdown
# Agentic Paper Review System (APRS)

A self-contained **multi-agent pipeline** that performs an end-to-end peer-review of scientific manuscripts by orchestrating several specialised reviewers—each powered directly by the OpenAI Chat Completions API.  
No external “agents” framework is required.

APRS accepts both plain text and PDF manuscripts. When a `.pdf` is provided, the text is automatically extracted using **pdfplumber**.

---

## Key Features

| Agent | Focus | Model (default) |
|-------|-------|-----------------|
| **Methodology Expert** | Experimental design, statistical rigour | `o3` |
| **Results Analyst** | Data analysis, figures & tables | `o3` |
| **Literature Expert** | Related work & citations | `gpt-4.1` |
| **Structure & Clarity Reviewer** | Logical flow, readability | `gpt-4.1-mini` |
| **Impact & Innovation Analyst** | Novelty, real-world impact | `gpt-4.1` |
| **Contradiction Checker** | Inconsistencies & unsupported claims | `o3` |
| **Ethics & Integrity Reviewer** | Research ethics, transparency | `gpt-4.1` |
| **AI Origin Detector** | Likelihood text was AI-generated | `gpt-4.1` |
| **Hallucination Detector** | Unsupported statements | `gpt-4o-2024-05-13` |
| **Review Coordinator** | Synthesises all reviews | `o3` |
| **Journal Editor** | Final editorial decision | `gpt-4.1` |

Behind the scenes agents run concurrently using Python `asyncio`; results are cached to speed up repeated runs.

Output includes:

* Individual reviews (`review_<agent>.txt`)
* Coordinator synthesis
* Final editorial decision
* Rich **Markdown report**, **executive summary**, and full **JSON** dump
  (all saved inside the chosen `output_dir`).
* Lightweight HTML dashboard and health report

---

## Quick Start

```bash
git clone https://github.com/<your-org>/agentic_paper.git
cd agentic_paper
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
````

### 1 · Add your OpenAI key

```bash
export OPENAI_API_KEY="sk-..."
```

### 2 · (Optional) tweak **`config.yaml`**

```yaml
# examples only – every field is optional
model_powerful: o3
max_parallel_agents: 4
temperature_methodology: 0.7
output_dir: output_reviews
```

### 3 · Run the review

```bash
python main.py path/to/paper.pdf \
  --config config.yaml        \  # optional
  --output-dir my_results     \  # optional
  --log-level DEBUG              # optional
```

Logs are streamed to console *and* `paper_review_system.log`.

---

## Output Structure (default)

```
output_reviews/
├── paper_info.json
├── review_Methodology_Expert.txt
├── review_Results_Analyst.txt
├── … (other reviewers)
├── review_coordinator.txt
├── review_editor.txt
├── review_report_YYYYMMDD_HHMMSS.md
├── executive_summary_YYYYMMDD_HHMMSS.md
├── dashboard_YYYYMMDD_HHMMSS.html
└── review_results_YYYYMMDD_HHMMSS.json
```

---

## Command-line Options

| Flag           | Default                  | Description                                              |
| -------------- | ------------------------ | -------------------------------------------------------- |
| `paper_path`   | –                        | Path to the paper file (`.txt` or `.pdf`). |
| `--config`     | `config.yaml`            | YAML overrides for any `Config` field.                   |
| `--output-dir` | `output_revisione_paper` | Where results are written.                               |
| `--log-level`  | `INFO`                   | `DEBUG`, `INFO`, `WARNING`, `ERROR`.                     |

---

## Dependencies

* Python 3.10+
* `openai`, `tenacity`, `pyyaml`, `pdfplumber`, `aiohttp`

All pinned in **`requirements.txt`**.

---

## Limitations & Future Work

* Relies on accurate plaintext extraction; complex PDFs may require manual cleanup.
* Very large papers (> 25 000 chars) may hit context limits—split or summarise if needed.
* “AI Origin Detector” is heuristic, not definitive.
* Batch processing is supported but not fully optimized for very large numbers of agents.

Contributions via pull request are welcome.

---

## License

MIT License © 2025 \[Your Name / Organisation]

```
```
