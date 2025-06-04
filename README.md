````markdown
# Agentic Paper Review System (APRS)

A **self-contained, multi-agent pipeline** that performs an end-to-end peer-review of scientific manuscripts by orchestrating several specialised reviewers—each powered *directly* by the OpenAI Chat Completions API.  
No external “agents” framework is required.

APRS accepts both plain-text **`.txt`** and **`.pdf`** manuscripts. When a PDF is provided, the text is automatically extracted with **pdfplumber**.

---

## Key Features

| Agent | Focus | Default model |
|-------|-------|---------------|
| **Methodology Expert**        | Experimental design, statistical rigour          | `o3` |
| **Results Analyst**           | Data analysis, figures & tables                  | `o3` |
| **Literature Expert**         | Related work & citations                         | `gpt-4.1` |
| **Structure & Clarity Reviewer** | Logical flow, readability                     | `gpt-4.1-mini` |
| **Impact & Innovation Analyst** | Novelty, real-world impact                    | `gpt-4.1` |
| **Contradiction Checker**     | Inconsistencies & unsupported claims             | `o3` |
| **Ethics & Integrity Reviewer** | Research ethics, transparency                 | `gpt-4.1` |
| **AI Origin Detector**        | Likelihood text was AI-generated                 | `gpt-4.1`* |
| **Hallucination Detector**    | Unsupported statements                           | `gpt-4.1` |
| **Review Coordinator**        | Synthesises all reviews                          | `o3` |
| **Journal Editor**            | Final editorial decision                         | `gpt-4.1` |

\* You can switch the AI Origin Detector to `o3` (or any other available model) via **`config.yaml`**.

* Agents run **concurrently** with `asyncio`; results are **cached** so repeated runs are fast.  
* Generates:
  * Individual reviews (`review_<agent>.txt`)
  * Coordinator synthesis
  * Editor’s decision
  * A rich **Markdown report**, **executive summary**, and full **JSON** dump
  * A modern **Tailwind-CSS HTML dashboard** and a system-health report

---

## Quick Start

```bash
git clone https://github.com/<your-org>/agentic_paper.git
cd agentic_paper
python -m venv .venv && source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
````

### 1 · Add your OpenAI key

```bash
export OPENAI_API_KEY="sk-…"
```

### 2 · (Optional) edit **`config.yaml`**

```yaml
# every field is optional – these just override defaults
model_powerful: o3
max_parallel_agents: 3          # 1–3 is usually enough
agent_timeout: 300              # seconds per reviewer
output_dir: output_revisione_paper

# Fine-tune individual reviewers
temperature_methodology: 0.7
temperature_ai_origin: 0.7      # NEW – tweak AI-origin detector behaviour
# model_ai_origin: o3           # uncomment to upgrade that reviewer
```

### 3 · Run the review

```bash
python main.py path/to/paper.pdf \
  --config config.yaml        \  # optional
  --output-dir my_results     \  # optional
  --log-level DEBUG              # optional
```

Logs stream to console *and* `paper_review_system.log`.

---

## Output Structure (default)

```
output_revisione_paper/
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

| Flag           | Default                  | Description                            |
| -------------- | ------------------------ | -------------------------------------- |
| `paper_path`   | –                        | Path to the `.txt` or `.pdf` file.     |
| `--config`     | `config.yaml`            | YAML overrides for any `Config` field. |
| `--output-dir` | `output_revisione_paper` | Destination directory for results.     |
| `--log-level`  | `INFO`                   | `DEBUG`, `INFO`, `WARNING`, `ERROR`.   |

---

## Dependencies

* Python ≥ 3.10
* `openai`, `tenacity`, `pyyaml`, `pdfplumber`, `aiohttp`
  (all versions pinned in **`requirements.txt`**)

---

## Limitations & Future Work

* Accurate plaintext extraction is vital; highly formatted PDFs may need manual cleanup.
* Very large papers (> 25 000 chars) can exceed model context—split or summarise if necessary.
* “AI Origin Detector” uses heuristics, not definitive proof.
* Batch mode works, but isn’t yet optimised for hundreds of simultaneous papers.

Pull requests are welcome!

---

## License

MIT License © 2025 \[Your Name / Organisation]

```
```
