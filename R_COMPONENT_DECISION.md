# `agentic_paper_R/` — Fate Decision

> Investigation deliverable for Prompt 7 of the Physalia 2026 refactor.
> No files removed in this step — the user picks A / B / C below.

## 1. What is actually in `agentic_paper_R/`

```
agentic_paper_R/
├── R/
│   ├── utils.R         160 lines   — logger setup, with_retry, hash_message, file helpers
│   ├── config.R         99 lines   — Config R6 class (YAML loader, validate)
│   ├── agents.R        409 lines   — Agent / AsyncAgent / CachingAsyncAgent R6 classes
│   └── file_manager.R  544 lines   — FileManager R6 class
└── config/config.yaml   18 lines   — example config
```

Total: **1,230 lines** of R-flavoured text.

### 1a. File-by-file findings

| File | Status | What it does today |
|---|---|---|
| `R/utils.R` | **Clean R** | Logger config via `logger`, retry-with-backoff (`with_retry`), MD5 message hashing (`hash_message`), safe `list.files`, `read_file_with_encodings`. Parses; would source. Self-contained. |
| `R/config.R` | **Clean R** | `Config <- R6Class(...)` with the same field set as the v1 Python `Config` dataclass, YAML reader, `validate()` that raises on missing API key. Parses; would source. References `setup_logging` and `log_*` from utils.R. |
| `R/agents.R` | **Partially un-parseable** | Lines 1-190: valid `Agent`, `AsyncAgent`, `CachingAsyncAgent` R6 classes wrapping `httr2`/`jsonlite` against OpenAI Chat Completions. **Line 191** introduces a closing ```` ``` ```` fence followed by 218 lines of natural-language narration ("**Note:** I've made the `arun` in `AsyncAgent`…") and the inlined source of two further files (`paper_info.R`, `file_manager.R`) that should have been written out separately. **`source("R/agents.R")` would fail**: R does not accept Markdown code-fence backticks at top level. |
| `R/file_manager.R` | **Partially un-parseable** | Lines 1-149: valid `FileManager` R6 class (PDF read via `pdftools`, JSON/text save, save_review, get_reviews). **Line 150**: closing fence + 387 lines of narration and the inlined source of `paper_analyzer.R`. Same `source()` failure as above. |
| `R/paper_info.R` | **Missing** | Referenced as "**5. `R/paper_info.R`**" inside agents.R narration but never extracted to its own file. The `PaperInfo` R6 class only exists as commentary. |
| `R/paper_analyzer.R` | **Missing** | Same situation — referenced inside file_manager.R narration. Crucial class (regex section detection, AI extraction) never exists as a file. |
| `R/orchestrator.R` / `R/main.R` | **Missing** | No entry point. There is no `Rscript` you can call to actually run a review. |
| `config/config.yaml` | **Clean YAML** | Model defaults: `o3 / gpt-4.1 / gpt-4.1-mini`. Disagrees with the Python config defaults. |

### 1b. Is the Python pipeline calling it?

**No.** Grepping the whole Python tree for `agentic_paper_R`, `reticulate`, `Rscript`, `subprocess` invocations against `.R`, or `os.system` against R returns **zero results**. The two sides have never communicated.

```bash
$ grep -rn 'agentic_paper_R\|reticulate\|Rscript\|subprocess.*\.R' agentic_paper/ main.py
(no output)
```

### 1c. Is it still working given the v2 Python side?

It was never working as an integrated component. As of v2 (Prompt 6) the gap is wider:
- The R config tracks `o3 / gpt-4.1` model names; Python uses `gpt-5.4-mini` and the new multi-vendor catalog.
- The R `Agent` hits OpenAI Chat Completions directly — no Responses API, no Anthropic, no Gemini, no thinking budget, no structured outputs.
- The R `FileManager` writes flat-file reviews; the Python v2 writes typed `AnnotatedReview` JSON under `output/<run_id>/`.

In other words, even if you fixed the unparseable files, the R side is feature-parity with the **v1.0 (pre-refactor)** Python — a code path the rest of the project has already moved past.

---

## 2. Honest reading of how this came to be

`agents.R` line 192 onward and `file_manager.R` line 150 onward read exactly like a chat transcript that was pasted verbatim into the file: explanatory paragraphs ("I've made the `arun` in `AsyncAgent` call `super$run()` for now"), section headers ("**5. `R/paper_info.R`**"), and tri-backtick fences delimiting code blocks. The intent was clearly to split this into multiple files; the split never happened.

This is not a defensible "R port"; it's an AI generation artifact that was committed before it was ready. Treating it as a salvageable feature would be sunk-cost reasoning.

---

## 3. The three options, evaluated

### Option A — Integrate properly (reticulate or subprocess)

**Cost (realistic):** ~1 sprint (3-5 dev days):
- Decide interop model: `reticulate` (in-process; needs R installed in every venv) vs `Rscript` subprocess (cleaner; needs only an `Rscript` binary on PATH). Recommend **subprocess** to keep CI simple.
- Write a small R script that the orchestrator calls with the paper text + extracted statistics. Output: structured JSON the Python side validates against a new pydantic schema.
- Add R to the GitHub Actions matrix (`r-lib/actions/setup-r@v2`).
- Decide what the R component actually *does*. The genuinely high-value answer: **statcheck-style p-value consistency checking** (cross-check reported `p`-values against test statistic + df, flag arithmetic errors). LLMs are bad at this; the [`statcheck`](https://cran.r-project.org/package=statcheck) package is best-in-class. This would be a **genuinely unique** feature versus MARG / PAK4L / ChatGPT-as-reviewer.

**Value:** High *if and only if* you commit to the statcheck angle. Mediocre if A means "make the existing R reimplementation work" — that adds maintenance burden without new capability.

**Risk:** Most of the existing R code is unrelated to statcheck. Adopting A means **discarding** ~80% of `agentic_paper_R/` and writing the new statcheck integration from scratch. So in practice, A is closer to "C + write something new".

### Option B — Spin out into `Agentic_Paper_R` sister repo

**Cost:** ~2 hours (extract the two valid files, fix the parseable subset, add a minimal README and `LICENSE`, push to a new repo, link from this README).

**Value:** Low. The extracted code is still feature-incomplete (no orchestrator, no main, no paper_info / paper_analyzer files) and feature-behind (no v2 schema, multi-provider, audit log). A sister repo would advertise a port that doesn't run.

**Risk:** Carrying public dead code in a separate repo is worse than carrying it in a folder: it implies maintenance.

### Option C — Delete + CHANGELOG note

**Cost:** ~10 minutes.
- `git rm -r agentic_paper_R/`
- Add a `## [2.0.0]` entry to CHANGELOG.md noting the removal with rationale.
- One line in the README's Roadmap mentioning a future R-side statistical-sanity component (no promise of timeline).

**Value:** Truth-in-advertising. The repo stops shipping ~1,230 lines of code that does not work, was never wired up, and is now feature-behind.

**Risk:** Loses optionality. Mitigated by the fact that the optionality was on broken code; the *idea* of an R component is recoverable from git history and from this document.

---

## 4. Recommendation

**Option C now. Option A later, as a clean new feature, only if you commit to the statcheck angle.**

Reasoning:
1. The existing R files are an AI-generation artifact, not a port. There is no "fix"; there is at best "rewrite". The longer they sit in `main`, the more they erode trust in the rest of the repo.
2. Option B (sister repo) is the worst of all worlds — it relocates the broken code instead of removing it, and it implies an open-source maintenance commitment for code nobody runs.
3. Option A is the right *eventual* move, but only as a fresh component built against the v2 orchestrator. The honest version of A is: "delete the existing R code, and when the Physalia roadmap genuinely needs statcheck integration, build a tiny `agentic_paper/checkers/statcheck_subprocess.py` plus a 30-line `checkers/statcheck.R` that wraps the `statcheck` package."

### Suggested next steps (only if you pick C)

1. `git rm -r agentic_paper_R/`
2. CHANGELOG entry:
   ```markdown
   ### Removed
   - `agentic_paper_R/`: the v1 R port was an AI-generation artifact (~1.2k lines,
     two files un-parseable, no orchestrator, never wired to the Python pipeline).
     Removed in favour of building a focused R-statcheck integration as a clean
     new component when the roadmap calls for it. See `R_COMPONENT_DECISION.md`
     in the git history for the full investigation.
   ```
3. Update README's Roadmap with a single bullet:
   ```markdown
   - R-side statistical sanity checking (statcheck) as a subprocess-invoked
     companion. **Not committed to a timeline.**
   ```
4. Keep this `R_COMPONENT_DECISION.md` file *in git history* so the rationale is preserved; safe to delete from `main` after the user acknowledges the decision.

### If you pick A or B instead

- **A**: open a tracking issue "R-statcheck integration", scope to subprocess (not reticulate), pick the cheapest probe (statcheck on a PDF's extracted text), and commit only after the prototype passes a real paper end-to-end.
- **B**: don't bother fixing the unparseable files in this repo before extracting. Move them as-is to `Agentic_Paper_R`, mark that repo "WIP, not for use", and link from this README's Roadmap. Accept that nobody will fork it.

---

*Generated as Prompt 7 of the Physalia 2026 refactor sequence. No files moved, no files deleted.*
