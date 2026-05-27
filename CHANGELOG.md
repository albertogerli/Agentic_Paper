# Changelog

All notable changes to the Multi-Agent Paper Review System.

## [2.1.0] - 2026-05-27

### Added
- **Auto-Mode for missing API keys**: when the BYOK form is submitted with only a
  subset of the keys a routing profile expects, every tier pointing at an
  unavailable provider is transparently remapped to a per-tier-equivalent model
  on a provider the user *did* supply (e.g. `tier_high: anthropic/claude-opus-4-7`
  → `google/gemini-3-pro` when only `GOOGLE_API_KEY` is given). Implemented in
  `routing.apply_auto_mode(routing, available_providers) → (RoutingConfig, list[str])`.
  Original input routing is never mutated. `thinking_budget` is preserved across
  the remap so reasoning-tier intensity carries over.
- **Auto-Mode UI banner**: a prominent yellow banner at the top of the run page
  lists each tier remap (`Anthropic not available → mapped High tier from
  anthropic/claude-sonnet-4-6 to google/gemini-3-pro`) so the user sees exactly
  what changed before the agents finish.
- `RunStatus.auto_mode_warnings: list[str]` carries the warnings from
  `_build_user_config` through `RunRegistry.start()` to the template renderer.
- `tests/test_auto_mode.py` with 8 unit tests covering no-op, full-remap,
  selective-remap, immutability, and thinking-budget preservation cases.

### Changed
- `web/server._build_user_config()` now returns `tuple[Config, list[str]]`
  instead of just `Config`. Auto-Mode only triggers when at least one BYOK key
  is supplied; runs that fall back to the server-side config are left alone.
  When the user explicitly picks a primary (provider, model) pair, that wins
  and auto-mode warnings are dropped.

## [2.0.0] - 2026-05-27

First public PyPI release: `pip install agentic-paper`. Major rewrite of the
v1 monolith into a typed, modular, multi-provider, fully-audited pipeline.
See README for the user-facing pitch; highlights below.

### Added
- Multi-provider routing (OpenAI / Anthropic / Google / OpenAI-compat local).
- Structured outputs via pydantic models for every reviewer.
- `StorageProvider` ABC + `LocalFileStorage` (S3 / DB ready).
- `ConcurrentAgentRunner` isolating asyncio fan-out from orchestrator logic.
- OpenAlex citation validator (no key required).
- R `statcheck` subprocess integration for numerical p-value sanity checking.
- FastAPI + HTMX web UI with live thinking-delta SSE stream.
- BYOK form (keys never written to disk, never logged).
- Routing profiles `max` / `std` / `quick`.
- Reproducibility: `--seed` flag forwarded to OpenAI + Google; per-call
  `audit.jsonl` row (12 fields).
- Retry-failed-agents endpoint.
- Compare-versions: v1/v2 diff feeds a Revision Assessor agent.
- 224-test pytest suite, CI on Python 3.10/3.11/3.12.

### Removed
- v1 R port (`agentic_paper_R/`) — was non-functional AI-generation artefact.

## [Unreleased] — v2.x refactor

### Removed
- **`agentic_paper_R/`**: the v1 R port (~1,230 lines across `agents.R`, `config.R`,
  `file_manager.R`, `utils.R`, and an example config) was an AI-generation artifact
  rather than a working component. Two of the four `.R` files were partially
  un-parseable (Markdown narration + code fences pasted inline); two further files
  referenced in the commentary (`paper_info.R`, `paper_analyzer.R`) were never
  extracted; there was no orchestrator entry point; and grepping the Python tree
  for `agentic_paper_R`, `reticulate`, or `Rscript` returned zero matches — the
  two sides had never communicated. See `R_COMPONENT_DECISION.md` for the full
  investigation. A clean R-side statistical-sanity component (statcheck-style
  p-value consistency checking) is on the roadmap but not committed to a timeline.

## [2.0.0] - 2025-11-04

### 🎯 Major Changes

#### Fixed Critical Bugs
- **Removed unsupported `reasoning` parameter**: GPT-5 API does not support the `reasoning` parameter. All references have been removed from agent initialization and API calls.
- **Fixed temperature values**: Updated all temperature settings to 1.0 (the only value supported by GPT-5).
- **Improved model selection algorithm**: Adjusted thresholds and weights to ensure proper use of GPT-5 for complex tasks:
  - Changed weight distribution: 40% paper complexity + 60% task complexity (previously 60%/40%)
  - Lowered threshold for GPT-5 from 0.75 to 0.65
  - Lowered threshold for GPT-5-mini from 0.5 to 0.45

#### Model Selection Improvements
The system now properly selects models based on task complexity:
- **GPT-5** (powerful): Used for complexity scores ≥ 0.65
  - Methodology Expert
  - Results Analyst
  - Contradiction Checker
  - Coordinator
  - Editor
- **GPT-5-Mini** (standard): Used for complexity scores ≥ 0.45
  - Literature Expert
  - Impact & Innovation Analyst
  - Hallucination Detector
  - Ethics & Integrity Reviewer
- **GPT-5-Nano** (basic): Used for complexity scores < 0.45
  - Structure & Clarity Reviewer
  - AI Origin Detector (when paper is simple)

### 🌍 Internationalization
- **Full English translation**: All code comments, log messages, docstrings, and instructions translated to English for GitHub publication
- **Professional documentation**: Added comprehensive README, requirements, LICENSE, and example configuration files

### 🔧 Configuration Updates
- Updated default output directory from `output_revisione_paper` to `output_paper_review`
- Added detailed configuration examples with comments
- Improved logging messages for clarity

### 📚 Documentation
- Added comprehensive README.md with:
  - Feature overview
  - Quick start guide
  - Architecture diagrams
  - Usage examples
  - Customization guide
  - Roadmap
- Created requirements.txt with all dependencies
- Added .gitignore for Python projects
- Included MIT License
- Created config.example.yaml with detailed comments

### 🚀 Performance
- Maintained prompt caching for up to 87.5% cost reduction
- Optimized parallel agent execution
- Enhanced error handling and retry logic

### 🏗️ Code Quality
- Improved code structure and organization
- Enhanced type hints and documentation
- Better error messages and logging
- No linter errors

## [1.0.0] - Previous Version

### Initial Features
- Multi-agent review system with 9 specialized agents
- Support for PDF and text files
- Markdown, JSON, and HTML report generation
- Parallel processing with asyncio
- Prompt caching support
- Dynamic model selection

---

## Migration Guide from 1.0 to 2.0

### Breaking Changes
1. **API Changes**: Removed `reasoning_effort` parameter from Agent initialization
2. **Temperature Changes**: All temperatures must be set to 1.0
3. **Configuration**: Renamed default output directory

### How to Migrate
1. Update your config.yaml:
   ```yaml
   # Remove any reasoning_effort settings
   # Update temperatures to 1.0
   temperature_methodology: 1.0
   temperature_results: 1.0
   # ... (all others to 1.0)
   ```

2. If you have custom agents, update initialization:
   ```python
   # Old (v1.0)
   agent = Agent(
       name="My_Agent",
       instructions="...",
       model="gpt-5",
       temperature=0.7,
       reasoning_effort="high"  # Remove this
   )
   
   # New (v2.0)
   agent = Agent(
       name="My_Agent",
       instructions="...",
       model="gpt-5",
       temperature=1.0  # Must be 1.0
   )
   ```

3. Update environment:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

### What Stays the Same
- Command-line interface
- Input file formats (PDF, TXT)
- Output formats (Markdown, JSON, HTML)
- Agent roles and responsibilities
- Overall workflow and architecture

