# Changelog

All notable changes to the Multi-Agent Paper Review System.

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

