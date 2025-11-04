# 🔬 Multi-Agent Scientific Paper Review System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI GPT-5](https://img.shields.io/badge/OpenAI-GPT--5-brightgreen.svg)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, AI-powered peer review system that leverages multiple specialized GPT-5 agents to provide comprehensive, professional scientific paper evaluations. Designed to simulate a real academic review committee with expert agents analyzing different aspects of research manuscripts.

## 🌟 Features

### 🤖 Multi-Agent Architecture
- **9 Specialized Reviewer Agents**: Each agent focuses on a specific review dimension
- **Intelligent Model Selection**: Automatically assigns GPT-5, GPT-5-Mini, or GPT-5-Nano based on task complexity
- **Parallel Processing**: Concurrent review execution for optimal performance
- **Coordinator & Editor**: Synthesizes reviews and provides editorial decisions

### 📊 Comprehensive Review Dimensions

| Agent | Focus Area | Complexity |
|-------|-----------|------------|
| **Methodology Expert** | Experimental design, statistical rigor, reproducibility | High |
| **Results Analyst** | Data analysis, interpretation, visualization quality | High |
| **Literature Expert** | Contextualization, citations, field positioning | Medium-High |
| **Structure & Clarity Reviewer** | Organization, readability, logical flow | Medium |
| **Impact & Innovation Analyst** | Novelty, significance, potential applications | High |
| **Contradiction Checker** | Logical consistency, internal contradictions | High |
| **Ethics & Integrity Reviewer** | Research ethics, transparency, bias assessment | Medium |
| **AI Origin Detector** | Identifies AI-generated content patterns | Medium |
| **Hallucination Detector** | Fact-checking, citation verification | Medium-High |

### ⚡ Performance Optimizations
- **Prompt Caching**: Up to 87.5% cost reduction on repeated API calls
- **Adaptive Model Selection**: Task-complexity scoring algorithm (40% paper complexity + 60% task complexity)
- **Async Processing**: Concurrent agent execution with proper error handling
- **Retry Logic**: Exponential backoff for API reliability

### 📁 Multiple Output Formats
- **Markdown Reports**: Detailed, structured review reports
- **JSON Export**: Machine-readable results for further processing
- **HTML Dashboard**: Interactive visualization with Tailwind CSS
- **Executive Summary**: High-level overview for quick assessment

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher required
python --version

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies

```txt
openai>=1.0.0
pdfplumber>=0.7.0
tenacity>=8.2.0
pyyaml>=6.0
aiohttp>=3.9.0
```

### API Key Setup

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### Basic Usage

```bash
# Review a paper (PDF or text file)
python main.py path/to/paper.pdf

# With custom configuration
python main.py path/to/paper.pdf --config config.yaml

# With custom output directory
python main.py path/to/paper.pdf --output-dir ./custom_output

# With debug logging
python main.py path/to/paper.pdf --log-level DEBUG
```

### Configuration File (Optional)

Create a `config.yaml` file to customize behavior:

```yaml
# Model Configuration
model_powerful: "gpt-5"
model_standard: "gpt-5-mini"
model_basic: "gpt-5-nano"

# Output Settings
output_dir: "output_paper_review"

# Performance Settings
max_parallel_agents: 6
agent_timeout: 600

# API Settings
use_prompt_caching: true
max_output_tokens: 16000

# Temperature (GPT-5 only supports 1.0)
temperature_methodology: 1.0
temperature_results: 1.0
temperature_literature: 1.0
temperature_structure: 1.0
temperature_impact: 1.0
temperature_contradiction: 1.0
temperature_ethics: 1.0
temperature_coordinator: 1.0
temperature_editor: 1.0
temperature_ai_origin: 1.0
temperature_hallucination: 1.0
```

## 📖 How It Works

### 1. Paper Analysis
```python
# Extract paper metadata and structure
paper_info = analyzer.extract_info(paper_text)
# → Title, Authors, Abstract, Sections
```

### 2. Complexity Assessment
```python
# Evaluate paper complexity (0.0 - 1.0)
complexity_score = await assess_paper_complexity(paper_text)
# → Determines appropriate model selection for each agent
```

### 3. Parallel Review Execution
```python
# Launch all review agents concurrently
reviews = await batch_process_agents(agent_list, paper_text)
# → Methodology, Results, Literature, etc.
```

### 4. Coordinator Synthesis
```python
# Aggregate and synthesize all reviews
coordinator_assessment = coordinator.run(all_reviews)
# → Comprehensive overview with consensus/disagreement analysis
```

### 5. Editorial Decision
```python
# Final publication recommendation
editor_decision = editor.run(coordinator_assessment)
# → Accept / Minor Revisions / Major Revisions / Reject
```

### 6. Report Generation
```python
# Multiple output formats
generate_reports(results)
# → Markdown, JSON, HTML Dashboard, Executive Summary
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Review Orchestrator                          │
│  • Manages workflow                                             │
│  • Coordinates agents                                           │
│  • Handles errors & retries                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
         Paper Analyzer              Agent Factory
         • Extract info              • Create agents
         • Assess complexity         • Select models
              │                             │
              └──────────────┬──────────────┘
                             │
              ┌──────────────┴──────────────────────────────┐
              │         Parallel Agent Execution            │
              ├─────────────────────────────────────────────┤
              │  Methodology │ Results │ Literature │ ...   │
              └──────────────┬──────────────────────────────┘
                             │
                      Coordinator Agent
                      • Synthesize reviews
                      • Identify consensus
                             │
                       Editor Agent
                       • Final decision
                       • Editorial guidance
                             │
                      Report Generation
                      • Markdown, JSON, HTML
```

## 💡 Advanced Features

### Dynamic Model Selection

The system uses a sophisticated algorithm to select the optimal model for each task:

```python
final_score = (paper_complexity * 0.4) + (task_complexity * 0.6)

if final_score >= 0.65:
    model = "gpt-5"           # Complex tasks
elif final_score >= 0.45:
    model = "gpt-5-mini"      # Moderate tasks
else:
    model = "gpt-5-nano"      # Simple tasks
```

### Prompt Caching

Leverages OpenAI's prompt caching for cost efficiency:

```python
messages.append({
    "role": "user",
    "content": paper_text,
    "cache_control": {"type": "ephemeral"}  # Cache the paper content
})
```

### Error Handling & Retries

Robust retry mechanism with exponential backoff:

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=120)
)
def run(self, message: str) -> str:
    # Agent execution with automatic retries
```

## 📊 Output Examples

### Markdown Report Structure
```markdown
# Peer Review Report

## Paper Information
**Title:** [Paper Title]
**Authors:** [Author List]
**Abstract:** [Abstract Text]

## Editorial Decision
[Accept/Minor Revisions/Major Revisions/Reject]

## Coordinator Assessment
[Synthesis of all reviews]

## Detailed Reviews
### Methodology Expert Review
[Detailed methodology analysis]

### Results Analyst Review
[Statistical analysis evaluation]
...
```

### HTML Dashboard
- Modern, responsive design with Tailwind CSS
- Review statistics and metrics
- Color-coded editorial decision
- Expandable review sections
- Professional typography and layout

## 🔧 Customization

### Creating Custom Agents

```python
def create_custom_agent(self) -> Agent:
    return Agent(
        name="Custom_Reviewer",
        instructions="""Your custom review instructions...""",
        model=self._determine_model_for_agent("custom"),
        temperature=1.0,
        max_output_tokens=self.config.max_output_tokens,
        use_caching=self.config.use_prompt_caching
    )
```

### Adjusting Complexity Thresholds

```python
# In AgentFactory class
AGENT_BASE_COMPLEXITY = {
    "custom_agent": 0.8,  # High complexity task
    # Add your custom agent complexity scores
}
```

## 📈 Performance Metrics

- **Average Review Time**: 2-5 minutes (depending on paper length and complexity)
- **Cost Optimization**: Up to 87.5% reduction with prompt caching
- **Concurrent Processing**: 6 agents run simultaneously
- **Token Efficiency**: Adaptive model selection minimizes costs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/paper-review-system.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with OpenAI's GPT-5 models
- Inspired by academic peer review best practices
- Designed for researchers, journal editors, and academic institutions

## 📞 Contact

For questions, suggestions, or collaboration opportunities:
- GitHub Issues: [Create an issue](https://github.com/yourusername/paper-review-system/issues)
- Email: your.email@example.com

## 🗺️ Roadmap

- [ ] Support for additional languages (multilingual reviews)
- [ ] Integration with arXiv and PubMed APIs
- [ ] Batch processing for multiple papers
- [ ] Comparative analysis between papers
- [ ] Custom agent templates and presets
- [ ] Web interface for easier access
- [ ] Integration with reference management tools (Zotero, Mendeley)
- [ ] PDF annotation export
- [ ] Citation network analysis

---

**⭐ If you find this project useful, please consider giving it a star!**

