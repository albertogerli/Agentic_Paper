"""
Multi-Agent System for Scientific Paper Review.
Alternative version without relying on the ``agents`` framework.

This system uses the OpenAI APIs directly instead of ``agents``.
"""

import os
import json
import re
import time
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
import yaml
from openai import OpenAI, AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import lru_cache
import aiohttp
import pdfplumber

# Logging configuration
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure the logging system."""
    logger = logging.getLogger("paper_review_system")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # File handler
    file_handler = logging.FileHandler('paper_review_system.log')
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

@dataclass
class Config:
    """Centralized configuration for the system."""
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model_powerful: str = "o3"
    model_standard: str = "gpt-4.1"
    model_basic: str = "gpt-4.1-mini"
    output_dir: str = "output_revisione_paper"
    max_parallel_agents: int = 3
    agent_timeout: int = 300  # seconds
    temperature_methodology: float = 1
    temperature_results: float = 1
    temperature_literature: float = 1
    temperature_structure: float = 1
    temperature_impact: float = 1
    temperature_contradiction: float = 1
    temperature_ethics: float = 1
    temperature_coordinator: float = 1
    temperature_editor: float = 1
    temperature_ai_origin: float = 1  # New temperature for AI Origin Detector
    temperature_hallucination: float = 1
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from a YAML file."""
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            return cls(**data)
        except FileNotFoundError:
            logger.warning(f"Config file {path} not found, using defaults")
            return cls()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return cls()
    
    def validate(self) -> bool:
        """Validate the configuration."""
        if not self.api_key:
            raise ValueError("API key not configured. Set OPENAI_API_KEY environment variable.")
        return True

# Alternative implementation of the agent system
class Agent:
    """Simplified implementation of an agent using the OpenAI API."""
    
    def __init__(self, name: str, instructions: str, model: str, temperature: float = 0.7):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.temperature = temperature
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize the OpenAI client."""
        config = Config()
        if config.api_key:
            self.client = OpenAI(api_key=config.api_key)
        else:
            logger.warning("OpenAI client not initialized - no API key")
  
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
    def run(self, message: str) -> str:
        """Run the agent with the given message."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        # Verify that the message is not empty
        if not message or not message.strip():
            raise ValueError("Message content cannot be empty")
        
        try:
            # Some models (o1-preview, o1-mini) only support temperature=1
            temperature = self.temperature if self.model not in ["o1-preview", "o1-mini"] else 1
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": message}
                ],
                temperature=temperature,

                max_completion_tokens=4000
            )
            
            result = response.choices[0].message.content
            logger.info(f"Agent {self.name} completed successfully")
            return result
        
        except Exception as e:
            logger.error(f"Error in agent {self.name}: {e}")
            raise


class AsyncAgent(Agent):
    """Asynchronous version of the agent."""

    async def arun(self, message: str) -> str:
        if not message or not message.strip():
            raise ValueError("Message content cannot be empty")

        client = AsyncOpenAI(api_key=Config().api_key)
        temperature = self.temperature if self.model not in ["o1-preview", "o1-mini"] else 1
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": message}
            ],
            temperature=temperature,
            max_completion_tokens=4000,
        )
        result = response.choices[0].message.content
        logger.info(f"Async agent {self.name} completed successfully")
        return result


class CachingAsyncAgent(AsyncAgent):
    """Asynchronous agent with result caching."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: Dict[int, str] = {}

    async def arun(self, message: str) -> str:
        key = hash(message)
        if key in self._cache:
            return self._cache[key]
        result = await super().arun(message)
        self._cache[key] = result
        return result

@dataclass
class PaperInfo:
    """Structured information about the paper."""
    title: str
    authors: str
    abstract: str
    length: int
    sections: List[str]
    file_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "length": self.length,
            "sections": self.sections,
            "file_path": self.file_path
        }

class FileManager:
    """Handle file operations with error management."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_json(self, data: Any, filename: str) -> bool:
        """Save data in JSON format with error handling."""
        filepath = self.output_dir / filename
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"JSON saved: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving JSON {filepath}: {e}")
            return False
    
    def save_text(self, text: str, filename: str) -> bool:
        """Save text to a file with error handling."""
        filepath = self.output_dir / filename
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Text file saved: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving text file {filepath}: {e}")
            return False

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Return the concatenated text from all pages of a PDF."""
        if not Path(pdf_path).exists():
            logger.error(f"PDF not found: {pdf_path}")
            return ""
        text_buf = StringIO()
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text(x_tolerance=1.5, y_tolerance=1.5)
                    text_buf.write(page_text or "")
                    text_buf.write("\n\n")
            return text_buf.getvalue()
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ""
    
    def save_review(self, reviewer_name: str, review_content: str) -> str:
        """Save a review from a reviewer."""
        filename = f"review_{reviewer_name.replace(' ', '_')}.txt"
        success = self.save_text(review_content, filename)
        if success:
            return f"Review successfully saved in {filename}"
        else:
            return f"Error saving review for {reviewer_name}"
    
    def get_reviews(self) -> Dict[str, str]:
        """Retrieve all saved reviews."""
        reviews = {}
        
        if not self.output_dir.exists():
            logger.warning("Output directory does not exist")
            return reviews
        
        try:
            for filepath in self.output_dir.glob("review_*.txt"):
                reviewer_name = filepath.stem[7:].replace('_', ' ')
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        reviews[reviewer_name] = f.read()
                except Exception as e:
                    logger.error(f"Error reading review {filepath}: {e}")
        except Exception as e:
            logger.error(f"Error accessing reviews: {e}")
        
        return reviews
    
    def read_paper(self, file_path: str) -> Optional[str]:
        """Read the content of a paper handling multiple encodings."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logger.info(f"Paper read successfully with {encoding} encoding")
                return content
            except UnicodeDecodeError:
                continue
            except FileNotFoundError:
                logger.error(f"File not found: {file_path}")
                return None
            except Exception as e:
                logger.error(f"Error reading file: {e}")
                return None
        
        logger.error(f"Could not read file with any encoding: {file_path}")
        return None

class PaperAnalyzer:
    """Analyze and extract information from the paper."""

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.api_key) if config.api_key else None

    def extract_info(self, paper_text: str) -> PaperInfo:
        """
        Extract structured information from the paper.
        It first tries to use an AI model for better accuracy and falls back
        to a regex-based method if the AI fails.
        """
        info = {}
        ai_success = False

        if self.client:
            try:
                # Truncate text to avoid excessive token usage for metadata extraction
                snippet = paper_text[:15000]
                
                prompt = f"""You are an expert assistant specializing in scientific literature. Your task is to extract the Title, Authors, and Abstract from the beginning of a scientific paper.

The text of the paper is provided below. Please analyze it and return the extracted information in a valid JSON format with the following keys: "title", "authors", "abstract".

- For "title", provide the full title of the paper.
- For "authors", list all authors, separated by commas.
- For "abstract", provide the full text of the abstract.

If any piece of information cannot be found, use the value "Not Found".

--- PAPER TEXT ---
{snippet}
--- END OF TEXT ---

Return only the JSON object, without any additional comments or explanations."""
                
                response = self.client.chat.completions.create(
                    model=self.config.model_basic,  # Use a fast model
                    messages=[
                        {"role": "system", "content": "You are an expert assistant for scientific literature analysis. Your output must be a single, valid JSON object."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                
                extracted_text = response.choices[0].message.content
                info = json.loads(extracted_text)
                
                # Basic validation of AI output
                if info.get("title") and info.get("title") not in ["Not Found", "Unknown title"]:
                    logger.info("Successfully extracted paper info using AI.")
                    ai_success = True
                else:
                    logger.warning("AI extraction did not find a valid title. Falling back to regex.")

            except Exception as e:
                logger.error(f"AI-based info extraction failed: {e}. Falling back to regex.")
        
        if not ai_success:
            logger.info("Using regex-based method to extract paper info.")
            regex_info = self._extract_info_with_regex(paper_text)
            # Merge results, giving preference to regex if AI failed or returned poor results
            info["title"] = regex_info.get("title") if regex_info.get("title") != "Unknown title" else info.get("title", "Unknown title")
            info["authors"] = regex_info.get("authors") if regex_info.get("authors") != "Unknown authors" else info.get("authors", "Unknown authors")
            info["abstract"] = regex_info.get("abstract") if regex_info.get("abstract") != "Abstract not found" else info.get("abstract", "Abstract not found")

        sections = self._identify_sections(paper_text)
        
        return PaperInfo(
            title=info.get("title", "Unknown title"),
            authors=info.get("authors", "Unknown authors"),
            abstract=info.get("abstract", "Abstract not found"),
            length=len(paper_text),
            sections=sections,
            file_path=None 
        )

    def _extract_info_with_regex(self, paper_text: str) -> Dict[str, str]:
        """Extract structured information from the paper using regex."""
        # Extract title
        lines = paper_text.split('\n')
        title = next((line.strip() for line in lines if line.strip()), "Unknown title")
        
        # Search for authors with an improved pattern
        author_patterns = [
            r'(?:Authors?|by|Autori|di):\s*([^\n]+)',
            r'^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)*)',
            r'(?:^|\n)([A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+(?:,\s*[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+)*)'
        ]
        
        authors = "Unknown authors"
        for pattern in author_patterns:
            match = re.search(pattern, paper_text, re.MULTILINE)
            if match:
                authors = match.group(1).strip()
                break
        
        # Search for the abstract with an improved pattern
        abstract_pattern = r'(?:Abstract|Summary|Riassunto|Sommario)[:.\n]\s*([^\n]+(?:\n[^\n]+)*?)(?:\n\n|\n[A-Z]|\n\d+\.|$)'
        abstract_match = re.search(abstract_pattern, paper_text, re.IGNORECASE | re.DOTALL)
        abstract = abstract_match.group(1).strip() if abstract_match else "Abstract not found"
        
        return {
            "title": title,
            "authors": authors,
            "abstract": abstract
        }

    @staticmethod
    def _identify_sections(paper_text: str) -> List[str]:
        """Identify the main sections of the paper using an improved approach."""
        
        # Common standard sections in scientific papers
        standard_sections = [
            "Abstract", "Introduction", "Background", "Related Work", "Literature Review",
            "Methods", "Methodology", "Materials and Methods", "Experimental Setup",
            "Results", "Experiments", "Evaluation", "Findings",
            "Discussion", "Analysis", "Implications", 
            "Conclusion", "Conclusions", "Future Work", "Limitations",
            "References", "Bibliography", "Acknowledgments", "Appendix"
        ]
        
        sections_found = []
        lines = paper_text.split('\n')
        
        # Patterns to identify sections
        section_patterns = [
            # Numbered sections (1. Introduction, 2.1 Methods, etc.)
            (r'^(?P<num>\d+(?:\.\d+)*)\s*\.?\s+(?P<title>[A-Z][A-Za-z\s\-:]+)$', True),
            # Sections with Roman numerals (I. Introduction, II. Methods)
            (r'^(?P<num>[IVX]+(?:\.[IVX]+)*)\s*\.?\s+(?P<title>[A-Z][A-Za-z\s\-:]+)$', True),
            # Unnumbered sections in all caps (INTRODUCTION, METHODS)
            (r'^(?P<title>[A-Z][A-Z\s\-]{2,})$', False),
            # Standard sections with or without numbering
            (r'^(?:\d+\.?\s+)?(?P<title>(?:' + '|'.join(standard_sections) + r'))\s*:?\s*$', False),
            # Sections using markdown headers
            (r'^#+\s+(?P<title>.+)$', False),
        ]
        
        # Analyze line by line
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) > 100:  # Skip empty or very long lines
                continue
                
            for pattern, has_num in section_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    title = match.group('title').strip()
                    
                    # Filter titles that are too short or too long
                    if 2 < len(title) < 50:
                        # Check that it is not regular text by inspecting nearby lines
                        prev_line = lines[i-1].strip() if i > 0 else ""
                        next_line = lines[i+1].strip() if i < len(lines)-1 else ""
                        
                        # If the previous line is empty or the next one seems to start a new paragraph
                        if (not prev_line or len(prev_line) < 10 or 
                            (next_line and (next_line[0].isupper() or not next_line[0].isalpha()))):
                            
                            # Normalize the title
                            if has_num and match.group('num'):
                                section_title = f"{match.group('num')}. {title.title()}"
                            else:
                                section_title = title.title()
                            
                            # Avoid duplicates
                            if section_title not in sections_found:
                                sections_found.append(section_title)
                    break
        
        # If no sections were found, try a heuristic approach
        if len(sections_found) < 3:
            sections_found = PaperAnalyzer._identify_sections_heuristic(paper_text, standard_sections)
        
        # Limit to 20 sections and remove very similar ones
        sections_found = PaperAnalyzer._filter_similar_sections(sections_found)[:20]
        
        return sections_found

    @staticmethod
    def _identify_sections_heuristic(paper_text: str, standard_sections: List[str]) -> List[str]:
        """Heuristic approach to identify sections when pattern matching fails."""
        sections_found = []
        text_lower = paper_text.lower()
        
        # Look for the standard sections in the text
        for section in standard_sections:
            section_lower = section.lower()
            
            # Look for variants of the section
            patterns = [
                f"\n{section_lower}\n",
                f"\n{section_lower}:",
                f"\n{section_lower}.",
                f"\n1. {section_lower}",
                f"\n2. {section_lower}",
                f"\n3. {section_lower}",
                f"\n4. {section_lower}",
                f"\n5. {section_lower}",
            ]
            
            for pattern in patterns:
                if pattern in text_lower:
                    sections_found.append(section)
                    break
        
        return sections_found

    @staticmethod
    def _filter_similar_sections(sections: List[str]) -> List[str]:
        """Remove duplicate or overly similar sections."""
        filtered = []
        
        for section in sections:
            # Normalize for comparison
            section_normalized = re.sub(r'^(?:\d+\.?\d*)\s*', '', section).lower()
            
            # Verify that it is not too similar to sections already added
            is_duplicate = False
            for existing in filtered:
                existing_normalized = re.sub(r'^(?:\d+\.?\d*)\s*', '', existing).lower()
                
                # Check similarity
                if (section_normalized == existing_normalized or
                    section_normalized in existing_normalized or
                    existing_normalized in section_normalized):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(section)
        
        return filtered

class AgentFactory:
    """Factory per creare agenti con configurazioni appropriate."""
    
    # Base complexity score for each agent's task.
    # Scale: 0.0 (simple) to 1.0 (complex).
    AGENT_BASE_COMPLEXITY = {
        "methodology": 0.9,
        "results": 0.7,
        "literature": 0.6,
        "structure": 0.3,
        "impact": 0.7,
        "contradiction": 0.9,
        "ethics": 0.5,
        "ai_origin": 0.4,
        "hallucination": 0.6,
        "coordinator": 1.0, # Always high, synthesizes all reviews
        "editor": 0.8,
        "author_editor_summary": 0.8
    }
    
    def __init__(self, config: Config, paper_complexity_score: float):
        self.config = config
        self.paper_complexity_score = paper_complexity_score
        self.file_manager = FileManager(config.output_dir)

    def _determine_model_for_agent(self, agent_name: str) -> str:
        """
        Determines the best model for an agent based on task and paper complexity.
        """
        base_task_complexity = self.AGENT_BASE_COMPLEXITY.get(agent_name, 0.5)
        
        # Combine paper complexity and task complexity.
        # Weighting: 60% paper complexity, 40% task complexity.
        final_score = (self.paper_complexity_score * 0.6) + (base_task_complexity * 0.4)

        if final_score >= 0.75:
            model = self.config.model_powerful
        elif final_score >= 0.5:
            model = self.config.model_standard
        else:
            model = self.config.model_basic
            
        logger.info(f"Selected model '{model}' for agent '{agent_name}' (score: {final_score:.2f})")
        return model

    def create_methodology_agent(self) -> Agent:
        return Agent(
            name="Methodology_Expert",
            instructions="""You are an expert in scientific methodology with a PhD and extensive experience in reviewing scientific papers.
Your task is to critically evaluate the methodology of the paper, focusing on the following aspects:
1. Validity and appropriateness of the chosen methods
2. Experimental rigor and control of variables
3. Sample size and representativeness
4. Correctness of statistical analyses
5. Presence and appropriate management of controls
6. Adequacy of measures to reduce bias and confounders
7. Reproducibility of experimental procedures
8. Consistency between stated methodology and presented results

Provide a detailed analysis IN ENGLISH, highlighting methodological strengths and criticalities.
Suggest specific improvements where appropriate.
Use a constructive but rigorous approach, as you would in a high-quality peer review.

Structure your review with clear sections:
- Overview of Methodology
- Strengths
- Weaknesses and Concerns
- Specific Recommendations

End your review with: "REVIEW COMPLETED - Methodology Expert" """,
            model=self._determine_model_for_agent("methodology"),
            temperature=self.config.temperature_methodology,
        )
    
    def create_results_agent(self) -> Agent:
        return Agent(
            name="Results_Analyst",
            instructions="""You are a statistician and data analyst specializing in the critical analysis of scientific results.
Your task is to evaluate the quality of the results and data analyses in the paper, focusing on:
1. Validity and robustness of the statistical analyses used
2. Correct interpretation of results and significance
3. Completeness of data presentation (are all relevant data shown?)
4. Appropriateness of visualizations (graphs, tables, figures)
5. Presence of potential analysis or interpretation errors
6. Consistency between presented results and drawn conclusions
7. Assessment of the limitations of results and their generalizability
8. Possibility of alternative explanations for the observed phenomena

Analyze in detail the results sections, figures, and tables, identifying inconsistencies or problems.
Provide constructive criticism IN ENGLISH on how to improve the presentation and analysis of data.

Structure your review with:
- Summary of Key Results
- Statistical Analysis Assessment
- Data Presentation Quality
- Interpretation Validity
- Recommendations for Improvement

End your review with: "REVIEW COMPLETED - Results Analyst" """,
            model=self._determine_model_for_agent("results"),
            temperature=self.config.temperature_results,
        )
    
    def create_literature_agent(self) -> Agent:
        return Agent(
            name="Literature_Expert",
            instructions="""You are an expert in the specific field of study of the paper, with in-depth knowledge of the relevant literature.
Your task is to evaluate how the paper fits into the context of existing literature:
1. Completeness and relevance of the literature review
2. Identification of potential gaps in references to important works
3. Evaluation of the originality and contribution of the paper in relation to the existing field
4. Correctness of citations and representation of others' work
5. Adequate contextualization of the research problem
6. Identification of potential connections with other relevant fields or literature

Provide a balanced assessment IN ENGLISH of the paper's positioning in the research field,
suggesting additions or changes in contextualization and bibliographic references.

End your review with: "REVIEW COMPLETED - Literature Expert" """,
            model=self._determine_model_for_agent("literature"),
            temperature=self.config.temperature_literature,
        )
    
    def create_structure_agent(self) -> Agent:
        return Agent(
            name="Structure_Clarity_Reviewer",
            instructions="""You are an editor specialized in evaluating academic manuscripts for clarity and structure.
Your task is to analyze the structural and communicative aspects of the paper:
1. Logic and coherence in the overall organization of the paper
2. Clarity of the abstract and adherence to the paper's contents
3. Effectiveness of the introduction in presenting the problem and objectives
4. Logical flow between sections and paragraphs
5. Clarity and precision of scientific language used
6. Adequacy of section titles and subtitles
7. Effectiveness of conclusions in summarizing the main results
8. Presence of redundancies, digressions, or superfluous parts

Provide concrete suggestions IN ENGLISH for improving the organization and expository clarity of the paper,
indicating specific sections to restructure, condense, or expand.

End your review with: "REVIEW COMPLETED - Structure & Clarity Reviewer" """,
            model=self._determine_model_for_agent("structure"),
            temperature=self.config.temperature_structure,
        )
    
    def create_impact_agent(self) -> Agent:
        return Agent(
            name="Impact_Innovation_Analyst",
            instructions="""You are an analyst of scientific trends and innovation with experience in evaluating the potential impact of research.
Your task is to evaluate the importance, novelty, and potential impact of the paper:
1. Degree of innovation and originality of the presented ideas
2. Relevance and significance of the addressed problems
3. Potential impact in the specific field and related areas
4. Identification of possible practical applications or future implications
5. Capacity of the paper to open new research directions
6. Positioning in relation to the main challenges in the field
7. Adequacy of conclusions in communicating the value of the contribution

Offer a balanced assessment IN ENGLISH of the work's importance in the current scientific context,
considering both strengths and limitations in terms of potential impact.

End your review with: "REVIEW COMPLETED - Impact & Innovation Analyst" """,
            model=self._determine_model_for_agent("impact"),
            temperature=self.config.temperature_impact,
        )
    
    def create_contradiction_agent(self) -> Agent:
        return Agent(
            name="Contradiction_Checker",
            instructions="""You are a skeptical reviewer with excellent analytical skills and attention to detail.
Your task is to identify contradictions, inconsistencies, and logical problems in the paper:
1. Incoherencies between statements in different parts of the text
2. Contradictions between presented data and drawn conclusions
3. Claims not supported by sufficient evidence
4. Problematic implicit assumptions
5. Potential logical fallacies or reasoning errors
6. Incongruities between stated objectives and actually presented results
7. Discrepancies between figures/tables and the text describing them
8. Significant omissions that weaken the argument

Be particularly attentive and critical, reporting precisely IN ENGLISH any identified problems,
citing specific sections or passages of the paper.

If you find no contradictions or significant inconsistencies, please state "No significant contradictions or inconsistencies were found after a careful review."

End your review with: "REVIEW COMPLETED - Contradiction Checker" """,
            model=self._determine_model_for_agent("contradiction"),
            temperature=self.config.temperature_contradiction,
        )
    
    def create_ethics_agent(self) -> Agent:
        return Agent(
            name="Ethics_Integrity_Reviewer",
            instructions="""You are an expert in research ethics and scientific integrity.
Your task is to evaluate the paper from an ethical and scientific integrity perspective:
1. Compliance with ethical standards in research conduct
2. Transparency on methodology and data
3. Proper attribution of others' work (appropriate citations)
4. Disclosure of potential conflicts of interest
5. Consideration of ethical implications of results or applications
6. Respect for privacy and informed consent (if applicable)
7. Assessment of possible bias or prejudice in the research
8. Adherence to open science and reproducibility principles

Provide a balanced assessment IN ENGLISH of ethical and integrity aspects, highlighting both 
positive practices and problematic areas, with suggestions for improvements.

End your review with: "REVIEW COMPLETED - Ethics & Integrity Reviewer" """,
            model=self._determine_model_for_agent("ethics"),
            temperature=self.config.temperature_ethics,
        )
    
    def create_ai_origin_detector_agent(self) -> Agent:
        return Agent(
            name="AI_Origin_Detector",
            instructions="""You are an AI Origin Detector. Your task is to analyze the provided scientific paper text and assess the likelihood that it was written by an AI, partially or entirely. 
Focus on aspects such as:
1. Writing style (e.g., overly formal, repetitive sentence structures, unusual vocabulary choices, lack of personal voice).
2. Content consistency and depth (e.g., superficial analysis, generic statements, lack of nuanced arguments, logical fallacies common in AI text).
3. Structural patterns (e.g., predictable organization, boilerplate phrases, unnaturally smooth transitions).
4. Presence of known AI writing tells or artifacts.
5. Compare against typical human academic writing styles.

Provide a detailed analysis IN ENGLISH, outlining your findings and the reasons for your assessment. 
Conclude with an estimated likelihood (e.g., Very Low, Low, Moderate, High, Very High) that the text has significant AI-generated portions.

End your review with: "REVIEW COMPLETED - AI Origin Detector\"""",
            model=self._determine_model_for_agent("ai_origin"),
            temperature=self.config.temperature_ai_origin,
        )

    def create_hallucination_detector(self) -> Agent:
        return Agent(
            name="Hallucination_Detector",
            instructions="""You are tasked with spotting potential hallucinations in the paper. Look for:\n1. Claims lacking citations\n2. Data inconsistent with official sources\n3. Conclusions not supported by presented data\n4. Invented or malformed references\nProvide a concise report IN ENGLISH detailing any suspicious statements.""",
            model=self._determine_model_for_agent("hallucination"),
            temperature=self.config.temperature_hallucination,
        )
    
    def create_coordinator_agent(self) -> Agent:
        return Agent(
            name="Review_Coordinator",
            instructions="""You are the coordinator of the peer review process for a scientific paper.
You will receive individual reviews from multiple expert reviewers. Your task is to:
1. Review all the feedback provided by the expert reviewers
2. Identify points of consensus and disagreement among reviewers
3. Synthesize the feedback into a structured overall assessment
4. Balance criticisms and strengths for a fair evaluation
5. Produce clear final recommendations (accept/revise/reject) with rationales
6. Highlight priorities for any requested revisions

Create a comprehensive, balanced summary IN ENGLISH of all reviewer feedback,
structured in a way that would be useful for both the authors and the editor.

Your final assessment should include:
- Executive summary of the paper's strengths and weaknesses
- Methodological soundness
- Quality of results and analyses
- Relevance and literature contextualization
- Structural clarity and organization
- Innovation and potential impact
- Logical consistency 
- Ethical considerations
- Final recommendation with clear justification

End with: "COORDINATOR ASSESSMENT COMPLETED" """,
            model=self._determine_model_for_agent("coordinator"),
            temperature=self.config.temperature_coordinator,
        )
    
    def create_editor_agent(self) -> Agent:
        return Agent(
            name="Journal_Editor",
            instructions="""You are the editor of a prestigious academic journal.
Based on all reviews including the coordinator's comprehensive assessment, your task is to:
1. Evaluate the paper from an editorial perspective
2. Consider the relevance and adequacy for the journal's audience
3. Provide a final judgment on the publishability of the paper
4. Elaborate specific editorial feedback for the authors

Provide a formal editorial decision IN ENGLISH, considering the potential interest for readers
and contribution to the field. Use formal and professional language.

Your decision should be one of:
- Accept as is
- Accept with minor revisions
- Revise and resubmit (major revisions)
- Reject

Include clear justification for your decision and specific guidance for authors.

End with: "EDITORIAL DECISION COMPLETED" """,
            model=self._determine_model_for_agent("editor"),
            temperature=self.config.temperature_editor,
        )
    
    def create_author_editor_summary_agent(self) -> Agent:
        return Agent(
            name="Author_Editor_Summary_Agent",
            instructions="""You are a senior scientific reviewer and editorial consultant. Your task is to synthesize all the reviews and the coordinator's assessment of a scientific paper into two distinct sections:

1. **Review for Author and Editor**: Write a discursive, technical, and human summary of the most important points, strengths, and weaknesses that emerged from the reviews. Use a constructive, professional, and clear tone, highlighting the main issues and suggestions for improvement. This section should be suitable for both the authors and the editor.

2. **Review for Editor Only**: Write a confidential summary for the editor, focusing on critical points, structural weaknesses, ethical or originality concerns, and any aspect that requires special editorial attention. Use a technical, discursive, and human language, providing a clear and reasoned overview of the most relevant editorial issues.

Base your synthesis on the content of all reviews and the coordinator's assessment. Structure your output as follows:

---
Review for Author and Editor:
[Your summary here]
---
Review for Editor Only:
[Your summary here]
---

End with: "SUMMARY AGENT COMPLETED".
""",
            model=self._determine_model_for_agent("author_editor_summary"),
            temperature=1.0,
        )
    
    def create_all_agents(self) -> Dict[str, Agent]:
        """Create all the required agents."""
        return {
            "methodology": self.create_methodology_agent(),
            "results": self.create_results_agent(),
            "literature": self.create_literature_agent(),
            "structure": self.create_structure_agent(),
            "impact": self.create_impact_agent(),
            "contradiction": self.create_contradiction_agent(),
            "ethics": self.create_ethics_agent(),
            "ai_origin": self.create_ai_origin_detector_agent(),
            "hallucination": self.create_hallucination_detector(),
            "coordinator": self.create_coordinator_agent(),
            "editor": self.create_editor_agent(),
            "author_editor_summary": self.create_author_editor_summary_agent(),
        }

class ReviewOrchestrator:
    """Main orchestrator for the review process."""
    
    def __init__(self, config: Config):
        self.config = config
        self.file_manager = FileManager(config.output_dir)
        self.paper_analyzer = PaperAnalyzer(config)
        self.agent_factory: Optional[AgentFactory] = None
        self.agents: Dict[str, Agent] = {}
        self.client = AsyncOpenAI(api_key=config.api_key) if config.api_key else None

    async def _assess_paper_complexity(self, paper_text: str) -> float:
        """
        Rates task complexity on a scale of 0.0 to 1.0 using an AI model.
        """
        if not self.client:
            logger.warning("No OpenAI client, using default complexity.")
            return 0.5

        try:
            # Using a snippet to be efficient
            snippet = paper_text[:8000]

            prompt = f"""You are a scientific review expert. Your task is to assess the complexity of the provided scientific paper snippet.
            Consider factors like:
            - Technical jargon and lexical density
            - Conceptual depth and abstraction
            - Methodological sophistication
            - Interdisciplinarity

            Based on your assessment, provide a single complexity score from 0.0 (very simple, e.g., a high school report) to 1.0 (extremely complex, e.g., a groundbreaking theoretical physics paper).

            Return your answer as a single JSON object with one key: "complexity_score".

            --- PAPER SNIPPET ---
            {snippet}
            --- END OF SNIPPET ---
            """
            
            response = await self.client.chat.completions.create(
                model="gpt-4.1-mini", # Always use the fast model for this assessment
                messages=[
                    {"role": "system", "content": "You are a scientific complexity analyzer. Your output must be a single, valid JSON object."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            
            result = json.loads(response.choices[0].message.content)
            score = float(result.get("complexity_score", 0.5))
            
            if 0.0 <= score <= 1.0:
                logger.info(f"Assessed paper complexity score: {score:.2f}")
                return score
            else:
                logger.warning(f"Invalid complexity score received: {score}. Using default 0.5.")
                return 0.5

        except Exception as e:
            logger.error(f"Failed to assess paper complexity: {e}. Using default 0.5.")
            return 0.5

    def execute_review_process(self, paper_text: str) -> Dict[str, Any]:
        """Execute the full review process with error handling."""
        try:
            # Assess complexity to inform agent creation
            complexity_score = asyncio.run(self._assess_paper_complexity(paper_text))
            
            # Now that models are selected, create the factory and agents
            self.agent_factory = AgentFactory(self.config, complexity_score)
            self.agents = self.agent_factory.create_all_agents()

            # Extract paper information
            paper_info = self.paper_analyzer.extract_info(paper_text)
            self.file_manager.save_json(paper_info.to_dict(), "paper_info.json")
            
            logger.info("Starting multi-agent peer review process...")
            
            # Prepare initial message
            initial_message = self._prepare_initial_message(paper_info, paper_text)
            
            # Run main reviewers
            reviews = self._execute_main_reviewers(initial_message)
            
            # Run coordinator
            coordinator_review = self._execute_coordinator(reviews)
            reviews["coordinator"] = coordinator_review

            # Run author/editor summary agent
            author_editor_summary = self._execute_author_editor_summary(reviews)
            reviews["author_editor_summary"] = author_editor_summary
            
            # Run editor
            editor_decision = self._execute_editor(reviews)
            
            # Summarize results
            final_results = self._synthesize_results(paper_info, reviews, editor_decision)
            
            # Generate reports
            self._generate_reports(final_results)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Critical error in review process: {e}")
            raise
    
    def _prepare_initial_message(self, paper_info: PaperInfo, paper_text: str) -> str:
        """Prepare the initial message for the agents."""

        # Always keep the full text of the paper. If it exceeds the
        # recommended threshold for some models, only log a warning.
        display_paper_text = paper_text
        original_length = len(paper_text)

        MAX_RECOMMENDED_CHARS = 25000
        if original_length > MAX_RECOMMENDED_CHARS:
            logger.info(
                f"Paper text is {original_length} characters; this may exceed some model limits "
                f"(recommended <= {MAX_RECOMMENDED_CHARS}). Using full text as requested."
            )

        prompt_template = (
            """Paper to be analyzed:

Title: {title}
Authors: {authors}
Abstract: {abstract}

Please conduct a comprehensive and thorough review of this scientific paper.
All reviewers should provide their comments IN ENGLISH.
Each reviewer should analyze the paper from their own expert perspective.

The paper content is as follows:

{text_content}
"""
        )

        return prompt_template.format(
            title=paper_info.title,
            authors=paper_info.authors,
            abstract=paper_info.abstract,
            text_content=display_paper_text
        )
    
    def _execute_main_reviewers(self, initial_message: str) -> Dict[str, str]:
        """Run the main reviewers using asynchronous batches."""
        main_agents = [
            "methodology",
            "results",
            "literature",
            "structure",
            "impact",
            "contradiction",
            "ethics",
            "ai_origin",
            "hallucination",
        ]
        return asyncio.run(self._batch_process_agents(main_agents, initial_message))

    async def _batch_process_agents(self, agent_names: List[str], message: str) -> Dict[str, str]:
        """Execute multiple agents in parallel using asyncio."""
        tasks = []
        for name in agent_names:
            agent = self.agents.get(name)
            if not agent:
                continue
            if isinstance(agent, AsyncAgent):
                tasks.append(asyncio.create_task(agent.arun(message)))
            else:
                loop = asyncio.get_running_loop()
                tasks.append(loop.run_in_executor(None, agent.run, message))

        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        reviews: Dict[str, str] = {}
        for name, result in zip(agent_names, results_list):
            if isinstance(result, Exception):
                logger.error(f"Error in agent {name}: {result}")
                reviews[name] = f"Error during review: {result}"
            else:
                reviews[name] = result
                self.file_manager.save_review(name, result)
        return reviews
    
    def _run_agent_with_review(self, agent: Agent, message: str, agent_name: str) -> str:
        """Run an agent and save its review."""
        try:
            review = agent.run(message)
            # Save the review
            self.file_manager.save_review(agent_name, review)
            return review
        except Exception as e:
            logger.error(f"Agent execution error for {agent_name}: {e}")
            raise
    
    def _execute_coordinator(self, reviews: Dict[str, str]) -> str:
        """Run the coordinator with all reviews."""
        coordinator = self.agents.get("coordinator")
        if not coordinator:
            logger.error("Coordinator agent not found")
            return "Coordinator review not available"
        
        # Prepare message with all the reviews
        reviews_text = "\n\n".join([
            f"=== {agent_name.upper()} REVIEW ===\n{review_content}"
            for agent_name, review_content in reviews.items()
        ])
        
        coordinator_message = f"""
Here are all the expert reviews for the paper:

{reviews_text}

Please provide your comprehensive coordinator assessment based on all these reviews.
"""
        
        try:
            coordinator_review = coordinator.run(coordinator_message)
            self.file_manager.save_review("coordinator", coordinator_review)
            return coordinator_review
        except Exception as e:
            logger.error(f"Error in coordinator: {e}")
            return f"Error in coordinator assessment: {str(e)}"
    
    def _execute_author_editor_summary(self, reviews: Dict[str, str]) -> str:
        """Esegue l'agente di sintesi per autore/editor."""
        summary_agent = self.agents.get("author_editor_summary")
        if not summary_agent:
            logger.error("Author/Editor Summary agent not found")
            return "Author/Editor summary not available"
        # Prepara il messaggio con tutte le review e il coordinatore
        reviews_text = "\n\n".join([
            f"=== {agent_name.upper()} REVIEW ===\n{review_content}"
            for agent_name, review_content in reviews.items()
        ])
        summary_message = f"""
Here are all the expert reviews and the coordinator's assessment for the paper:

{reviews_text}

Please provide the two requested summaries as per your instructions.
"""
        try:
            summary = summary_agent.run(summary_message)
            self.file_manager.save_review("author_editor_summary", summary)
            return summary
        except Exception as e:
            logger.error(f"Error in author/editor summary agent: {e}")
            return f"Error in author/editor summary: {str(e)}"
    
    def _execute_editor(self, all_reviews: Dict[str, str]) -> str:
        """Run the editor to produce the final decision."""
        editor = self.agents.get("editor")
        if not editor:
            logger.error("Editor agent not found")
            return "Editorial decision not available"
        
        # Prepare message with all reviews including the coordinator
        reviews_text = "\n\n".join([
            f"=== {agent_name.upper()} REVIEW ===\n{review_content}"
            for agent_name, review_content in all_reviews.items()
        ])
        
        editor_message = f"""
Here are all the reviews including the coordinator's assessment:

{reviews_text}

Please provide your editorial decision based on all these reviews.
"""
        
        try:
            editor_decision = editor.run(editor_message)
            self.file_manager.save_review("editor", editor_decision)
            return editor_decision
        except Exception as e:
            logger.error(f"Error in editor: {e}")
            return f"Error in editorial decision: {str(e)}"
    
    def _synthesize_results(self, paper_info: PaperInfo, reviews: Dict[str, str],
                          editor_decision: str) -> Dict[str, Any]:
        """Summarize the results of the reviews."""
        return {
            "paper_info": paper_info.to_dict(),
            "reviews": reviews,
            "editor_decision": editor_decision,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "models_used": {
                    "powerful": self.config.model_powerful,
                    "standard": self.config.model_standard,
                    "basic": self.config.model_basic
                },
                "num_reviewers": len(reviews)
            }
        }
    
    def _generate_reports(self, results: Dict[str, Any]) -> None:
        """Generate reports in various formats."""
        # Report Markdown
        report_md = self._generate_markdown_report(results)
        self.file_manager.save_text(report_md, f"review_report_{datetime.now():%Y%m%d_%H%M%S}.md")
        
        # Report JSON
        self.file_manager.save_json(results, f"review_results_{datetime.now():%Y%m%d_%H%M%S}.json")
        
        # Executive summary
        summary = self._generate_executive_summary(results)
        self.file_manager.save_text(summary, f"executive_summary_{datetime.now():%Y%m%d_%H%M%S}.md")

        # Dashboard HTML
        dashboard = ReviewDashboard().generate_html_dashboard(results)
        self.file_manager.save_text(dashboard, f"dashboard_{datetime.now():%Y%m%d_%H%M%S}.html")
    
    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate a detailed report in Markdown format."""
        paper_info = results["paper_info"]
        reviews = results["reviews"]
        editor_decision = results["editor_decision"]
        
        report = f"""# Peer Review Report

**Generated:** {results['timestamp']}

## Paper Information

**Title:** {paper_info['title']}

**Authors:** {paper_info['authors']}

**Abstract:**
{paper_info['abstract']}

**Document Length:** {paper_info['length']:,} characters

**Identified Sections:** {', '.join(paper_info['sections'][:10])}

## Review Configuration

**Models Used:**
- Primary: {results['config']['models_used']['powerful']}
- Standard: {results['config']['models_used']['standard']}
- Basic: {results['config']['models_used']['basic']}

**Number of Reviewers:** {results['config']['num_reviewers']}

## Editorial Decision

{editor_decision}

## Coordinator Assessment

{reviews.get('coordinator', 'No coordinator assessment available')}

## Author & Editor Summary

{reviews.get('author_editor_summary', 'No summary available')}

## Detailed Reviews

"""
        # Aggiungi revisioni individuali
        review_order = [
            "methodology",
            "results",
            "literature",
            "structure",
            "impact",
            "contradiction",
            "ethics",
            "ai_origin",
            "hallucination",
        ]
        for agent_type in review_order:
            if agent_type in reviews:
                report += f"### {agent_type.replace('_', ' ').title()} Review\n\n"
                report += reviews[agent_type]
                report += "\n\n---\n\n"
        return report
    
    def _generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate an executive summary."""
        paper_info = results["paper_info"]
        editor_decision = results["editor_decision"]
        coordinator_assessment = results["reviews"].get("coordinator", "")
        author_editor_summary = results["reviews"].get("author_editor_summary", "")
        summary = f"""# Executive Summary

**Paper:** {paper_info['title']}
**Authors:** {paper_info['authors']}
**Review Date:** {results['timestamp']}

## Editorial Decision

{editor_decision}

## Coordinator's Overall Assessment

{coordinator_assessment}

## Author & Editor Summary

{author_editor_summary}

## Review Summary

This paper has been reviewed by 8 specialized AI agents, each focusing on different aspects of the manuscript:

1. **Methodology Expert**: Evaluated experimental design and statistical rigor
2. **Results Analyst**: Assessed data analysis and presentation
3. **Literature Expert**: Reviewed contextualization and citations
4. **Structure & Clarity Reviewer**: Analyzed organization and readability
5. **Impact & Innovation Analyst**: Evaluated novelty and potential contribution
6. **Contradiction Checker**: Identified inconsistencies and logical issues
7. **Ethics & Integrity Reviewer**: Assessed ethical compliance and transparency
8. **AI Origin Detector**: Assessed the likelihood of AI authorship

The reviews were synthesized by a Review Coordinator and evaluated by a Journal Editor for the final publication decision.

---

For the complete detailed reviews, please refer to the full report.
"""
        return summary


class ReviewDashboard:
    """Generate a well-structured and pleasant HTML dashboard."""

    def generate_html_dashboard(self, results: Dict[str, Any]) -> str:
        """Create a modern, styled HTML dashboard for review results."""
        paper = results.get("paper_info", {})
        reviews = results.get("reviews", {})
        timestamp = results.get("timestamp", "")
        editor_decision = results.get("editor_decision", "")
        
        def esc(text: str) -> str:
            import html
            return html.escape(str(text))
        
        # Extract the editor's decision (search common patterns)
        decision_class = "bg-gray-100"
        decision_icon = "📋"
        if "accept as is" in editor_decision.lower():
            decision_class = "bg-green-100 border-green-300 text-green-900"
            decision_icon = "✅"
        elif "minor revisions" in editor_decision.lower():
            decision_class = "bg-blue-100 border-blue-300 text-blue-900"
            decision_icon = "🔧"
        elif "major revisions" in editor_decision.lower():
            decision_class = "bg-yellow-100 border-yellow-300 text-yellow-900"
            decision_icon = "⚠️"
        elif "reject" in editor_decision.lower():
            decision_class = "bg-red-100 border-red-300 text-red-900"
            decision_icon = "❌"
        
        # Calculate statistics
        total_reviews = len(reviews)
        total_words = sum(len(review.split()) for review in reviews.values())
        
        # HTML with modern design
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Paper Review Dashboard - {esc(paper.get('title', 'Untitled'))}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; }}
        .gradient-bg {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .review-card {{
            transition: all 0.3s ease;
        }}
        .review-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.1);
        }}
        .progress-bar {{
            transition: width 1s ease-in-out;
        }}
        details summary {{
            cursor: pointer;
            user-select: none;
        }}
        details[open] summary {{
            margin-bottom: 1rem;
        }}
        .review-content {{
            max-height: 500px;
            overflow-y: auto;
        }}
        .review-content::-webkit-scrollbar {{
            width: 6px;
        }}
        .review-content::-webkit-scrollbar-track {{
            background: #f1f1f1;
        }}
        .review-content::-webkit-scrollbar-thumb {{
            background: #888;
            border-radius: 3px;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .fade-in {{
            animation: fadeIn 0.6s ease-out forwards;
        }}
    </style>
</head>
<body class="bg-gray-50">
    <!-- Header -->
    <div class="gradient-bg text-white">
        <div class="container mx-auto px-6 py-12">
            <h1 class="text-4xl font-bold mb-2">📚 Peer Review Dashboard</h1>
            <p class="text-purple-100">Advanced Multi-Agent Review System</p>
        </div>
    </div>
    
    <!-- Main Content -->
    <div class="container mx-auto px-6 py-8 max-w-7xl">
        <!-- Paper Info Card -->
        <div class="bg-white rounded-lg shadow-lg p-8 mb-8 fade-in">
            <h2 class="text-2xl font-semibold mb-6 flex items-center">
                <span class="bg-purple-100 text-purple-600 p-2 rounded-lg mr-3">📄</span>
                Paper Information
            </h2>
            <div class="grid md:grid-cols-2 gap-6">
                <div>
                    <h3 class="text-sm font-medium text-gray-500 uppercase tracking-wide mb-2">Title</h3>
                    <p class="text-lg font-medium text-gray-900">{esc(paper.get('title', 'Not specified'))}</p>
                </div>
                <div>
                    <h3 class="text-sm font-medium text-gray-500 uppercase tracking-wide mb-2">Authors</h3>
                    <p class="text-lg text-gray-700">{esc(paper.get('authors', 'Not specified'))}</p>
                </div>
                <div>
                    <h3 class="text-sm font-medium text-gray-500 uppercase tracking-wide mb-2">Document Length</h3>
                    <p class="text-lg text-gray-700">{paper.get('length', 0):,} characters</p>
                </div>
                <div>
                    <h3 class="text-sm font-medium text-gray-500 uppercase tracking-wide mb-2">Review Date</h3>
                    <p class="text-lg text-gray-700">{esc(timestamp)}</p>
                </div>
            </div>
            {f"""
            <div class="mt-6">
                <h3 class="text-sm font-medium text-gray-500 uppercase tracking-wide mb-2">Abstract</h3>
                <div class="relative">
                    <div id="abstract-content" class="text-gray-700 leading-relaxed overflow-hidden transition-all duration-300" style="max-height: 150px;">
                        <p>{esc(paper.get('abstract', 'Not available'))}</p>
                    </div>
                    <div id="abstract-gradient" class="absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-white to-transparent pointer-events-none"></div>
                    <button id="abstract-toggle" class="mt-2 text-purple-600 hover:text-purple-700 font-medium text-sm focus:outline-none">
                        Show more ▼
                    </button>
                </div>
            </div>
            """ if paper.get('abstract') and len(paper.get('abstract', '')) > 300 else f"""
            <div class="mt-6">
                <h3 class="text-sm font-medium text-gray-500 uppercase tracking-wide mb-2">Abstract</h3>
                <p class="text-gray-700 leading-relaxed">{esc(paper.get('abstract', 'Not available'))}</p>
            </div>
            """ if paper.get('abstract') else ""}
        </div>
        
        <!-- Editorial Decision -->
        <div class="bg-white rounded-lg shadow-lg p-8 mb-8 fade-in" style="animation-delay: 0.1s;">
            <h2 class="text-2xl font-semibold mb-6 flex items-center">
                <span class="text-2xl mr-3">{decision_icon}</span>
                Editorial Decision
            </h2>
            <div class="{decision_class} border-2 rounded-lg p-6">
                <pre class="whitespace-pre-wrap text-sm leading-relaxed">{esc(editor_decision)}</pre>
            </div>
        </div>
        
        <!-- Review Statistics -->
        <div class="grid md:grid-cols-3 gap-6 mb-8">
            <div class="bg-white rounded-lg shadow-lg p-6 text-center fade-in" style="animation-delay: 0.2s;">
                <div class="text-3xl font-bold text-purple-600">{total_reviews}</div>
                <div class="text-gray-600 mt-2">Expert Reviews</div>
            </div>
            <div class="bg-white rounded-lg shadow-lg p-6 text-center fade-in" style="animation-delay: 0.3s;">
                <div class="text-3xl font-bold text-purple-600">{total_words:,}</div>
                <div class="text-gray-600 mt-2">Total Words</div>
            </div>
            <div class="bg-white rounded-lg shadow-lg p-6 text-center fade-in" style="animation-delay: 0.4s;">
                <div class="text-3xl font-bold text-purple-600">{total_words // max(total_reviews, 1)}</div>
                <div class="text-gray-600 mt-2">Avg Words/Review</div>
            </div>
        </div>
        """
        # Sezione sintesi Author & Editor Summary
        author_editor_summary = reviews.get("author_editor_summary", "")
        if author_editor_summary:
            html += f'''
    <!-- Author & Editor Summary -->
    <div class="bg-white rounded-lg shadow-lg p-8 mb-8 fade-in" style="animation-delay: 0.45s;">
        <h2 class="text-2xl font-semibold mb-6 flex items-center">
            <span class="bg-yellow-100 text-yellow-600 p-2 rounded-lg mr-3">📝</span>
            Author & Editor Summary
        </h2>
        <div class="bg-yellow-50 border-l-4 border-yellow-300 rounded-lg p-6">
            <pre class="whitespace-pre-wrap text-base leading-relaxed text-gray-800">{esc(author_editor_summary)}</pre>
        </div>
    </div>
'''
        # Individual Reviews
        html += '''
        <!-- Individual Reviews -->
        <div class="bg-white rounded-lg shadow-lg p-8 fade-in" style="animation-delay: 0.5s;">
            <h2 class="text-2xl font-semibold mb-6 flex items-center">
                <span class="bg-purple-100 text-purple-600 p-2 rounded-lg mr-3">👥</span>
                Expert Reviews
            </h2>
            <div class="space-y-4">
'''
        
        # Mappa per icone e colori dei revisori
        reviewer_styles = {
            "methodology": ("🔬", "bg-blue-50 border-blue-200"),
            "results": ("📊", "bg-green-50 border-green-200"),
            "literature": ("📚", "bg-yellow-50 border-yellow-200"),
            "structure": ("🏗️", "bg-purple-50 border-purple-200"),
            "impact": ("💡", "bg-pink-50 border-pink-200"),
            "contradiction": ("⚡", "bg-red-50 border-red-200"),
            "ethics": ("⚖️", "bg-indigo-50 border-indigo-200"),
            "ai_origin": ("🤖", "bg-gray-50 border-gray-200"),
            "hallucination": ("🔍", "bg-orange-50 border-orange-200"),
            "coordinator": ("🎯", "bg-teal-50 border-teal-200"),
        }
        
        # Ordine preferito per i revisori
        review_order = ["coordinator", "methodology", "results", "literature", 
                       "structure", "impact", "contradiction", "ethics", 
                       "ai_origin", "hallucination"]
        
        # Aggiungi le review nell'ordine specificato
        for reviewer_type in review_order:
            if reviewer_type in reviews:
                icon, style = reviewer_styles.get(reviewer_type, ("📝", "bg-gray-50 border-gray-200"))
                review_content = reviews[reviewer_type]
                word_count = len(review_content.split())
                
                html += f"""
                    <details class="review-card {style} border-2 rounded-lg overflow-hidden">
                        <summary class="p-6 hover:bg-gray-50 transition-colors">
                            <div class="flex items-center justify-between">
                                <div class="flex items-center">
                                    <span class="text-2xl mr-3">{icon}</span>
                                    <div>
                                        <h3 class="text-lg font-semibold">{reviewer_type.replace('_', ' ').title()}</h3>
                                        <p class="text-sm text-gray-600">{word_count} words</p>
                                    </div>
                                </div>
                                <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
                                </svg>
                            </div>
                        </summary>
                        <div class="px-6 pb-6">
                            <div class="review-content bg-white rounded-lg p-4 border border-gray-200">
                                <pre class="whitespace-pre-wrap text-sm text-gray-700 leading-relaxed">{esc(review_content)}</pre>
                            </div>
                        </div>
                    </details>
"""
        
        html += """
            </div>
        </div>
    </div>
    
    <!-- Footer -->
    <footer class="bg-gray-800 text-white py-8 mt-16">
        <div class="container mx-auto px-6 text-center">
            <p class="text-gray-400">Generated by Advanced Multi-Agent Paper Review System</p>
            <p class="text-sm text-gray-500 mt-2">Powered by OpenAI GPT Models</p>
        </div>
    </footer>
    
    <script>
        // Smooth reveal animation on scroll
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                }
            });
        }, observerOptions);
        
        document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));
        
        // Toggle animation for details
        document.querySelectorAll('details').forEach(detail => {
            detail.addEventListener('toggle', (e) => {
                if (e.target.open) {
                    e.target.querySelector('.review-content').style.animation = 'fadeIn 0.3s ease-out';
                }
            });
        });
        
        // Handle expand/collapse for abstract
        const abstractContent = document.getElementById('abstract-content');
        const abstractToggle = document.getElementById('abstract-toggle');
        const abstractGradient = document.getElementById('abstract-gradient');
        
        if (abstractToggle) {
            let isExpanded = false;
            
            abstractToggle.addEventListener('click', () => {
                isExpanded = !isExpanded;
                
                if (isExpanded) {
                    abstractContent.style.maxHeight = abstractContent.scrollHeight + 'px';
                    abstractToggle.textContent = 'Show less ▲';
                    abstractGradient.style.display = 'none';
                } else {
                    abstractContent.style.maxHeight = '150px';
                    abstractToggle.textContent = 'Show more ▼';
                    abstractGradient.style.display = 'block';
                }
            });
            
            // Check if the abstract is short enough to show entirely
            if (abstractContent && abstractContent.scrollHeight <= 150) {
                abstractToggle.style.display = 'none';
                abstractGradient.style.display = 'none';
                abstractContent.style.maxHeight = 'none';
            }
        }
    </script>
</body>
</html>"""
        
        return html

def system_health_check(config: Config) -> Dict[str, Any]:
    """Perform a basic integrity check of the system."""
    report: Dict[str, Any] = {"storage_ok": Path(config.output_dir).exists()}
    start = time.time()
    try:
        client = OpenAI(api_key=config.api_key)
        client.models.list()
        report["api_latency"] = time.time() - start
        report["api_ok"] = True
    except Exception as e:
        report["api_ok"] = False
        report["api_error"] = str(e)
        report["api_latency"] = None
    return report

def main():
    """Main function with improved error handling."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Advanced Multi-Agent System for Scientific Paper Review"
    )
    parser.add_argument("paper_path", help="Path to the paper file to review")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging with specified level
    global logger
    logger = setup_logging(args.log_level)
    
    try:
        # Load configuration
        config = Config.from_yaml(args.config)
        # Se l'utente non specifica --output-dir, crea una directory unica con timestamp
        if args.output_dir:
            config.output_dir = args.output_dir
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config.output_dir = f"output_revisione_paper_{timestamp}"
        
        # Validate configuration
        config.validate()
        health = system_health_check(config)
        logger.info(f"System health: {health}")
        
        # Read paper (PDF or text)
        file_manager = FileManager(config.output_dir)
        if args.paper_path.lower().endswith(".pdf"):
            paper_text = file_manager.extract_text_from_pdf(args.paper_path)
        else:
            paper_text = file_manager.read_paper(args.paper_path)
        
        if not paper_text:
            logger.error("Failed to read paper file")
            return 1
        
        logger.info(f"Paper loaded successfully. Length: {len(paper_text):,} characters")
        
        # Run review process
        orchestrator = ReviewOrchestrator(config)
        results = orchestrator.execute_review_process(paper_text)
        
        logger.info(f"Review process completed. Results saved in: {config.output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
