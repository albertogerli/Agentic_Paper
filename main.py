"""
Sistema Multi-Agente per la Revisione di Paper Scientifici.
Versione alternativa senza dipendenza dal framework 'agents'.

Questo sistema usa le OpenAI API direttamente invece del framework
`agents`.
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

# Configurazione logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configura il sistema di logging."""
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
    """Configurazione centralizzata del sistema."""
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model_powerful: str = "o3"
    model_standard: str = "gpt-4.1"
    model_basic: str = "gpt-4.1-mini"
    output_dir: str = "output_revisione_paper"
    max_parallel_agents: int = 3
    agent_timeout: int = 300  # secondi
    temperature_methodology: float = 1
    temperature_results: float = 1
    temperature_literature: float = 1
    temperature_structure: float = 1
    temperature_impact: float = 1
    temperature_contradiction: float = 1
    temperature_ethics: float = 1
    temperature_coordinator: float = 1
    temperature_editor: float = 1
    temperature_ai_origin: float = 1  # Nuova temperatura per AI Origin Detector
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Carica configurazione da file YAML."""
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
        """Valida la configurazione."""
        if not self.api_key:
            raise ValueError("API key not configured. Set OPENAI_API_KEY environment variable.")
        return True

# Implementazione alternativa del sistema di agenti
class Agent:
    """Implementazione semplificata di un agente usando OpenAI API."""
    
    def __init__(self, name: str, instructions: str, model: str, temperature: float = 0.7):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.temperature = temperature
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Inizializza il client OpenAI."""
        config = Config()
        if config.api_key:
            self.client = OpenAI(api_key=config.api_key)
        else:
            logger.warning("OpenAI client not initialized - no API key")
  
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
    def run(self, message: str) -> str:
        """Esegue l'agente con il messaggio dato."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        # Verifica che il messaggio non sia vuoto
        if not message or not message.strip():
            raise ValueError("Message content cannot be empty")
        
        try:
            # Alcuni modelli (o1-preview, o1-mini) supportano solo temperature=1
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
    """Versione asincrona dell'agente."""

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
    """Agente asincrono con caching dei risultati."""

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
    """Informazioni strutturate sul paper."""
    title: str
    authors: str
    abstract: str
    length: int
    sections: List[str]
    file_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte in dizionario."""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "length": self.length,
            "sections": self.sections,
            "file_path": self.file_path
        }

class FileManager:
    """Gestisce le operazioni su file con gestione degli errori."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_json(self, data: Any, filename: str) -> bool:
        """Salva dati in formato JSON con gestione errori."""
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
        """Salva testo in un file con gestione errori."""
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
        """Restituisce il testo concatenato di tutte le pagine di un PDF."""
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
        """Salva la revisione di un revisore."""
        filename = f"review_{reviewer_name.replace(' ', '_')}.txt"
        success = self.save_text(review_content, filename)
        if success:
            return f"Review successfully saved in {filename}"
        else:
            return f"Error saving review for {reviewer_name}"
    
    def get_reviews(self) -> Dict[str, str]:
        """Recupera tutte le revisioni salvate."""
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
        """Legge il contenuto di un paper con gestione encoding multipli."""
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
    """Analizza e estrae informazioni dal paper."""
    
    @staticmethod
    def extract_info(paper_text: str) -> PaperInfo:
        """Estrae informazioni strutturate dal paper."""
        # Estrai titolo
        lines = paper_text.split('\n')
        title = next((line.strip() for line in lines if line.strip()), "Unknown title")
        
        # Cerca autori con pattern migliorato
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
        
        # Cerca abstract con pattern migliorato
        abstract_pattern = r'(?:Abstract|Summary|Riassunto|Sommario)[:.\n]\s*([^\n]+(?:\n[^\n]+)*?)(?:\n\n|\n[A-Z]|\n\d+\.|$)'
        abstract_match = re.search(abstract_pattern, paper_text, re.IGNORECASE | re.DOTALL)
        abstract = abstract_match.group(1).strip() if abstract_match else "Abstract not found"
        
        # Identifica sezioni
        sections = PaperAnalyzer._identify_sections(paper_text)
        
        return PaperInfo(
            title=title,
            authors=authors,
            abstract=abstract[:500] + "..." if len(abstract) > 500 else abstract,
            length=len(paper_text),
            sections=sections
        )
    
    @staticmethod
    def _identify_sections(paper_text: str) -> List[str]:
        """Identifica le principali sezioni del paper con pattern migliorato."""
        section_patterns = [
            r'(?:^|\n)#+\s*([^\n]+)',  # Markdown headers
            r'(?:^|\n)(\d+\.?\s+[A-Z][^\n]+)',  # Numbered sections
            r'(?:^|\n)([A-Z][A-Z\s]{2,})\n',  # All caps headers
            r'(?:^|\n)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\n(?:-{3,}|={3,})',  # Underlined headers
        ]
        
        sections = []
        for pattern in section_patterns:
            matches = re.findall(pattern, paper_text, re.MULTILINE)
            sections.extend([m.strip() for m in matches if len(m.strip()) > 2])
        
        # Rimuovi duplicati mantenendo l'ordine
        seen = set()
        unique_sections = []
        for section in sections:
            if section not in seen:
                seen.add(section)
                unique_sections.append(section)
        
        return unique_sections[:20]  # Limita a 20 sezioni

class AgentFactory:
    """Factory per creare agenti con configurazioni appropriate."""
    
    def __init__(self, config: Config):
        self.config = config
        self.file_manager = FileManager(config.output_dir)
    
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
            model=self.config.model_powerful,
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
            model=self.config.model_powerful,
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
            model=self.config.model_standard,
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
            model=self.config.model_basic,
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
            model=self.config.model_standard,
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

End your review with: "REVIEW COMPLETED - Contradiction Checker" """,
            model=self.config.model_powerful,
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
            model=self.config.model_standard,
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
            model=self.config.model_standard, # o model_powerful a seconda della necessità
            temperature=self.config.temperature_ai_origin,
        )

    def create_hallucination_detector(self) -> Agent:
        return Agent(
            name="Hallucination_Detector",
            instructions="""You are tasked with spotting potential hallucinations in the paper. Look for:\n1. Claims lacking citations\n2. Data inconsistent with official sources\n3. Conclusions not supported by presented data\n4. Invented or malformed references\nProvide a concise report IN ENGLISH detailing any suspicious statements.""",
            model="gpt-4o-2024-05-13",
            temperature=self.config.temperature_standard if hasattr(self.config, 'temperature_standard') else 1,
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
            model=self.config.model_powerful,
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
            model=self.config.model_standard,
            temperature=self.config.temperature_editor,
        )
    
    def create_all_agents(self) -> Dict[str, Agent]:
        """Crea tutti gli agenti necessari."""
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
            "editor": self.create_editor_agent()
        }

class ReviewOrchestrator:
    """Orchestratore principale del processo di revisione."""
    
    def __init__(self, config: Config):
        self.config = config
        self.file_manager = FileManager(config.output_dir)
        self.agent_factory = AgentFactory(config)
        self.agents = {}
    
    def execute_review_process(self, paper_text: str) -> Dict[str, Any]:
        """Esegue il processo completo di revisione con gestione errori."""
        try:
            # Estrai informazioni sul paper
            paper_info = PaperAnalyzer.extract_info(paper_text)
            self.file_manager.save_json(paper_info.to_dict(), "paper_info.json")
            
            logger.info("Starting multi-agent peer review process...")
            
            # Crea agenti
            self.agents = self.agent_factory.create_all_agents()
            
            # Prepara messaggio iniziale
            initial_message = self._prepare_initial_message(paper_info, paper_text)
            
            # Esegui revisori principali
            reviews = self._execute_main_reviewers(initial_message)
            
            # Esegui coordinatore
            coordinator_review = self._execute_coordinator(reviews)
            reviews["coordinator"] = coordinator_review
            
            # Esegui editor
            editor_decision = self._execute_editor(reviews)
            
            # Sintetizza risultati
            final_results = self._synthesize_results(paper_info, reviews, editor_decision)
            
            # Genera report
            self._generate_reports(final_results)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Critical error in review process: {e}")
            raise
    
    def _prepare_initial_message(self, paper_info: PaperInfo, paper_text: str) -> str:
        """Prepara il messaggio iniziale per gli agenti."""

        # Manteniamo sempre il testo completo del paper. Se supera la soglia
        # consigliata per alcuni modelli, emettiamo solo un avviso di log.
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
        """Esegue i revisori principali usando batch asincroni."""
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
        """Esegue più agenti in parallelo con asyncio."""
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
        """Esegue un agente e salva la sua revisione."""
        try:
            review = agent.run(message)
            # Salva la revisione
            self.file_manager.save_review(agent_name, review)
            return review
        except Exception as e:
            logger.error(f"Agent execution error for {agent_name}: {e}")
            raise
    
    def _execute_coordinator(self, reviews: Dict[str, str]) -> str:
        """Esegue il coordinatore con tutte le revisioni."""
        coordinator = self.agents.get("coordinator")
        if not coordinator:
            logger.error("Coordinator agent not found")
            return "Coordinator review not available"
        
        # Prepara messaggio con tutte le revisioni
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
    
    def _execute_editor(self, all_reviews: Dict[str, str]) -> str:
        """Esegue l'editor per la decisione finale."""
        editor = self.agents.get("editor")
        if not editor:
            logger.error("Editor agent not found")
            return "Editorial decision not available"
        
        # Prepara messaggio con tutte le revisioni incluso coordinatore
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
        """Sintetizza i risultati delle revisioni."""
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
        """Genera report in vari formati."""
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
        """Genera report dettagliato in Markdown."""
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
        """Genera sommario esecutivo."""
        paper_info = results["paper_info"]
        editor_decision = results["editor_decision"]
        coordinator_assessment = results["reviews"].get("coordinator", "")
        
        summary = f"""# Executive Summary

**Paper:** {paper_info['title']}
**Authors:** {paper_info['authors']}
**Review Date:** {results['timestamp']}

## Editorial Decision

{editor_decision}

## Coordinator's Overall Assessment

{coordinator_assessment}

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
    """Genera un semplice dashboard HTML riassuntivo."""

    def generate_html_dashboard(self, results: Dict[str, Any]) -> str:
        html = ["<html><head><meta charset='utf-8'><title>Review Dashboard</title></head><body>"]
        html.append(f"<h1>Review Results {results['timestamp']}</h1>")
        html.append("<h2>Reviews</h2><ul>")
        for name, review in results.get('reviews', {}).items():
            html.append(f"<li>{name}: {len(review.split())} words</li>")
        html.append("</ul>")
        html.append(f"<h2>Editorial Decision</h2><p>{results.get('editor_decision','')}</p>")
        html.append("</body></html>")
        return "\n".join(html)


def system_health_check(config: Config) -> Dict[str, Any]:
    """Esegue un controllo di base dell'integrità del sistema."""
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
    """Funzione principale con gestione errori migliorata."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Advanced Multi-Agent System for Scientific Paper Review"
    )
    parser.add_argument("paper_path", help="Path to the paper file to review")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging con livello specificato
    global logger
    logger = setup_logging(args.log_level)
    
    try:
        # Carica configurazione
        config = Config.from_yaml(args.config)
        if args.output_dir:
            config.output_dir = args.output_dir
        
        # Valida configurazione
        config.validate()
        health = system_health_check(config)
        logger.info(f"System health: {health}")
        
        # Leggi paper (PDF o testo)
        file_manager = FileManager(config.output_dir)
        if args.paper_path.lower().endswith(".pdf"):
            paper_text = file_manager.extract_text_from_pdf(args.paper_path)
        else:
            paper_text = file_manager.read_paper(args.paper_path)
        
        if not paper_text:
            logger.error("Failed to read paper file")
            return 1
        
        logger.info(f"Paper loaded successfully. Length: {len(paper_text):,} characters")
        
        # Esegui processo di revisione
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
