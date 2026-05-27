"""Paper parsing, metadata extraction, complexity scoring, and file I/O."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import pdfplumber
from openai import AsyncOpenAI, OpenAI

from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class PaperInfo:
    """Structured information about the paper."""

    title: str
    authors: str
    abstract: str
    length: int
    sections: list[str]
    file_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "length": self.length,
            "sections": self.sections,
            "file_path": self.file_path,
        }


class PDFExtractor:
    """Reads paper text from disk. Storage-free / stateless.

    Separating PDF extraction from storage lets the orchestrator swap the
    storage backend (S3, DB, in-memory) without having to also rewrite the
    PDF parsing path. Same goes the other way: a future PDF extractor with
    multi-column heuristics can plug in without touching ``StorageProvider``.
    """

    # Fallback encodings tried in order when reading plain-text papers.
    DEFAULT_ENCODINGS: tuple[str, ...] = ("utf-8", "latin-1", "cp1252", "iso-8859-1")

    def extract_from_pdf(self, pdf_path: str | Path) -> str:
        """Return concatenated text from all pages of a PDF, or '' on failure.

        Failure modes (missing file, malformed PDF, pdfplumber exception) are
        logged + swallowed — callers get an empty string and decide what to
        do. The orchestrator treats empty-text as a hard failure upstream.
        """
        if not Path(pdf_path).exists():
            logger.error("PDF not found: %s", pdf_path)
            return ""
        buf = StringIO()
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text(x_tolerance=1.5, y_tolerance=1.5)
                    buf.write(page_text or "")
                    buf.write("\n\n")
            return buf.getvalue()
        except Exception as e:
            logger.error("PDF extraction failed: %s", e)
            return ""

    def read_text_file(self, file_path: str | Path) -> str | None:
        """Read a text paper trying multiple encodings.

        Returns the file contents on first encoding that decodes cleanly, or
        ``None`` when no encoding works / the file is missing.
        """
        for encoding in self.DEFAULT_ENCODINGS:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                logger.info("Paper read successfully with %s encoding", encoding)
                return content
            except UnicodeDecodeError:
                continue
            except FileNotFoundError:
                logger.error("File not found: %s", file_path)
                return None
            except Exception as e:
                logger.error("Error reading file: %s", e)
                return None
        logger.error("Could not read file with any encoding: %s", file_path)
        return None

    def extract(self, file_path: str | Path) -> str | None:
        """Auto-route by extension: ``.pdf`` → :meth:`extract_from_pdf`,
        anything else → :meth:`read_text_file`. Returns ``None`` on failure."""
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            return self.extract_from_pdf(path) or None
        return self.read_text_file(path)


class FileManager:
    """Backward-compat facade for v1 callers.

    The v1 ``FileManager`` mixed three concerns: PDF / text extraction,
    on-disk storage, and reverse-name lookups. v2 splits them into
    :class:`PDFExtractor` + :class:`~agentic_paper.storage.LocalFileStorage`.
    This class keeps the old surface alive so the CLI, the web runner, and
    existing tests can keep constructing ``FileManager(path)`` and calling
    the original methods unchanged.

    **Prefer the new abstractions in new code.** The orchestrator now
    instantiates a :class:`~agentic_paper.storage.StorageProvider`
    (injectable) for persistence and a :class:`PDFExtractor` for input
    parsing.
    """

    def __init__(self, output_dir: str) -> None:
        # Lazy import to avoid a circular dependency with ``storage.py``.
        from .storage import LocalFileStorage

        self._storage = LocalFileStorage(output_dir)
        self._pdf = PDFExtractor()

    # ------------------------------------------------------------------ attrs

    @property
    def output_dir(self) -> Path:
        """Path attribute v1 exposed; kept for callers that compute
        ``file_manager.output_dir / "results.json"``."""
        return self._storage.output_dir

    # ------------------------------------------------------------ delegation

    def save_json(self, data: Any, filename: str) -> bool:
        return self._storage.save_json(data, filename)

    def save_text(self, text: str, filename: str) -> bool:
        return self._storage.save_text(text, filename)

    def save_review(self, reviewer_name: str, review_content: str) -> str:
        return self._storage.save_review(reviewer_name, review_content)

    def get_reviews(self) -> dict[str, str]:
        return self._storage.get_reviews()

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Return concatenated text from all pages of a PDF, or '' on failure."""
        return self._pdf.extract_from_pdf(pdf_path)

    def read_paper(self, file_path: str) -> str | None:
        """Read a text paper trying multiple encodings."""
        return self._pdf.read_text_file(file_path)


class PaperAnalyzer:
    """Extract title/authors/abstract/sections from a paper text."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.client: OpenAI | None = OpenAI(api_key=config.api_key) if config.api_key else None

    def extract_info(self, paper_text: str) -> PaperInfo:
        info: dict[str, Any] = {}
        ai_success = False

        if self.client:
            try:
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
                    model=self.config.model_basic,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert assistant for scientific literature analysis. Your output must be a single, valid JSON object.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=1.0,
                    response_format={"type": "json_object"},
                    max_completion_tokens=2000,
                )

                extracted_text = response.choices[0].message.content
                info = json.loads(extracted_text)

                if info.get("title") and info.get("title") not in ["Not Found", "Unknown title"]:
                    logger.info("Successfully extracted paper info using AI.")
                    ai_success = True
                else:
                    logger.warning("AI extraction did not find a valid title. Falling back to regex.")

            except Exception as e:
                logger.error("AI-based info extraction failed: %s. Falling back to regex.", e)

        if not ai_success:
            logger.info("Using regex-based method to extract paper info.")
            regex_info = self._extract_info_with_regex(paper_text)
            info["title"] = regex_info.get("title", "Unknown title")
            info["authors"] = regex_info.get("authors", "Unknown authors")
            info["abstract"] = regex_info.get("abstract", "Abstract not found")

        sections = self._identify_sections(paper_text)

        return PaperInfo(
            title=info.get("title", "Unknown title"),
            authors=info.get("authors", "Unknown authors"),
            abstract=info.get("abstract", "Abstract not found"),
            length=len(paper_text),
            sections=sections,
            file_path=None,
        )

    def _extract_info_with_regex(self, paper_text: str) -> dict[str, str]:
        lines = paper_text.split("\n")
        title = next((line.strip() for line in lines if line.strip()), "Unknown title")

        author_patterns = [
            r"(?:Authors?|by|Autori|di):\s*([^\n]+)",
            r"^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)*)",
            r"(?:^|\n)([A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+(?:,\s*[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+)*)",
        ]
        authors = "Unknown authors"
        for pattern in author_patterns:
            match = re.search(pattern, paper_text, re.MULTILINE)
            if match:
                authors = match.group(1).strip()
                break

        abstract_pattern = r"(?:Abstract|Summary|Riassunto|Sommario)[:.\n]\s*([^\n]+(?:\n[^\n]+)*?)(?:\n\n|\n[A-Z]|\n\d+\.|$)"
        abstract_match = re.search(abstract_pattern, paper_text, re.IGNORECASE | re.DOTALL)
        abstract = abstract_match.group(1).strip() if abstract_match else "Abstract not found"

        return {"title": title, "authors": authors, "abstract": abstract}

    @staticmethod
    def _is_valid_section_context(prev_line: str, next_line: str) -> bool:
        """Decide whether a candidate header sits in a believable context.

        A section header is plausible when:
          * the previous line is empty / very short (i.e. the header is its
            own paragraph), OR
          * the next line starts a new sentence (capital letter or a
            non-alphabetic char like a number).

        Pulled out of the giant ``or``-chained boolean inside
        ``_identify_sections``: it was the single biggest contributor to that
        function's cyclomatic complexity. Pure / side-effect-free so it's
        cheap to call once per line.
        """
        if not prev_line:
            return True
        if len(prev_line) < 10:
            return True
        if not next_line:
            return False
        return next_line[0].isupper() or not next_line[0].isalpha()

    @staticmethod
    def _extract_section_title(
        line: str, patterns: list[tuple[str, bool]],
    ) -> str | None:
        """Try each ``(pattern, has_num)`` against ``line``; return a
        formatted section title on first match, ``None`` otherwise.

        Behaviour matches the original ``for pattern, has_num in patterns:
        … break`` loop bit-for-bit:
          * First pattern that matches wins (no further patterns tried).
          * If the matched title falls outside ``2 < len(title) < 50``, the
            line is rejected outright — we do NOT try the remaining patterns,
            preserving the original ``break``-on-first-match semantics.
          * Numbered patterns yield ``"N.N. Title Cased"``, others yield
            ``"Title Cased"`` (calling ``str.title()`` exactly as before).
        """
        for pattern, has_num in patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if not match:
                continue
            title = match.group("title").strip()
            # Match found but title length disqualifies → break-equivalent.
            if not (2 < len(title) < 50):
                return None
            if has_num and match.group("num"):
                return f"{match.group('num')}. {title.title()}"
            return title.title()
        return None

    @staticmethod
    def _identify_sections(paper_text: str) -> list[str]:
        standard_sections = [
            "Abstract", "Introduction", "Background", "Related Work", "Literature Review",
            "Methods", "Methodology", "Materials and Methods", "Experimental Setup",
            "Results", "Experiments", "Evaluation", "Findings",
            "Discussion", "Analysis", "Implications",
            "Conclusion", "Conclusions", "Future Work", "Limitations",
            "References", "Bibliography", "Acknowledgments", "Appendix",
        ]

        section_patterns: list[tuple[str, bool]] = [
            (r"^(?P<num>\d+(?:\.\d+)*)\s*\.?\s+(?P<title>[A-Z][A-Za-z\s\-:]+)$", True),
            (r"^(?P<num>[IVX]+(?:\.[IVX]+)*)\s*\.?\s+(?P<title>[A-Z][A-Za-z\s\-:]+)$", True),
            (r"^(?P<title>[A-Z][A-Z\s\-]{2,})$", False),
            (r"^(?:\d+\.?\s+)?(?P<title>(?:" + "|".join(standard_sections) + r"))\s*:?\s*$", False),
            (r"^#+\s+(?P<title>.+)$", False),
        ]

        lines = paper_text.split("\n")
        sections_found: list[str] = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or len(stripped) > 100:
                continue
            prev_line = lines[i - 1].strip() if i > 0 else ""
            next_line = lines[i + 1].strip() if i < len(lines) - 1 else ""
            if not PaperAnalyzer._is_valid_section_context(prev_line, next_line):
                continue
            title = PaperAnalyzer._extract_section_title(stripped, section_patterns)
            if title is None:
                continue
            if title not in sections_found:
                sections_found.append(title)

        if len(sections_found) < 3:
            sections_found = PaperAnalyzer._identify_sections_heuristic(paper_text, standard_sections)

        sections_found = PaperAnalyzer._filter_similar_sections(sections_found)[:20]
        return sections_found

    @staticmethod
    def _identify_sections_heuristic(paper_text: str, standard_sections: list[str]) -> list[str]:
        sections_found: list[str] = []
        text_lower = paper_text.lower()
        for section in standard_sections:
            section_lower = section.lower()
            patterns = [
                f"\n{section_lower}\n",
                f"\n{section_lower}:",
                f"\n{section_lower}.",
                f"\n1. {section_lower}",
                f"\n2. {section_lower}",
            ]
            for pattern in patterns:
                if pattern in text_lower:
                    sections_found.append(section)
                    break
        return sections_found

    @staticmethod
    def _filter_similar_sections(sections: list[str]) -> list[str]:
        filtered: list[str] = []
        for section in sections:
            section_normalized = re.sub(r"^(?:\d+\.?\d*)\s*", "", section).lower()
            is_duplicate = False
            for existing in filtered:
                existing_normalized = re.sub(r"^(?:\d+\.?\d*)\s*", "", existing).lower()
                if (
                    section_normalized == existing_normalized
                    or section_normalized in existing_normalized
                    or existing_normalized in section_normalized
                ):
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append(section)
        return filtered


async def assess_paper_complexity(paper_text: str, config: Config) -> float:
    """Rate paper complexity on a 0.0-1.0 scale using a small model. Returns 0.5 on failure."""
    if not config.api_key:
        logger.warning("No OpenAI client, using default complexity.")
        return 0.5

    client = AsyncOpenAI(api_key=config.api_key)
    seed = getattr(config, "seed", None)
    try:
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

        kwargs: dict[str, Any] = dict(
            model=config.model_basic,
            messages=[
                {
                    "role": "system",
                    "content": "You are a scientific complexity analyzer. Your output must be a single, valid JSON object.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
            response_format={"type": "json_object"},
            max_completion_tokens=200,
        )
        if seed is not None:
            kwargs["seed"] = seed
        response = await client.chat.completions.create(**kwargs)

        result = json.loads(response.choices[0].message.content)
        score = float(result.get("complexity_score", 0.5))
        if 0.0 <= score <= 1.0:
            logger.info("Assessed paper complexity score: %.2f", score)
            return score
        logger.warning("Invalid complexity score received: %s. Using default 0.5.", score)
        return 0.5

    except Exception as e:
        logger.error("Failed to assess paper complexity: %s. Using default 0.5.", e)
        return 0.5
