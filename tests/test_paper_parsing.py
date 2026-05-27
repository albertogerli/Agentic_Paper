"""Paper parsing tests — PDF text extraction + regex-based info + section detection."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentic_paper.config import Config
from agentic_paper.paper import FileManager, PaperAnalyzer


def test_pdf_extract_text_yields_three_pages(sample_paper_pdf: Path, tmp_path: Path) -> None:
    fm = FileManager(str(tmp_path / "out"))
    text = fm.extract_text_from_pdf(str(sample_paper_pdf))
    assert text, "extracted text should be non-empty"
    # The 3-page fixture mentions each section heading at least once.
    for section in ("Introduction", "Methods", "Results", "Discussion", "Conclusion", "References"):
        assert section.lower() in text.lower(), f"section '{section}' should appear in extracted text"


def test_pdf_missing_returns_empty(tmp_path: Path) -> None:
    fm = FileManager(str(tmp_path / "out"))
    assert fm.extract_text_from_pdf(str(tmp_path / "nope.pdf")) == ""


def test_filemanager_save_and_read_text(tmp_path: Path) -> None:
    fm = FileManager(str(tmp_path / "out"))
    assert fm.save_text("hello\nworld\n", "x.txt") is True
    assert (tmp_path / "out" / "x.txt").read_text() == "hello\nworld\n"


def test_extract_info_regex_path_finds_title_and_abstract(
    monkeypatch: pytest.MonkeyPatch,
    sample_paper_text: str,
) -> None:
    # Force the regex path by ensuring no API key is configured.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    analyzer = PaperAnalyzer(Config(api_key=""))
    assert analyzer.client is None, "no client should be initialized without an API key"

    info = analyzer.extract_info(sample_paper_text)
    assert info.length == len(sample_paper_text)
    assert "Minimal Study of Stub Reviewers" in info.title
    assert "synthetic study" in info.abstract.lower()
    # At least two of the headings should be picked up by the section identifier.
    detected = " ".join(info.sections).lower()
    found = [name for name in ("introduction", "methods", "results", "discussion", "conclusion")
             if name in detected]
    assert len(found) >= 2, f"expected ≥2 section keywords, found {found}"


def test_identify_sections_is_pure_and_bounded(sample_paper_text: str) -> None:
    sections_a = PaperAnalyzer._identify_sections(sample_paper_text)
    sections_b = PaperAnalyzer._identify_sections(sample_paper_text)
    assert sections_a == sections_b, "section identification must be deterministic"
    assert len(sections_a) <= 20, "section list should be capped at 20"
