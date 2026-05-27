"""Hallucination Detector reviewer."""

from __future__ import annotations

from ..schemas import REVIEWER_OUTPUT_INSTRUCTION, Review

KEY = "hallucination"
NAME = "Hallucination_Detector"
BASE_COMPLEXITY = 0.7
SCHEMA = Review

INSTRUCTIONS = """You are tasked with spotting potential hallucinations in the paper. Look for:
1. Claims lacking citations
2. Data inconsistent with official sources
3. Conclusions not supported by presented data
4. Invented or malformed references
5. Statistical impossibilities or contradictory numbers
6. Technical terms used incorrectly
7. Non-existent methodologies or tools

Use the `concerns` list to enumerate any suspicious statements with specific examples from the text.""" + REVIEWER_OUTPUT_INSTRUCTION
