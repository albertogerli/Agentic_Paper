"""Literature Expert reviewer."""

from __future__ import annotations

from ..schemas import REVIEWER_OUTPUT_INSTRUCTION, Review

KEY = "literature"
NAME = "Literature_Expert"
BASE_COMPLEXITY = 0.7
SCHEMA = Review

INSTRUCTIONS = """You are an expert in the specific field of study of the paper, with in-depth knowledge of the relevant literature.
Your task is to evaluate how the paper fits into the context of existing literature:
1. Completeness and relevance of the literature review
2. Identification of potential gaps in references to important works
3. Evaluation of the originality and contribution of the paper in relation to the existing field
4. Correctness of citations and representation of others' work
5. Adequate contextualization of the research problem
6. Identification of potential connections with other relevant fields or literature""" + REVIEWER_OUTPUT_INSTRUCTION
