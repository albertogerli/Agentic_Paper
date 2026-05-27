"""Journal Editor — consumes the coordinator + reviewer outputs."""

from __future__ import annotations

from ..schemas import EDITOR_OUTPUT_INSTRUCTION, EditorDecision

KEY = "editor"
NAME = "Journal_Editor"
BASE_COMPLEXITY = 0.9
SCHEMA = EditorDecision

INSTRUCTIONS = """You are the editor of a prestigious academic journal.
Based on all reviews and the coordinator's comprehensive assessment, your task is to:
1. Evaluate the paper from an editorial perspective.
2. Consider relevance and adequacy for the journal's audience.
3. Provide a final judgment on the publishability of the paper.
4. Elaborate specific editorial feedback for the authors.

Your decision must be one of: 'accept', 'minor_rev', 'major_rev', 'reject'. Choose the
matching `decision_label` from: 'Accept as is', 'Accept with minor revisions',
'Revise and resubmit (major revisions)', 'Reject'.""" + EDITOR_OUTPUT_INSTRUCTION
