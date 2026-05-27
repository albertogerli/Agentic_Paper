"""Generate the 3-page synthetic test PDF.

Run directly to regenerate ``data/fixtures/sample_paper.pdf``::

    python tests/_make_pdf.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path


def build_pdf(path: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter

    # ---- Page 1: title, authors, abstract, introduction
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "A Minimal Study of Stub Reviewers")
    c.setFont("Helvetica", 11)
    c.drawString(72, height - 100, "Authors: Alice Researcher, Bob Scientist, Carol Tester")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, height - 140, "Abstract")
    c.setFont("Helvetica", 10)
    abstract = (
        "This paper presents a synthetic study used to validate the testing fixtures of an "
        "automated multi-agent peer review system. We describe a fictional methodology, report "
        "fabricated results, and conclude with self-referential observations to enable end-to-end "
        "pipeline tests without depending on third-party publications."
    )
    y = height - 158
    for line in textwrap.wrap(abstract, width=85):
        c.drawString(72, y, line)
        y -= 13

    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y - 14, "1. Introduction")
    c.setFont("Helvetica", 10)
    intro = (
        "Automated peer review systems need deterministic fixtures. This paper exists only as a "
        "fixture for pdfplumber. It contains the usual section headings so the section identifier "
        "has something to find."
    )
    y = y - 30
    for line in textwrap.wrap(intro, width=85):
        c.drawString(72, y, line)
        y -= 13
    c.showPage()

    # ---- Page 2: methods + results
    y = height - 72
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y, "2. Methods")
    c.setFont("Helvetica", 10)
    methods = (
        "We invented an experiment with N=3 fictional reviewers and counted imaginary citations. "
        "No data were harmed in the production of this fixture."
    )
    y -= 18
    for line in textwrap.wrap(methods, width=85):
        c.drawString(72, y, line)
        y -= 13

    y -= 12
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y, "3. Results")
    c.setFont("Helvetica", 10)
    results = (
        "All three stub reviewers reached consensus that the paper exists. p=0.04. We report a "
        "Cohen's d of 0.6 and note that the confidence interval crosses zero, which we will "
        "address in section 4."
    )
    y -= 18
    for line in textwrap.wrap(results, width=85):
        c.drawString(72, y, line)
        y -= 13
    c.showPage()

    # ---- Page 3: discussion, conclusion, references
    y = height - 72
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y, "4. Discussion")
    c.setFont("Helvetica", 10)
    discussion = (
        "Limitations include the fact that this paper is entirely fictional. Implications: the "
        "automated review pipeline will produce deterministic outputs against this fixture."
    )
    y -= 18
    for line in textwrap.wrap(discussion, width=85):
        c.drawString(72, y, line)
        y -= 13

    y -= 12
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y, "5. Conclusion")
    c.setFont("Helvetica", 10)
    conclusion = (
        "Stub reviewers approved the paper with minor revisions. The fixture is valid. Future "
        "work consists of writing tests that exercise this paper through the full pipeline."
    )
    y -= 18
    for line in textwrap.wrap(conclusion, width=85):
        c.drawString(72, y, line)
        y -= 13

    y -= 18
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y, "References")
    c.setFont("Helvetica", 9)
    refs = [
        "[1] Stub, A. (2026). On Stub-Driven Development. Journal of Tests, 1(1), 1-9.",
        "[2] Mock, B. (2026). Fixtures as a Design Discipline. Proc. of CI/CD, 42-47.",
    ]
    y -= 14
    for r in refs:
        c.drawString(72, y, r)
        y -= 12
    c.showPage()
    c.save()


if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "sample_paper.pdf"
    build_pdf(target)
    print(f"Wrote {target} ({target.stat().st_size} bytes)", file=sys.stderr)
