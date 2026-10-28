"""Check eval/questions.jsonl against the corpus.

For every answerable question, each part of `source_passage` (split on "...")
must appear in the cited paper's text. For every unanswerable question, the
named subject must not appear anywhere in the corpus. Text is compared after
reducing both sides to lowercase letters and digits, because PDF extraction
breaks words across lines and turns "fi" into a ligature.

Usage:
    python eval/check_questions.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

from pypdf import PdfReader

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS = EVAL_DIR / "questions.jsonl"
CORPUS_DIR = EVAL_DIR / "corpus"

# The subject each unanswerable question asks about. None of these may occur in the corpus,
# matched as a whole phrase (case-insensitive) so "Claude 3" does not match "Claude-1.3".
UNANSWERABLE_SUBJECTS = {
    "GraphRAG", "Gemini 1.5", "Mixtral", "text-embedding-3", "pgvector",
    "LangGraph", "Llama 3", "Mistral", "GPT-4o", "Claude 3",
}

logging.getLogger("pypdf").setLevel(logging.ERROR)

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}


def normalize(text: str) -> str:
    for lig, plain in LIGATURES.items():
        text = text.replace(lig, plain)
    return re.sub(r"[^a-z0-9]", "", text.lower())


def paper_text(arxiv_id: str) -> str:
    reader = PdfReader(CORPUS_DIR / f"{arxiv_id}.pdf")
    return "".join((page.extract_text() or "") + "\n" for page in reader.pages)


def main() -> int:
    rows = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    raw = {p.stem: paper_text(p.stem) for p in sorted(CORPUS_DIR.glob("*.pdf"))}
    texts = {stem: normalize(text) for stem, text in raw.items()}
    failures = []

    for row in rows:
        if not row["answerable"]:
            continue
        text = texts.get(row["source"])
        if text is None:
            failures.append(f"{row['id']}: source {row['source']} is not in the corpus")
            continue
        for part in row["source_passage"].split("..."):
            if normalize(part) not in text:
                failures.append(f"{row['id']}: passage not found in {row['source']}: {part.strip()[:80]!r}")

    everything = re.sub(r"\s+", " ", " ".join(raw.values()))
    for subject in sorted(UNANSWERABLE_SUBJECTS):
        if re.search(rf"(?<!\w){re.escape(subject)}(?!\w)", everything, re.IGNORECASE):
            failures.append(f"unanswerable subject {subject!r} appears in the corpus")

    answerable = sum(r["answerable"] for r in rows)
    print(f"{len(rows)} questions: {answerable} answerable, {len(rows) - answerable} unanswerable")
    for failure in failures:
        print("FAIL", failure)
    if not failures:
        print("every passage found in its paper; no unanswerable subject appears in the corpus")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
