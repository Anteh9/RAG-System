"""Sentence-aware chunking utilities for the custom RAG pipeline."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def normalize_whitespace(text: str) -> str:
    """Collapse arbitrary whitespace to single spaces while preserving content."""
    return re.sub(r"\s+", " ", text or "").strip()


class SentenceAwareChunker:
    """Sentence-boundary-aware chunker for retrieval corpora."""

    def __init__(self, max_chars: int = 800, overlap: int = 0):
        self.max_chars = max_chars
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        return chunk_sentence_aware(text, max_chars=self.max_chars, overlap=self.overlap)


def split_sentences(text: str) -> List[str]:
    """Split text into sentence-like units without breaking numeric or statistic fragments."""
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []

    parts = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    sentences = [part.strip() for part in parts if part and part.strip()]
    if not sentences:
        return [cleaned]

    return sentences


def split_long_sentence(sentence: str, max_chars: int = 800) -> List[str]:
    """Break a long sentence on clause boundaries before it exceeds the maximum chunk size."""
    if len(sentence) <= max_chars:
        return [sentence]

    for separator in ["; ", " - ", " | ", ": ", ", "]:
        if separator in sentence:
            pieces = [part.strip() for part in re.split(re.escape(separator), sentence) if part and part.strip()]
            if len(pieces) <= 1:
                continue

            merged: List[str] = []
            current = ""
            for piece in pieces:
                candidate = f"{current} {piece}".strip()
                if not current:
                    current = piece
                    continue
                if len(candidate) <= max_chars:
                    current = candidate
                    continue

                if current:
                    merged.append(current)
                current = piece

            if current:
                merged.append(current)

            return [part for part in merged if part]

    return [sentence.strip()]


def chunk_sentence_aware(text: str, max_chars: int = 800, overlap: int = 0) -> List[str]:
    """Bundle complete sentences into chunks with no mid-sentence cuts."""
    sentences: List[str] = []
    for sentence in split_sentences(text):
        for piece in split_long_sentence(sentence, max_chars=max_chars):
            sentences.append(piece)

    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    for sentence in sentences:
        candidate = " ".join(current + [sentence])
        if len(candidate) <= max_chars:
            current.append(sentence)
            continue

        if current:
            chunks.append(" ".join(current))
        current = [sentence]

    if current:
        chunks.append(" ".join(current))

    if overlap > 0:
        overlapped: List[str] = []
        for index, chunk in enumerate(chunks):
            if index == 0:
                overlapped.append(chunk)
                continue

            tail = overlapped[-1][-overlap:]
            overlapped.append(f"{tail} {chunk}".strip())
        return overlapped

    return chunks


def csv_row_to_text(row: Dict[str, Any], row_index: int) -> str:
    """Translate one election record into a sentence-like retrieval unit."""
    values = {
        "year": row.get("Year", "unknown"),
        "old_region": row.get("Old Region", "unknown"),
        "new_region": row.get("New Region", "unknown"),
        "code": row.get("Code", "unknown"),
        "candidate": row.get("Candidate", "unknown"),
        "party": row.get("Party", "unknown"),
        "votes": row.get("Votes", "unknown"),
        "votes_pct": row.get("Votes(%)", row.get("Votes (%)", "unknown")),
    }

    return (
        f"Election Result Row {row_index}: year: {values['year']}. old region: {values['old_region']}. new region: "
        f"{values['new_region']}. code: {values['code']}. candidate: {values['candidate']}. party: {values['party']}. "
        f"votes: {values['votes']}. votes(%): {values['votes_pct']}"
    )


def build_processed_chunks(
    csv_path: str | Path = "Ghana_Election_Result.csv",
    budget_path: str | Path = "cleaned_budget_text.txt",
    output_path: str | Path = "processed_chunks.json",
    chunk_size: int = 800,
) -> List[Dict[str, Any]]:
    """Build a sentence-safe corpus from the election CSV and budget text."""
    chunks: List[Dict[str, Any]] = []

    csv_file = Path(csv_path)
    if csv_file.exists():
        with csv_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader):
                text = csv_row_to_text(row, row_index)
                chunks.append(
                    {
                        "text": text,
                        "source": "election_csv",
                        "chunk_id": len(chunks),
                        "chunk_config": {"chunk_size": chunk_size, "overlap": 0, "strategy_name": "sentence_aware"},
                        "metadata": {
                            "row_index": row_index,
                            "data_type": "election_result",
                            "columns_present": list(row.keys()),
                        },
                    }
                )

    budget_file = Path(budget_path)
    if budget_file.exists():
        budget_text = budget_file.read_text(encoding="utf-8")
        budget_sentences = split_sentences(budget_text)
        for sentence in budget_sentences:
            for part in chunk_sentence_aware(sentence, max_chars=chunk_size, overlap=0):
                chunks.append(
                    {
                        "text": part,
                        "source": "budget_text",
                        "chunk_id": len(chunks),
                        "chunk_config": {"chunk_size": chunk_size, "overlap": 0, "strategy_name": "sentence_aware"},
                        "metadata": {"data_type": "budget_summary", "sentence_count": len(budget_sentences)},
                    }
                )

    output_file = Path(output_path)
    output_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return chunks


def ensure_processed_chunks(
    output_path: str | Path = "processed_chunks.json",
    csv_path: str | Path = "Ghana_Election_Result.csv",
    budget_path: str | Path = "cleaned_budget_text.txt",
    chunk_size: int = 800,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Regenerate the chunk cache if it is missing or malformed."""
    output_file = Path(output_path)
    if output_file.exists() and not force:
        try:
            data = json.loads(output_file.read_text(encoding="utf-8"))
            if isinstance(data, list) and data and all(isinstance(item, dict) and "text" in item for item in data):
                return data
        except (json.JSONDecodeError, OSError, TypeError):
            pass

    return build_processed_chunks(
        csv_path=csv_path,
        budget_path=budget_path,
        output_path=output_path,
        chunk_size=chunk_size,
    )
