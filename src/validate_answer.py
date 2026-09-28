"""Strict UTF-8 submission validation, with a JSON report and file SHA-256.

The CSV is read with the standard library, so identifiers are never interpreted
as numbers and leading zeros/case remain intact. This does not evaluate recall.
Run locally: ``python src/validate_answer.py --answer answer.csv``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
ITEM_ID = re.compile(r"[0-9a-f]{16}\Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_rows(
    header: Sequence[str] | None,
    rows: Iterable[Sequence[str]],
    expected_query_ids: Sequence[str],
    corpus_item_ids: Sequence[str],
    max_examples: int = 30,
) -> dict:
    """Validate literal CSV fields; useful independently for small smoke tests."""
    expected = set(expected_query_ids)
    corpus = set(corpus_item_ids)
    if len(expected) != len(expected_query_ids):
        raise ValueError("Reference benchmark queries contain duplicate query_id")
    if any(not isinstance(x, str) or len(x) != 16 for x in expected):
        raise ValueError("Reference benchmark query_id is not a 16-character string")
    if any(not isinstance(x, str) or ITEM_ID.fullmatch(x) is None for x in corpus):
        raise ValueError("Reference corpus contains malformed item_id")

    errors: Counter = Counter()
    examples: list[dict] = []

    def error(code: str, **details):
        errors[code] += 1
        if len(examples) < max_examples:
            examples.append({"error": code, **details})

    if list(header or []) != ["query_id", "answer"]:
        error(
            "invalid_header", actual=list(header or []), expected=["query_id", "answer"]
        )
    seen: Counter = Counter()
    candidates_per_row: list[int] = []
    row_count = 0
    for row_count, row in enumerate(rows, start=1):
        line = row_count + 1  # Header occupies line 1 (logical CSV rows).
        if len(row) != 2:
            error("wrong_column_count", row=line, columns=len(row))
            continue
        query_id, answer = row
        seen[query_id] += 1
        if len(query_id) != 16:
            error("invalid_query_id_length", row=line, query_id=query_id)
        if query_id not in expected:
            error("unknown_query_id", row=line, query_id=query_id)
        if seen[query_id] > 1:
            error("duplicate_query_id", row=line, query_id=query_id)
        # Zero candidates is legal. For nonempty answers insist on one ASCII
        # space, catching accidental list/comma/tab formats and stray spaces.
        ids = [] if answer == "" else answer.split(" ")
        candidates_per_row.append(len(ids))
        if answer and (" ".join(answer.split()) != answer or any(x == "" for x in ids)):
            error("invalid_answer_separator", row=line, query_id=query_id)
        if len(ids) > 50:
            error("too_many_candidates", row=line, query_id=query_id, count=len(ids))
        if len(ids) != len(set(ids)):
            error("duplicate_item_id", row=line, query_id=query_id)
        for item_id in ids:
            if ITEM_ID.fullmatch(item_id) is None:
                error(
                    "invalid_item_id_format",
                    row=line,
                    query_id=query_id,
                    item_id=item_id,
                )
            if item_id not in corpus:
                error(
                    "item_not_in_corpus", row=line, query_id=query_id, item_id=item_id
                )
    missing = sorted(expected - set(seen))
    extra = sorted(set(seen) - expected)
    if missing:
        errors["missing_query_id"] = len(missing)
        for query_id in missing[: max(0, max_examples - len(examples))]:
            examples.append({"error": "missing_query_id", "query_id": query_id})
    if row_count != len(expected_query_ids):
        error("wrong_row_count", actual=row_count, expected=len(expected_query_ids))
    counts = candidates_per_row or [0]
    return {
        "valid": not bool(errors),
        "expected_query_count": len(expected_query_ids),
        "actual_row_count": row_count,
        "unique_query_count": len(seen),
        "corpus_item_count": len(corpus),
        "columns": list(header or []),
        "candidate_count": {
            "minimum": min(counts),
            "maximum": max(counts),
            "mean": sum(candidates_per_row) / max(1, len(candidates_per_row)),
            "empty_answers": sum(x == 0 for x in candidates_per_row),
        },
        "error_counts": dict(errors),
        "missing_query_ids_sample": missing[:max_examples],
        "extra_query_ids_sample": extra[:max_examples],
        "error_examples": examples,
        "all_errors_count": sum(errors.values()),
    }


def validate_answer(answer: Path, queries: Path, items: Path) -> dict:
    expected = (
        pq.read_table(queries, columns=["query_id"]).column("query_id").to_pylist()
    )
    corpus = pq.read_table(items, columns=["item_id"]).column("item_id").to_pylist()
    sha256 = file_sha256(answer)
    try:
        # utf-8-sig accepts plain UTF-8 and a possible UTF-8 BOM; the BOM is
        # separately reported and cannot become part of the first column name.
        with answer.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, delimiter=",", strict=True)
            report = validate_rows(next(reader, None), reader, expected, corpus)
    except (UnicodeDecodeError, csv.Error) as exc:
        report = {
            "valid": False,
            "error_counts": {"invalid_utf8_or_csv": 1},
            "error_examples": [{"error": "invalid_utf8_or_csv", "detail": str(exc)}],
        }
    with answer.open("rb") as stream:
        has_bom = stream.read(3) == b"\xef\xbb\xbf"
    report.update(
        {
            "answer_path": str(answer.resolve()),
            "answer_bytes": answer.stat().st_size,
            "sha256": sha256,
            "utf8_bom": has_bom,
            "benchmark_queries_path": str(queries.resolve()),
            "benchmark_items_path": str(items.resolve()),
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer", type=Path, default=ROOT / "answer.csv")
    parser.add_argument(
        "--queries", type=Path, default=ROOT / "data/benchmark_queries.parquet"
    )
    parser.add_argument(
        "--items", type=Path, default=ROOT / "data/benchmark_items.parquet"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/answer_validation.json"
    )
    args = parser.parse_args()
    try:
        report = validate_answer(args.answer, args.queries, args.items)
    except (OSError, ValueError) as exc:
        report = {
            "valid": False,
            "error_counts": {"input_or_reference_error": 1},
            "error_examples": [
                {"error": "input_or_reference_error", "detail": str(exc)}
            ],
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
