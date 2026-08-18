"""Rank two or more method summaries without inventing missing results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_FIELDS = {"method", "mask_iou", "precision", "recall", "latency_ms"}


def compare(rows: list[dict]) -> dict:
    if len(rows) < 2:
        raise ValueError("At least two method results are required for comparison")
    missing = [sorted(REQUIRED_FIELDS - set(row)) for row in rows]
    if any(missing):
        raise ValueError(f"Method result is missing fields: {missing}")
    ranked = sorted(rows, key=lambda row: (row["mask_iou"], row["recall"], -row["latency_ms"]), reverse=True)
    return {
        "ranking": ranked,
        "selection_note": "Ranking is based on measured mask_iou, recall, and latency; fill values from the held-out test split.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON list of method summaries")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
