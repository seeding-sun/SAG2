#!/usr/bin/env python3
"""Create a deterministic hop-stratified MuSiQue subset for paired A/B runs."""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import UTC, datetime
from pathlib import Path


def parse_strata(value: str) -> dict[str, int]:
    result = {}
    for part in value.split(","):
        name, separator, count = part.partition(":")
        if not separator or not name or not count.isdigit() or int(count) < 1:
            raise argparse.ArgumentTypeError(f"Invalid stratum: {part}")
        result[name.strip()] = int(count)
    if not result:
        raise argparse.ArgumentTypeError("At least one stratum is required")
    return result


def question_hop(question_id: str) -> str:
    match = re.match(r"^(\d+hop)", question_id)
    if not match:
        raise ValueError(f"Cannot infer hop from MuSiQue question id: {question_id}")
    return match.group(1)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument(
        "--strata",
        type=parse_strata,
        default="2hop:40,3hop:30,4hop:30",
    )
    args = parser.parse_args()
    samples = json.loads((args.dataset_dir / "musique.json").read_text(encoding="utf-8"))
    grouped: dict[str, list[str]] = {}
    for sample in samples:
        question_id = sample["id"]
        grouped.setdefault(question_hop(question_id), []).append(question_id)
    selected = []
    counts = {}
    rng = random.Random(args.seed)
    for stratum, requested in sorted(args.strata.items()):
        available = sorted(grouped.get(stratum, []))
        if requested > len(available):
            raise ValueError(
                f"Requested {requested} {stratum} questions but only {len(available)} exist"
            )
        selected.extend(rng.sample(available, requested))
        counts[stratum] = requested
    if len(selected) != len(set(selected)):
        raise ValueError("Stratified sampling produced duplicate question ids")
    selected.sort()
    atomic_json(
        args.output,
        {
            "dataset": "musique",
            "seed": args.seed,
            "created_at": datetime.now(UTC).isoformat(),
            "strata": counts,
            "count": len(selected),
            "ids": selected,
        },
    )
    print(f"Wrote {len(selected)} questions to {args.output}")


if __name__ == "__main__":
    main()
