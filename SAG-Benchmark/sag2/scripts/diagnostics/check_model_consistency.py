#!/usr/bin/env python3
"""Audit which model actually served each evidence batch in a run.

The gateway in front of vLLM silently remaps model names when a requested
model is not deployed (e.g. ``qwen3.5-122b-a10b`` -> ``qwen3.8-27b``).  The
run's ``config.json`` records what was *requested*; only
``details.jsonl[].jev_usage.batches[].model`` records what actually answered.

This script compares the two and reports per-question contamination, so a run
can never be reported as single-model when it is not.

Usage:
    python -m scripts.diagnostics.check_model_consistency output/run_dir
    python -m scripts.diagnostics.check_model_consistency output/run_dir --json
    python -m scripts.diagnostics.check_model_consistency output/run_dir --allow qwen3.8-27b

Exit codes:
    0  every batch used the configured model (or an explicitly allowed one)
    1  contamination found
    2  run is incomplete or missing required fields
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def model_counts(row: dict[str, Any]) -> Counter[str]:
    usage = row.get("jev_usage") or {}
    return Counter(batch.get("model") for batch in usage.get("batches") or [])


def hop_of(question_id: str) -> str:
    return question_id.split("__")[0][0] + "hop"


def audit(run_dir: Path, allowed: set[str]) -> tuple[int, dict[str, Any]]:
    config_path = run_dir / "config.json"
    details_path = run_dir / "details.jsonl"
    if not config_path.exists() or not details_path.exists():
        return 2, {"error": f"missing config.json or details.jsonl in {run_dir}"}

    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = config.get("evidence_model")
    rows = load_rows(details_path)

    totals: Counter[str] = Counter()
    contaminated: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        counts = model_counts(row)
        totals.update(counts)
        unexpected = {
            model: count
            for model, count in counts.items()
            if model not in {expected, *allowed}
        }
        if unexpected:
            contaminated.append(
                {
                    "row": index,
                    "question": row.get("dataset_sample_id"),
                    "hop": hop_of(str(row.get("dataset_sample_id"))),
                    "unexpected": unexpected,
                    "batches": sum(counts.values()),
                    "recall_at_10": (row.get("metrics") or {}).get("Recall@10"),
                }
            )

    total_batches = sum(totals.values())
    report: dict[str, Any] = {
        "run": str(run_dir),
        "requested_model": expected,
        "allowed_extra": sorted(allowed),
        "question_count": len(rows),
        "batches": total_batches,
        "model_mix": dict(totals),
        "contaminated_questions": len(contaminated),
        "contaminated_batches": sum(
            count
            for model, count in totals.items()
            if model not in {expected, *allowed}
        ),
        "details": contaminated,
    }
    if contaminated:
        report["contaminated_batches_pct"] = (
            report["contaminated_batches"] / total_batches if total_batches else 0.0
        )
        clean = [row for i, row in enumerate(rows) if i not in {c["row"] for c in contaminated}]
        if clean:
            report["clean_subset"] = {
                "n": len(clean),
                "recall_at_10": statistics.mean(
                    (row.get("metrics") or {}).get("Recall@10", 0.0) for row in clean
                ),
            }
        report["hops_affected"] = dict(
            Counter(c["hop"] for c in contaminated)
        )
        return 1, report
    return 0, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="run directory containing config.json and details.jsonl")
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="MODEL",
        help="model name to accept in addition to config.evidence_model (repeatable)",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    code, report = audit(args.run_dir, set(args.allow))

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return code

    if code == 2:
        print(f"ERROR: {report['error']}", file=sys.stderr)
        return code

    print(f"run            : {report['run']}")
    print(f"requested model: {report['requested_model']}")
    print(f"questions      : {report['question_count']}")
    print(f"batches        : {report['batches']}")
    print("actual models  :")
    for model, count in sorted(report["model_mix"].items(), key=lambda kv: -kv[1]):
        pct = count / report["batches"] if report["batches"] else 0.0
        flag = "" if model == report["requested_model"] else "  <-- NOT the configured model"
        print(f"  {str(model):28s} {count:6d}  {pct:6.2%}{flag}")

    if code == 1:
        print(
            f"\nCONTAMINATED: {report['contaminated_questions']} questions, "
            f"{report['contaminated_batches']} batches "
            f"({report['contaminated_batches_pct']:.2%})"
        )
        print(f"affected hops: {report['hops_affected']}")
        if "clean_subset" in report:
            print(
                f"clean subset  : n={report['clean_subset']['n']}, "
                f"R@10={report['clean_subset']['recall_at_10']:.4f}"
            )
        print("\nper-question detail:")
        for item in report["details"]:
            print(
                f"  row {item['row']:3d}  {item['hop']:5s}  "
                f"{str(item['question'])[:40]:42s} "
                f"{item['unexpected']} / {item['batches']} batches"
            )
        return code

    print("\nOK: every batch used the configured model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
