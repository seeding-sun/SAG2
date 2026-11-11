#!/usr/bin/env python3
"""Summarize a paired BM25 versus hybrid SAG2 experiment."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

from sample_identity import index_unique_records, validate_identity_coverage


def load_details(directory: Path) -> dict[str, dict]:
    path = directory / "details.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return index_unique_records(rows, "dataset_sample_id", str(path))


def question_hop(question_id: str) -> str:
    match = re.match(r"^(\d+hop)", question_id)
    return match.group(1) if match else "other"


def bootstrap_mean(values: list[float], *, samples: int, seed: int) -> dict:
    if not values:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(
            sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        )
    means.sort()
    return {
        "mean": sum(values) / len(values),
        "low": means[int(0.025 * (len(means) - 1))],
        "high": means[int(0.975 * (len(means) - 1))],
    }


def pair_metrics(
    a_rows: dict[str, dict],
    b_rows: dict[str, dict],
    section: str,
    name: str,
    *,
    samples: int,
    seed: int,
) -> dict:
    paired = [
        b_rows[key][section][name] - a_rows[key][section][name]
        for key in sorted(a_rows)
    ]
    return bootstrap_mean(paired, samples=samples, seed=seed)


def summarize(
    a_rows: dict[str, dict],
    b_rows: dict[str, dict],
    *,
    samples: int,
    seed: int,
) -> dict:
    metric_names = list(next(iter(a_rows.values()))["metrics"])
    initial_names = list(next(iter(a_rows.values()))["initial_metrics"])
    all_names = list(next(iter(a_rows.values()))["all_evidence_metrics"])
    result = {
        "question_count": len(a_rows),
        "metrics": {
            name: pair_metrics(
                a_rows, b_rows, "metrics", name, samples=samples, seed=seed
            )
            for name in metric_names
        },
        "initial_metrics": {
            name: {
                "a_mean": statistics.fmean(
                    a_rows[key]["initial_metrics"][name] for key in a_rows
                ),
                "b_mean": statistics.fmean(
                    b_rows[key]["initial_metrics"][name] for key in b_rows
                ),
            }
            for name in initial_names
        },
        "all_evidence_metrics": {
            name: pair_metrics(
                a_rows,
                b_rows,
                "all_evidence_metrics",
                name,
                samples=samples,
                seed=seed,
            )
            for name in all_names
        },
        "hop_strata": {},
    }
    for metric_name in ("Recall@10", "AllEvidence@10", "initial_entry_recall"):
        deltas = []
        for key in sorted(a_rows):
            if metric_name == "Recall@10":
                a_value = a_rows[key]["metrics"][metric_name]
                b_value = b_rows[key]["metrics"][metric_name]
            elif metric_name == "AllEvidence@10":
                a_value = a_rows[key]["all_evidence_metrics"][metric_name]
                b_value = b_rows[key]["all_evidence_metrics"][metric_name]
            else:
                a_value = a_rows[key][metric_name]
                b_value = b_rows[key][metric_name]
            deltas.append((question_hop(key), b_value - a_value))
        groups = defaultdict(list)
        for hop, delta in deltas:
            groups[hop].append(delta)
        result["hop_strata"][metric_name] = {
            hop: {
                "n": len(values),
                "mean_delta": statistics.fmean(values),
            }
            for hop, values in sorted(groups.items())
        }
    added_gold = 0
    added_total = 0
    for key in sorted(a_rows):
        a_candidates = {
            row["id"] for row in a_rows[key]["initial_candidates"]
        }
        b_candidates = {
            row["id"]: row for row in b_rows[key]["initial_candidates"]
        }
        added = set(b_candidates) - a_candidates
        added_total += len(added)
        added_gold += sum(b_candidates[chunk_id]["is_gold"] for chunk_id in added)
    result["hybrid_initial_delta"] = {
        "added_candidates": added_total,
        "added_gold": added_gold,
        "noise_count": added_total - added_gold,
        "noise_rate": (added_total - added_gold) / added_total if added_total else 0.0,
    }
    a_r10 = [row["metrics"]["Recall@10"] for row in a_rows.values()]
    b_r10 = [row["metrics"]["Recall@10"] for row in b_rows.values()]
    result["recall10_win_tie_loss"] = {
        "b_wins": sum(b > a for a, b in zip(a_r10, b_r10, strict=True)),
        "ties": sum(b == a for a, b in zip(a_r10, b_r10, strict=True)),
        "a_wins": sum(b < a for a, b in zip(a_r10, b_r10, strict=True)),
    }
    result["cost"] = {
        "a_judgments": sum(row["jev_usage"]["judgments"] for row in a_rows.values()),
        "b_judgments": sum(row["jev_usage"]["judgments"] for row in b_rows.values()),
        "a_input_tokens": sum(row["jev_usage"]["input_tokens"] for row in a_rows.values()),
        "b_input_tokens": sum(row["jev_usage"]["input_tokens"] for row in b_rows.values()),
        "a_mean_query_ms": statistics.fmean(
            row["elapsed_ms"] for row in a_rows.values()
        ),
        "b_mean_query_ms": statistics.fmean(
            row["elapsed_ms"] for row in b_rows.values()
        ),
    }
    return result


def markdown_report(result: dict) -> str:
    lines = [
        "# SAG2 Paired Hybrid Comparison",
        "",
        f"Questions: {result['question_count']}",
        "",
        "## Paired deltas",
        "",
        "| Metric | B - A | 95% bootstrap |",
        "|---|---:|---:|",
    ]
    for section in ("metrics", "all_evidence_metrics"):
        for name, value in result[section].items():
            lines.append(
                f"| {name} | {value['mean']:+.4f} | "
                f"[{value['low']:+.4f}, {value['high']:+.4f}] |"
            )
    lines.extend(
        [
            "",
            "## Initial retrieval",
            "",
            f"- Added candidates: {result['hybrid_initial_delta']['added_candidates']}",
            f"- Added gold evidence: {result['hybrid_initial_delta']['added_gold']}",
            f"- Noise rate: {result['hybrid_initial_delta']['noise_rate']:.2%}",
            "",
            "## Recall@10",
            "",
            f"- B wins: {result['recall10_win_tie_loss']['b_wins']}",
            f"- Ties: {result['recall10_win_tie_loss']['ties']}",
            f"- A wins: {result['recall10_win_tie_loss']['a_wins']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-dir", type=Path, required=True)
    parser.add_argument("--b-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261015)
    args = parser.parse_args()
    a_rows = load_details(args.a_dir)
    b_rows = load_details(args.b_dir)
    validate_identity_coverage(set(a_rows), set(b_rows), "paired A/B details")
    for key in a_rows:
        if a_rows[key]["question"].strip() != b_rows[key]["question"].strip():
            raise ValueError(f"Question text mismatch for {key}")
    result = summarize(
        a_rows,
        b_rows,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(markdown_report(result), encoding="utf-8")
    print(f"Wrote {args.output} and {markdown_path}")


if __name__ == "__main__":
    main()
