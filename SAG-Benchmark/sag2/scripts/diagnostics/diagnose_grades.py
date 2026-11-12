"""Acceptance diagnostics for the graded scoring scale (arm D1 versus the A1 baseline).

Everything is recomputed from `details.jsonl` rather than read from the summary, and the
ranking is replayed from the trace so a mismatch against the recorded ids is caught.

Usage:
    python scripts/diagnostics/diagnose_grades.py \
        --run ../../output/abc-100/H1/D1-H1 \
        --baseline ../../output/abc-100/H1/A1-H1 \
        --threshold 0.4
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_gold(
    dataset_dir: Path, index_path: Path
) -> tuple[dict[str, set[int]], dict[str, int]]:
    """Map each question to its gold document ids, using full text to bridge the id spaces."""
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        text_to_id: dict[str, int] = {}
        for document_id, full_doc in connection.execute(
            "SELECT document_id, full_doc FROM chunks"
        ):
            text_to_id.setdefault(full_doc, document_id)
    finally:
        connection.close()
    samples = {
        sample["id"]: sample
        for sample in json.loads((dataset_dir / "musique.json").read_text(encoding="utf-8"))
    }
    gold: dict[str, set[int]] = {}
    for question_id, sample in samples.items():
        ids = set()
        for paragraph in sample["paragraphs"]:
            if not paragraph.get("is_supporting"):
                continue
            text = (
                f"{paragraph['title']}\n"
                f"{paragraph.get('paragraph_text') or paragraph.get('text') or ''}"
            ).strip()
            document_id = text_to_id.get(text)
            if document_id is not None:
                ids.add(document_id)
        gold[question_id] = ids
    return gold, text_to_id


def hops(row: dict) -> list[dict]:
    return [step for step in row["trace"] if step["stage"] == "jev"]


def first_hop_scores(row: dict) -> dict[int, float]:
    steps = hops(row)
    return {entry["id"]: entry["score"] for entry in steps[0]["scores"]} if steps else {}


def final_scores(row: dict) -> dict[int, float]:
    steps = hops(row)
    return {entry["id"]: entry["score"] for entry in steps[-1]["scores"]} if steps else {}


def grade_of(score: float) -> int:
    return int(round(score * 10))


def replay_top_k(row: dict, k: int) -> list[int]:
    """Reproduce the delivered order from the trace so the diagnostic can self-check."""
    steps = hops(row)
    if not steps:
        return []
    scores = final_scores(row)
    fusion = {}
    for step in row["trace"]:
        if step["stage"] in {"bm25", "expansion", "hybrid", "dense", "rrf"}:
            for position, candidate_id in enumerate(step.get("candidate_ids", [])):
                fusion.setdefault(candidate_id, 1 / (61 + position))
    ordered = sorted(scores, key=lambda cid: (-scores[cid], -fusion.get(cid, 0.0), cid))
    return ordered[:k]


def kendall_tau(left: dict[int, float], right: dict[int, float]) -> float | None:
    shared = sorted(set(left) & set(right))
    if len(shared) < 2:
        return None
    concordant = discordant = 0
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            a = left[shared[i]] - left[shared[j]]
            b = right[shared[i]] - right[shared[j]]
            product = a * b
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else None


def report_grades(rows: list[dict], threshold: float, label: str) -> dict:
    histogram: Counter = Counter()
    seeds_per_question: list[int] = []
    no_seed = 0
    eligible_per_question: list[int] = []
    pool_sizes: list[int] = []

    for row in rows:
        steps = hops(row)
        if not steps:
            continue
        hop_zero = steps[0]
        scores = {entry["id"]: entry["score"] for entry in hop_zero["scores"]}
        for score in scores.values():
            histogram[grade_of(score)] += 1
        pool_sizes.append(len(scores))
        eligible = sum(1 for score in scores.values() if score >= threshold)
        eligible_per_question.append(eligible)
        seed_ids = hop_zero.get("seed_ids", [])
        seeds_per_question.append(len(seed_ids))
        if not seed_ids:
            no_seed += 1

    total = max(len(rows), 1)
    return {
        "label": label,
        "questions": len(rows),
        "grade_histogram": dict(sorted(histogram.items())),
        "distinct_grades_used": sum(1 for count in histogram.values() if count),
        "seeds_per_question": {
            "mean": sum(seeds_per_question) / total,
            "min": min(seeds_per_question, default=0),
            "max": max(seeds_per_question, default=0),
            "histogram": dict(sorted(Counter(seeds_per_question).items())),
        },
        "eligible_per_question": sum(eligible_per_question) / total,
        "no_seed_questions": no_seed,
        "no_seed_ratio": no_seed / total,
        "pool_size": {
            "mean": sum(pool_sizes) / total,
            "min": min(pool_sizes, default=0),
            "max": max(pool_sizes, default=0),
        },
    }


def report_seed_precision(rows: list[dict], gold: dict[str, set[int]], threshold: float) -> dict:
    slots = hits = eligible_slots = eligible_hits = padded_slots = padded_hits = 0
    for row in rows:
        steps = hops(row)
        if not steps:
            continue
        relevant = gold.get(row["dataset_sample_id"], set())
        hop_zero = steps[0]
        scores = {entry["id"]: entry["score"] for entry in hop_zero["scores"]}
        for seed_id in hop_zero.get("seed_ids", []):
            score = scores.get(seed_id, 0.0)
            slots += 1
            if score >= threshold:
                eligible_slots += 1
                if seed_id in relevant:
                    eligible_hits += 1
            else:
                padded_slots += 1
                if seed_id in relevant:
                    padded_hits += 1
            if seed_id in relevant:
                hits += 1
    return {
        "seed_slots": slots,
        "seed_gold": hits,
        "seed_precision": hits / slots if slots else 0.0,
        "eligible_slots": eligible_slots,
        "eligible_gold": eligible_hits,
        "eligible_precision": eligible_hits / eligible_slots if eligible_slots else 0.0,
        "padded_slots": padded_slots,
        "padded_gold": padded_hits,
        "padded_precision": padded_hits / padded_slots if padded_slots else 0.0,
    }


def report_metrics(rows: list[dict], gold: dict[str, set[int]], threshold: float) -> dict:
    grouped = {"has_seed": [], "no_seed": []}
    overall = []
    for row in rows:
        steps = hops(row)
        if not steps:
            continue
        relevant = gold.get(row["dataset_sample_id"], set())
        if not relevant:
            continue
        retrieved = replay_top_k(row, 10)
        recall = len(relevant & set(retrieved)) / len(relevant)
        overall.append(recall)
        key = "has_seed" if steps[0].get("seed_ids") else "no_seed"
        grouped[key].append(recall)
    return {
        "recall_at_10": sum(overall) / len(overall) if overall else 0.0,
        "has_seed": {
            "questions": len(grouped["has_seed"]),
            "recall_at_10": (
                sum(grouped["has_seed"]) / len(grouped["has_seed"])
                if grouped["has_seed"]
                else 0.0
            ),
        },
        "no_seed": {
            "questions": len(grouped["no_seed"]),
            "recall_at_10": (
                sum(grouped["no_seed"]) / len(grouped["no_seed"]) if grouped["no_seed"] else 0.0
            ),
        },
    }


def delivered_document_ids(row: dict, text_to_id: dict[str, int], k: int = 10) -> set[int]:
    """Map the delivered passages back to document ids via their full text."""
    return {
        text_to_id[text]
        for text in row.get("retrieved_docs", [])[:k]
        if text in text_to_id
    }


def self_check(rows: list[dict], text_to_id: dict[str, int]) -> tuple[int, int]:
    """Replay the ranking and compare the delivered set.

    rank_dpp only reorders the leading candidate_k slice, so the delivered top-10 set
    must equal the pre-DPP top-10 set. A mismatch means the replay logic is wrong.
    """
    agree = total = 0
    for row in rows:
        recorded = delivered_document_ids(row, text_to_id)
        if not recorded:
            continue
        replayed = set(replay_top_k(row, 10))
        total += 1
        if replayed == recorded:
            agree += 1
    return agree, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--compare", type=Path, help="second run for a Kendall tau comparison")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT.parents[1] / "dataset")
    parser.add_argument(
        "--index", type=Path, default=ROOT / ".cache" / "sag2" / "musique.sqlite"
    )
    args = parser.parse_args()

    rows = load_rows(args.run / "details.jsonl")
    gold, text_to_id = load_gold(args.dataset_dir, args.index)

    print(f"===== run {args.run.name} (n={len(rows)}) =====")
    print(f"  threshold = {args.threshold}")

    current = report_grades(rows, args.threshold, "current")
    print("\n--- grade distribution ---")
    for grade, count in current["grade_histogram"].items():
        print(f"  grade {grade:2d}: {count:5d}")
    print(f"  distinct grades used: {current['distinct_grades_used']}")
    print(
        "  seeds/question: mean {mean:.2f}, min {min}, max {max}".format(
            **current["seeds_per_question"]
        )
    )
    print(f"  seeds/question histogram: {current['seeds_per_question']['histogram']}")
    print(f"  eligible (>= threshold)/question: {current['eligible_per_question']:.2f}")
    print(
        f"  no-seed questions: {current['no_seed_questions']}"
        f" ({current['no_seed_ratio']:.1%})"
    )
    print(
        "  scored pool size: mean {mean:.1f}, min {min}, max {max}".format(
            **current["pool_size"]
        )
    )

    precision = report_seed_precision(rows, gold, args.threshold)
    print("\n--- seed precision ---")
    print(
        f"  all seeds         : {precision['seed_gold']}/{precision['seed_slots']}"
        f" = {precision['seed_precision']:.2%}"
    )
    print(
        f"  eligible seeds    : {precision['eligible_gold']}/{precision['eligible_slots']}"
        f" = {precision['eligible_precision']:.2%}"
    )
    print(
        f"  padded (below thr): {precision['padded_gold']}/{precision['padded_slots']}"
        f" = {precision['padded_precision']:.2%}"
    )

    metrics = report_metrics(rows, gold, args.threshold)
    print("\n--- metrics (replayed) ---")
    print(f"  Recall@10 overall : {metrics['recall_at_10']:.4f}")
    print(
        f"  has-seed ({metrics['has_seed']['questions']:2d} q):"
        f" {metrics['has_seed']['recall_at_10']:.4f}"
    )
    print(
        f"  no-seed  ({metrics['no_seed']['questions']:2d} q):"
        f" {metrics['no_seed']['recall_at_10']:.4f}"
    )

    agree, total = self_check(rows, text_to_id)
    print(f"\n--- self check ---\n  delivered top-10 set reproduced: {agree}/{total}")

    if args.baseline:
        base_rows = load_rows(args.baseline / "details.jsonl")
        base = report_grades(base_rows, args.threshold, "baseline")
        base_precision = report_seed_precision(base_rows, gold, args.threshold)
        base_metrics = report_metrics(base_rows, gold, args.threshold)
        print(f"\n===== baseline {args.baseline.name} (n={len(base_rows)}) =====")
        print(
            "  distinct grades used: {0}, seeds/question mean {1:.2f}, "
            "no-seed {2:.1%}".format(
                base["distinct_grades_used"],
                base["seeds_per_question"]["mean"],
                base["no_seed_ratio"],
            )
        )
        print(
            f"  seed precision {base_precision['seed_precision']:.2%},"
            f" Recall@10 {base_metrics['recall_at_10']:.4f}"
        )

    if args.compare:
        other_rows = {row["dataset_sample_id"]: row for row in load_rows(args.compare / "details.jsonl")}
        taus = []
        for row in rows:
            other = other_rows.get(row["dataset_sample_id"])
            if other is None:
                continue
            tau = kendall_tau(first_hop_scores(row), first_hop_scores(other))
            if tau is not None:
                taus.append(tau)
        print(f"\n===== batch comparison: {args.run.name} vs {args.compare.name} =====")
        if taus:
            print(f"  hop-0 Kendall tau: mean {sum(taus) / len(taus):.4f} over {len(taus)} questions")
            print(f"  min {min(taus):.4f}, max {max(taus):.4f}")
            below = sum(1 for value in taus if value < 0.8)
            print(f"  questions below tau 0.8: {below}/{len(taus)}")
        else:
            print("  no comparable questions")


if __name__ == "__main__":
    main()
