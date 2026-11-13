"""Paired comparison between two SAG2 arms on the same question set.

Recomputes every metric from `details.jsonl` (never reads `metrics.json`), bridges the
two id spaces through the passage text stored in the BM25 index, then reports per-K
paired deltas with a bootstrap interval and a sign test.

Usage:
    python scripts/diagnostics/compare_arms.py \
        --baseline ../../output/abc-100/H1/A1-H1 \
        --arm ../../output/grade-100/H1/D1-H1 \
        --output ../../output/grade-100/H1/comparison-D1vsA1.json
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
K_LIST = (1, 2, 5, 10)


def load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    with (path / "details.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["dataset_sample_id"]] = row
    return rows


def passage_text_to_document_id(index: Path) -> dict[str, int]:
    connection = sqlite3.connect(f"file:{index}?mode=ro", uri=True)
    try:
        mapping: dict[str, int] = {}
        for document_id, full_doc in connection.execute(
            "SELECT document_id, full_doc FROM chunks"
        ):
            mapping.setdefault(full_doc, document_id)
        return mapping
    finally:
        connection.close()


def load_gold(dataset_dir: Path, text_to_id: dict[str, int]) -> dict[str, set[int]]:
    samples = json.loads((dataset_dir / "musique.json").read_text(encoding="utf-8"))
    gold: dict[str, set[int]] = {}
    for sample in samples:
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
        gold[sample["id"]] = ids
    return gold


def delivered_document_ids(row: dict, text_to_id: dict[str, int], k: int) -> list[int]:
    ordered = []
    for text in row.get("retrieved_docs", [])[:k]:
        if text in text_to_id:
            ordered.append(text_to_id[text])
    return ordered


def recall_at(ordered: list[int], gold: set[int], k: int) -> float:
    if not gold:
        return 0.0
    return len(gold & set(ordered[:k])) / len(gold)


def hop_depth(row: dict) -> str:
    return row["dataset_sample_id"].split("__")[0][0] + "hop"


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down, so four K values do not inflate the false positive rate."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(pvalues) - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT.parents[1] / "dataset")
    parser.add_argument("--index", type=Path, default=ROOT / ".cache" / "sag2" / "musique.sqlite")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()

    text_to_id = passage_text_to_document_id(args.index)
    gold = load_gold(args.dataset_dir, text_to_id)
    base = load_rows(args.baseline)
    arm = load_rows(args.arm)
    shared = sorted(set(base) & set(arm))
    if not shared:
        raise SystemExit("no overlapping questions; the two runs are not comparable")
    only_base = len(set(base) - set(arm))
    only_arm = len(set(arm) - set(base))

    scored: dict[str, dict[str, list[float]]] = {}
    for name, rows in (("baseline", base), ("arm", arm)):
        per_question = {}
        for question_id in shared:
            row = rows[question_id]
            ordered = delivered_document_ids(row, text_to_id, max(K_LIST))
            per_question[question_id] = {
                k: recall_at(ordered, gold[question_id], k) for k in K_LIST
            }
        scored[name] = per_question

    rng = random.Random(args.seed)
    summary = {
        "baseline": str(args.baseline),
        "arm": str(args.arm),
        "questions_compared": len(shared),
        "questions_only_in_baseline": only_base,
        "questions_only_in_arm": only_arm,
        "metrics": {},
    }

    print(f"===== {args.arm.name} vs {args.baseline.name} (n={len(shared)}) =====")
    if only_base or only_arm:
        print(f"  WARNING: {only_base} baseline-only, {only_arm} arm-only questions dropped")
    print(f"{'K':>3s}{'baseline':>10s}{'arm':>9s}{'delta':>9s}{'95% CI':>20s}{'p':>8s}{'p(Holm)':>9s}")

    pvalues = []
    for k in K_LIST:
        deltas = [scored["arm"][q][k] - scored["baseline"][q][k] for q in shared]
        observed = sum(deltas) / len(deltas)
        boots = []
        for _ in range(args.resamples):
            sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
            boots.append(sum(sample) / len(sample))
        boots.sort()
        low = boots[int(0.025 * len(boots))]
        high = boots[int(0.975 * len(boots)) - 1]
        up = sum(1 for value in deltas if value > 1e-9)
        down = sum(1 for value in deltas if value < -1e-9)
        changed = up + down
        # two-sided exact sign test
        if changed:
            tail = sum(_binom(changed, i) for i in range(min(up, down) + 1)) / 2**changed
            pvalue = min(1.0, 2 * tail)
        else:
            pvalue = 1.0
        pvalues.append(pvalue)
        summary["metrics"][f"R@{k}"] = {
            "baseline": sum(scored["baseline"][q][k] for q in shared) / len(shared),
            "arm": sum(scored["arm"][q][k] for q in shared) / len(shared),
            "delta": observed,
            "ci95": [low, high],
            "sign_test": {"up": up, "down": down, "unchanged": len(shared) - changed},
            "p": pvalue,
        }
        print(
            f"{k:3d}{summary['metrics'][f'R@{k}']['baseline']:10.4f}"
            f"{summary['metrics'][f'R@{k}']['arm']:9.4f}{observed:+9.4f}"
            f"  [{low:+.4f}, {high:+.4f}]{pvalue:8.3f}",
            end="",
        )
        print()

    adjusted = holm(pvalues)
    for k, value in zip(K_LIST, adjusted):
        summary["metrics"][f"R@{k}"]["p_holm"] = value
    print(f"{'':3s}{'':10s}{'':9s}{'':9s}{'':20s}{'':8s}{'Holm adj':>9s}")
    for k, value in zip(K_LIST, adjusted):
        print(f"{k:3d}{'':28s}{value:18.3f}")

    # per-hop-depth breakdown
    depths = sorted({hop_depth(arm[q]) for q in shared})
    summary["by_depth"] = {}
    print("\n--- R@10 by hop depth ---")
    for depth in depths:
        group = [q for q in shared if hop_depth(arm[q]) == depth]
        b = sum(scored["baseline"][q][10] for q in group) / len(group)
        a = sum(scored["arm"][q][10] for q in group) / len(group)
        summary["by_depth"][depth] = {
            "questions": len(group),
            "baseline_R@10": b,
            "arm_R@10": a,
            "delta": a - b,
        }
        print(f"  {depth:5s} n={len(group):3d}  {b:.4f} -> {a:.4f}  ({a - b:+.4f})")

    # seed accounting
    for name, rows in (("baseline", base), ("arm", arm)):
        slots = padded = gold_seeds = 0
        for question_id in shared:
            steps = [step for step in rows[question_id]["trace"] if step["stage"] == "jev"]
            if not steps:
                continue
            hop_zero = steps[0]
            scores = {entry["id"]: entry["score"] for entry in hop_zero["scores"]}
            for seed_id in hop_zero.get("seed_ids", []):
                slots += 1
                if scores.get(seed_id, 0.0) < 0.4:
                    padded += 1
                if seed_id in gold[question_id]:
                    gold_seeds += 1
        summary[f"{name}_seeds"] = {
            "slots": slots,
            "padded": padded,
            "padded_ratio": padded / slots if slots else 0.0,
            "precision": gold_seeds / slots if slots else 0.0,
        }

    print("\n--- seed accounting (hop 0) ---")
    for name in ("baseline", "arm"):
        entry = summary[f"{name}_seeds"]
        print(
            f"  {name:9s} slots={entry['slots']:5d} padded={entry['padded']:5d}"
            f" ({entry['padded_ratio']:.1%}) precision={entry['precision']:.2%}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nwritten {args.output}")


def _binom(n: int, k: int) -> int:
    from math import comb

    return comb(n, k)


if __name__ == "__main__":
    main()
