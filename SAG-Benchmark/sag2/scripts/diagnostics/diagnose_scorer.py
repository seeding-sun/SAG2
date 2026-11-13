"""Diagnose the evidence scorer's output and the provenance of the expansion terms.

Reports, per arm and hop level:
  * how many gold passages in the scoring pool received score 0 (they can never be ranked)
  * the score histogram of gold vs non-gold passages
  * how many expansion terms came from a gold seed, and how many occur verbatim inside a
    gold passage of the same question
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
from pathlib import Path

SAG2_DIR = Path(__file__).resolve().parents[2]
REPO = SAG2_DIR.parents[1]
INDEX = SAG2_DIR / ".cache" / "sag2" / "musique.sqlite"
DATASET = REPO / "dataset"
OUTPUT = REPO / "output"


def gold_map() -> dict[str, set[int]]:
    db = sqlite3.connect(INDEX)
    text_to_id: dict[str, int] = {}
    for document_id, full_doc in db.execute("SELECT document_id, full_doc FROM chunks"):
        text_to_id.setdefault(full_doc, document_id)
    db.close()
    out: dict[str, set[int]] = {}
    for sample in json.loads((DATASET / "musique.json").read_text()):
        docs = []
        for paragraph in sample["paragraphs"]:
            if not paragraph.get("is_supporting"):
                continue
            text = (
                f"{paragraph['title']}\n"
                f"{paragraph.get('paragraph_text') or paragraph.get('text') or ''}"
            )
            docs.append(text.strip())
        out[sample["id"]] = {text_to_id[t] for t in docs if t in text_to_id}
    return out


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score_distribution(gold, arm: str, level: str) -> dict:
    rows = load(OUTPUT / "abc-100" / level / f"{arm}-{level}" / "details.jsonl")
    gold_scores: list[float] = []
    nongold_scores: list[float] = []
    for row in rows:
        supporting = gold[row["dataset_sample_id"]]
        stages = [t for t in row["trace"] if t["stage"] == "jev"]
        stage = stages[0] if level == "H0" else stages[-1]
        for item in stage["scores"]:
            (gold_scores if item["id"] in supporting else nongold_scores).append(item["score"])
    zeros = sum(1 for s in gold_scores if s == 0)
    return {
        "gold_in_pool": len(gold_scores),
        "gold_zero": zeros,
        "gold_zero_rate": zeros / len(gold_scores) if gold_scores else 0.0,
        "gold_mean": sum(gold_scores) / len(gold_scores) if gold_scores else 0.0,
        "nongold_mean": sum(nongold_scores) / len(nongold_scores) if nongold_scores else 0.0,
        "gold_histogram": dict(collections.Counter(round(s, 1) for s in gold_scores)),
        "nongold_histogram": dict(collections.Counter(round(s, 1) for s in nongold_scores)),
    }


def term_provenance(gold, arm: str, level: str) -> dict:
    rows = load(OUTPUT / "abc-100" / level / f"{arm}-{level}" / "details.jsonl")
    db = sqlite3.connect(INDEX)
    texts = {did: full for did, full in db.execute("SELECT document_id, full_doc FROM chunks")}
    db.close()

    terms_total = from_gold_seed = verbatim = 0
    questions_with_gold_term = questions_with_verbatim = 0
    seed_weight = collections.Counter()
    for row in rows:
        supporting = gold[row["dataset_sample_id"]]
        supporting_text = " ".join(texts[i] for i in supporting).lower()
        expansion = [t for t in row["trace"] if t["stage"] == "expansion"]
        if not expansion:
            continue
        gold_term = verbatim_term = False
        for entry in expansion:
            for term in entry["term_scores"]:
                terms_total += 1
                seed_weight[round(term["seed_weight"], 2)] += 1
                if term["seed_document_id"] in supporting:
                    from_gold_seed += 1
                    gold_term = True
                if term["term"].lower() in supporting_text:
                    verbatim += 1
                    verbatim_term = True
        questions_with_gold_term += int(gold_term)
        questions_with_verbatim += int(verbatim_term)

    return {
        "expansion_terms": terms_total,
        "from_gold_seed": from_gold_seed,
        "from_gold_seed_rate": from_gold_seed / terms_total if terms_total else 0.0,
        "verbatim_in_gold": verbatim,
        "verbatim_in_gold_rate": verbatim / terms_total if terms_total else 0.0,
        "questions_with_gold_seed_term": questions_with_gold_term,
        "questions_with_verbatim_term": questions_with_verbatim,
        "questions": len(rows),
        "seed_weight_histogram": dict(seed_weight),
    }


def seed_composition(gold, arm: str, level: str) -> dict:
    """Decompose the seed slots into scorer-approved vs padding, and measure term quality by
    seed type. Seed precision alone is misleading: with seed_k=10 and ~2.9 gold per question
    the ceiling is ~29%, and padding slots dominate the headline number.
    """
    rows = load(OUTPUT / "abc-100" / level / f"{arm}-{level}" / "details.jsonl")
    db = sqlite3.connect(INDEX)
    texts = {did: full for did, full in db.execute("SELECT document_id, full_doc FROM chunks")}
    db.close()

    approved_slots = approved_gold = padding_slots = padding_gold = 0
    ceiling_slots = 0
    positive_per_question: list[int] = []
    term_by_approved = term_by_approved_hit = 0
    term_by_padding = term_by_padding_hit = 0

    for row in rows:
        supporting = gold[row["dataset_sample_id"]]
        supporting_text = " ".join(texts[i] for i in supporting).lower()
        hop0 = next(t for t in row["trace"] if t["stage"] == "jev" and t["hop"] == 0)
        scores = {s["id"]: s["score"] for s in hop0["scores"]}
        seeds = hop0["seed_ids"]
        ceiling_slots += min(len(supporting), len(seeds))
        positive_per_question.append(sum(1 for v in scores.values() if v > 0))
        for cid in seeds:
            if scores.get(cid, 0) > 0:
                approved_slots += 1
                approved_gold += int(cid in supporting)
            else:
                padding_slots += 1
                padding_gold += int(cid in supporting)

        for entry in row["trace"]:
            if entry["stage"] != "expansion":
                continue
            for term in entry["term_scores"]:
                hit = term["term"].lower() in supporting_text
                if scores.get(term["seed_document_id"], 0) > 0:
                    term_by_approved += 1
                    term_by_approved_hit += int(hit)
                else:
                    term_by_padding += 1
                    term_by_padding_hit += int(hit)

    total_slots = approved_slots + padding_slots
    pool_gold = sum(
        1
        for row in rows
        for s in next(t for t in row["trace"] if t["stage"] == "jev" and t["hop"] == 0)["scores"]
        if s["id"] in gold[row["dataset_sample_id"]]
    )
    seed_gold = approved_gold + padding_gold
    return {
        "seed_slots": total_slots,
        "seed_gold": seed_gold,
        "seed_precision": seed_gold / total_slots if total_slots else 0.0,
        "precision_ceiling": ceiling_slots / total_slots if total_slots else 0.0,
        "approved_slots": approved_slots,
        "approved_gold": approved_gold,
        "approved_precision": approved_gold / approved_slots if approved_slots else 0.0,
        "padding_slots": padding_slots,
        "padding_gold": padding_gold,
        "padding_precision": padding_gold / padding_slots if padding_slots else 0.0,
        "padding_share": padding_slots / total_slots if total_slots else 0.0,
        "pool_gold": pool_gold,
        "seed_recall": seed_gold / pool_gold if pool_gold else 0.0,
        "mean_positive_per_question": (
            sum(positive_per_question) / len(positive_per_question)
            if positive_per_question else 0.0
        ),
        "questions_with_10_positives": sum(1 for n in positive_per_question if n >= 10),
        "terms_by_approved_seed": term_by_approved,
        "terms_by_approved_seed_gold_hit": term_by_approved_hit,
        "terms_by_padding_seed": term_by_padding,
        "terms_by_padding_seed_gold_hit": term_by_padding_hit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="A1")
    parser.add_argument("--hop", default="H1", choices=("H0", "H1"))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    gold = gold_map()
    pairs = [(a, h) for h in ("H0", "H1") for a in ("A1", "B1", "C1")] if args.all \
        else [(args.arm, args.hop)]
    report = {}
    for arm, level in pairs:
        key = f"{arm}-{level}"
        distribution = score_distribution(gold, arm, level)
        print(f"===== {key} =====")
        print(f"  gold in scoring pool: {distribution['gold_in_pool']}, "
              f"scored 0: {distribution['gold_zero']} "
              f"({distribution['gold_zero_rate'] * 100:.2f}%)")
        print(f"  mean score  gold={distribution['gold_mean']:.4f}  "
              f"non-gold={distribution['nongold_mean']:.4f}")
        print(f"  gold histogram: {distribution['gold_histogram']}")
        report[key] = {"scores": distribution}

        seed = seed_composition(gold, arm, level)
        print(f"  seed slots: {seed['seed_slots']}, gold {seed['seed_gold']} "
              f"=> precision {seed['seed_precision'] * 100:.2f}% "
              f"(ceiling {seed['precision_ceiling'] * 100:.2f}%)")
        print(f"    approved (score>0): {seed['approved_slots']} slots, gold "
              f"{seed['approved_gold']} = {seed['approved_precision'] * 100:.2f}%")
        print(f"    padding  (score=0): {seed['padding_slots']} slots "
              f"({seed['padding_share'] * 100:.1f}%), gold {seed['padding_gold']} "
              f"= {seed['padding_precision'] * 100:.2f}%")
        print(f"    seed recall of pool gold: {seed['seed_gold']}/{seed['pool_gold']} "
              f"= {seed['seed_recall'] * 100:.2f}%")
        print(f"    mean positives per question: {seed['mean_positive_per_question']:.1f}, "
              f"questions with >=10: {seed['questions_with_10_positives']}")
        report[key]["seed"] = seed
        if level != "H0":
            provenance = term_provenance(gold, arm, level)
            print(f"  expansion terms: {provenance['expansion_terms']}, "
                  f"from a gold seed: {provenance['from_gold_seed']} "
                  f"({provenance['from_gold_seed_rate'] * 100:.2f}%)")
            print(f"  terms occurring verbatim in a gold passage: "
                  f"{provenance['verbatim_in_gold']} "
                  f"({provenance['verbatim_in_gold_rate'] * 100:.2f}%), "
                  f"questions with >=1: {provenance['questions_with_verbatim_term']}"
                  f"/{provenance['questions']}")
            approved = seed["terms_by_approved_seed"]
            if approved:
                print(f"  terms from approved seed: {approved}, "
                      f"gold hit {seed['terms_by_approved_seed_gold_hit']} "
                      f"({seed['terms_by_approved_seed_gold_hit'] / approved * 100:.2f}%)")
            padding_terms = seed["terms_by_padding_seed"]
            if padding_terms:
                print(f"  terms from padding seed:  {padding_terms}, "
                      f"gold hit {seed['terms_by_padding_seed_gold_hit']} "
                      f"({seed['terms_by_padding_seed_gold_hit'] / padding_terms * 100:.2f}%)")
            print(f"  seed weight histogram: {provenance['seed_weight_histogram']}")
            report[key]["terms"] = provenance
        print()
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
