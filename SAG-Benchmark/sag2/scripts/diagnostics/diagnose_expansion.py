"""Where does the SAG2 expansion gain go?

Partitions the gold-passage funnel for a recorded run and labels each question with the
stage at which it failed:

  no-entry       no gold passage in the initial top-100 pool
  no-bridge      an entry was found, but the expansion hop added no new gold
  lost-in-rank   every gold passage reached the final pool but not the final top-10
  still-partial  some gold reached the final top-10, but not all of it
  solved         the whole gold set is in the final top-10

Usage
  python scripts/diagnostics/diagnose_expansion.py --arm A1 --hop H1
  python scripts/diagnostics/diagnose_expansion.py --all
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

# CorpusIndex stores source_id as a 0-based string while document_id is 1-based, and the
# runner reports retrieved_source_ids, so document_id = int(source_id) + 1.
def gold_map() -> dict[str, set[int]]:
    db = sqlite3.connect(INDEX)
    text_to_id: dict[str, int] = {}
    for document_id, full_doc in db.execute("SELECT document_id, full_doc FROM chunks"):
        text_to_id.setdefault(full_doc, document_id)
    db.close()

    samples = json.loads((DATASET / "musique.json").read_text())
    out: dict[str, set[int]] = {}
    for sample in samples:
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


def hop_level(question_id: str) -> str:
    if question_id.startswith("2hop"):
        return "2hop"
    if question_id.startswith("3hop"):
        return "3hop"
    return "4hop"


def classify(n_gold: int, in_pool0: int, added: int, pool1: int, final: int) -> str:
    if in_pool0 == 0:
        return "no-entry"
    if final == n_gold:
        return "solved"
    if pool1 == n_gold:
        return "lost-in-rank"
    if added == 0:
        return "no-bridge"
    return "still-partial"


def report(arm: str, level: str) -> dict:
    rows = load(OUTPUT / "abc-100" / level / f"{arm}-{level}" / "details.jsonl")
    gold = gold_map()
    buckets: collections.Counter[str] = collections.Counter()
    per_hop: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    seed_gold = seed_total = 0
    added_gold = expansion_candidates = 0
    surviving_initial_gold = dropped_initial_gold = 0
    funnel = collections.Counter()
    questions = []

    for row in rows:
        qid = row["dataset_sample_id"]
        g = gold.get(qid, set())
        n_gold = len(g)
        pool0 = set(row["initial_candidate_ids"])
        in_pool0 = len(pool0 & g)
        jev0 = next(t for t in row["trace"] if t["stage"] == "jev" and t["hop"] == 0)
        seeds = set(jev0["seed_ids"])
        seed_gold += len(seeds & g)
        seed_total += len(seeds)

        added = 0
        pool1 = pool0
        if level != "H0":
            jev_last = [t for t in row["trace"] if t["stage"] == "jev"][-1]
            pool1 = {s["id"] for s in jev_last["scores"]}
            expansion = next(t for t in row["trace"] if t["stage"] == "expansion")
            new_ids = set(expansion["candidate_ids"]) - pool0
            expansion_candidates += len(new_ids)
            added = len(new_ids & g)
            added_gold += added

        final_ids = {int(s) + 1 for s in row["retrieved_source_ids"]}
        final_gold = len(final_ids & g)
        label = classify(n_gold, in_pool0, added, len(pool1 & g), final_gold)
        buckets[label] += 1
        per_hop[hop_level(qid)][label] += 1

        if level != "H0":
            # Expansion hits are disjoint from the initial pool, so any initial-pool gold that
            # reaches hop 1 without coming from expansion must have survived as a seed.
            surviving_initial_gold += len(pool1 & g) - added
            dropped_initial_gold += in_pool0 - (len(pool1 & g) - added)

        funnel["total"] += n_gold
        funnel["in_pool0"] += in_pool0
        funnel["in_seeds"] += len(seeds & g)
        funnel["added"] += added
        funnel["in_pool1"] += len(pool1 & g)
        funnel["in_final10"] += final_gold
        questions.append(
            {
                "qid": qid,
                "n_gold": n_gold,
                "pool0": in_pool0,
                "seeds": len(seeds & g),
                "added": added,
                "pool1": len(pool1 & g),
                "final": final_gold,
                "label": label,
            }
        )

    print(f"===== {arm} {level} (n={len(rows)}) =====")
    print(f"  seed precision: {seed_gold}/{seed_total} = {seed_gold / seed_total * 100:.2f}%")
    if level != "H0":
        out_of_pool = funnel["total"] - funnel["in_pool0"]
        print(f"  expansion new candidates: {expansion_candidates}, gold among them: "
              f"{added_gold} ({added_gold / expansion_candidates * 100:.3f}%)")
        print(f"  out-of-pool gold recovered by expansion: {added_gold}/{out_of_pool} = "
              f"{added_gold / out_of_pool * 100:.2f}%")
        print(f"  initial-pool gold: {funnel['in_pool0']} -> surviving {surviving_initial_gold}, "
              f"dropped {dropped_initial_gold}, "
              f"net {surviving_initial_gold + added_gold - funnel['in_pool0']:+d}")
        print(f"  surviving initial gold vs gold in seeds: "
              f"{surviving_initial_gold} vs {funnel['in_seeds']} "
              f"({'identical' if surviving_initial_gold == funnel['in_seeds'] else 'DIFFERENT'})")
    print(f"  question buckets: {dict(buckets)}")
    for name in ("2hop", "3hop", "4hop"):
        print(f"    {name}: {dict(per_hop[name])}")
    print("  gold funnel: " + "  ".join(f"{k}={v}" for k, v in funnel.items()))
    print()
    return {"arm": arm, "level": level, "buckets": dict(buckets), "funnel": dict(funnel),
            "seed_precision": seed_gold / seed_total, "questions": questions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="A1")
    parser.add_argument("--hop", default="H1", choices=("H0", "H1"))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    pairs = [(a, h) for h in ("H0", "H1") for a in ("A1", "B1", "C1")] if args.all \
        else [(args.arm, args.hop)]
    results = [report(arm, level) for arm, level in pairs]
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
