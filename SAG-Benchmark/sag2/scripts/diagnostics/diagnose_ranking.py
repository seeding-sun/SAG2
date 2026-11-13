"""Separate the ranking loss from the retrieval loss on recorded traces (no model calls).

Every strategy below re-sorts the SAME scoring pool, so the difference between them is
attributable to the ranker alone.

  retrieval  sort by in-hop candidate position          -- no evidence scorer at all
  jev        sort by (-score, position, id)             -- the pipeline as actually run
  perfect    force gold=1 / non-gold=0, then (-score, position) -- a flawless scorer
  oracle     gold first, then position                  -- a flawless ranker

Also provides two counterfactuals:

  --accumulate   keep the initial pool at hop 1 instead of replacing it
  --alpha-sweep  blend the binary scorer with the retrieval prior
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

SAG2_DIR = Path(__file__).resolve().parents[2]
REPO = SAG2_DIR.parents[1]
INDEX = SAG2_DIR / ".cache" / "sag2" / "musique.sqlite"
DATASET = REPO / "dataset"
OUTPUT = REPO / "output"

RETRIEVAL_STAGES = ("bm25", "dense", "hybrid", "expansion")


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


def pool_state(row: dict) -> tuple[dict[int, int], dict[int, float], set[int]]:
    """(creation position, latest score, final pool).

    The final pool is the candidate set of the LAST jev stage only. `latest` keeps the most
    recent score for every candidate ever judged, which the accumulate counterfactual needs.
    """
    position: dict[int, int] = {}
    for entry in row["trace"]:
        if entry["stage"] in RETRIEVAL_STAGES:
            for index, cid in enumerate(entry["candidate_ids"]):
                position.setdefault(cid, index)
    latest: dict[int, float] = {}
    final_pool: set[int] = set()
    for entry in row["trace"]:
        if entry["stage"] == "jev":
            final_pool = {score["id"] for score in entry["scores"]}
            for score in entry["scores"]:
                latest[score["id"]] = score["score"]
    return position, latest, final_pool


def evaluate(gold: dict[str, set[int]], arm: str, level: str) -> dict:
    rows = load(OUTPUT / "abc-100" / level / f"{arm}-{level}" / "details.jsonl")
    tally = {k: [0.0, 0] for k in ("retrieval", "jev", "perfect", "oracle")}
    reproduction = 0
    survival: list[float] = []
    for row in rows:
        g = gold[row["dataset_sample_id"]]
        position, latest, pool = pool_state(row)
        info = {cid: (position.get(cid, 10**6), latest[cid]) for cid in pool}

        def top(key) -> set[int]:
            return set(sorted(pool, key=key)[:10])

        strategies = {
            "retrieval": top(lambda c: (info[c][0], c)),
            "jev": top(lambda c: (-info[c][1], info[c][0], c)),
            "perfect": top(lambda c: (0 if c in g else 1, info[c][0], c)),
            "oracle": top(lambda c: (0 if c in g else 1, c)),
        }
        for name, chosen in strategies.items():
            tally[name][0] += len(g & chosen) / len(g)
            tally[name][1] += int(g <= chosen)

        expected = {int(s) + 1 for s in row["retrieved_source_ids"]}
        if strategies["jev"] == expected:
            reproduction += 1
        if level != "H0":
            initial = set(row["initial_candidate_ids"])
            survival.append(len(initial & pool) / len(initial))

    n = len(rows)
    return {
        "arm": arm, "level": level, "n": n,
        "rates": {k: {"R@10": v[0] / n, "AE@10": v[1] / n} for k, v in tally.items()},
        "reproduction_matches": reproduction,
        "initial_pool_survival": sum(survival) / len(survival) if survival else None,
    }


def accumulate(gold: dict[str, set[int]], arm: str, level: str) -> dict:
    """Counterfactual: keep the initial candidates at hop 1 instead of dropping them.

    Also includes a ``perfect_seeds`` row: the pool is rebuilt with the OFFICIAL construction
    (top-5 protected + 10 seeds + expansion hits) but the 10 seeds are chosen with gold labels.
    That isolates whether the accumulated-pool gain comes from the pool construction itself or
    from the scorer's poor seed selection.
    """
    rows = load(OUTPUT / "abc-100" / level / f"{arm}-{level}" / "details.jsonl")
    out = {k: {"R@10": 0.0, "AE@10": 0, "pool_gold": 0, "pool_size": 0}
           for k in ("current", "accumulated", "current_perfect", "accumulated_perfect",
                     "perfect_seeds")}
    for row in rows:
        g = gold[row["dataset_sample_id"]]
        position, latest, pool = pool_state(row)
        hop0 = next(t for t in row["trace"] if t["stage"] == "jev" and t["hop"] == 0)
        hop0_scores = {s["id"]: s["score"] for s in hop0["scores"]}
        initial = set(row["initial_candidate_ids"])
        merged = pool | initial
        merged_info = {cid: (position.get(cid, 10**6),
                             latest.get(cid, hop0_scores.get(cid, 0.0)))
                       for cid in merged}

        expansion = next((t for t in row["trace"] if t["stage"] == "expansion"), None)
        expansion_ids = set(expansion["candidate_ids"]) if expansion else set()
        protected = set(sorted(initial, key=lambda c: c)[:5])
        ideal_seeds = set(sorted(initial, key=lambda c: (0 if c in g else 1, c))[:10])
        ideal_pool = protected | ideal_seeds | (expansion_ids - initial)

        for name, candidates, info, perfect in (
            ("current", pool, {c: (position.get(c, 10**6), latest[c]) for c in pool}, False),
            ("accumulated", merged, merged_info, False),
            ("current_perfect", pool, {c: (position.get(c, 10**6), latest[c]) for c in pool}, True),
            ("accumulated_perfect", merged, merged_info, True),
            ("perfect_seeds", ideal_pool,
             {c: (position.get(c, 10**6), latest.get(c, hop0_scores.get(c, 0.0)))
              for c in ideal_pool}, True),
        ):
            if perfect:
                key = lambda c: (0 if c in g else 1, info[c][0], c)  # noqa: E731
            else:
                key = lambda c: (-info[c][1], info[c][0], c)  # noqa: E731
            chosen = set(sorted(candidates, key=key)[:10])
            out[name]["R@10"] += len(g & chosen) / len(g)
            out[name]["AE@10"] += int(g <= chosen)
            out[name]["pool_gold"] += len(g & candidates)
            out[name]["pool_size"] += len(candidates)
    n = len(rows)
    for value in out.values():
        value["R@10"] /= n
        value["AE@10"] /= n
        value["pool_size"] /= n
    return out


def alpha_sweep(gold: dict[str, set[int]], arm: str, level: str,
                alphas: tuple[float, ...]) -> dict[str, dict[str, float]]:
    """score = alpha * jev + (1 - alpha) * 61/(61 + position)."""
    rows = load(OUTPUT / "abc-100" / level / f"{arm}-{level}" / "details.jsonl")
    tally = {a: {"R@10": 0.0, "AE@10": 0} for a in alphas}
    for row in rows:
        g = gold[row["dataset_sample_id"]]
        position, latest, pool = pool_state(row)
        for alpha in alphas:
            def key(c):
                i = position.get(c, 10**6)
                return (-(alpha * latest[c] + (1 - alpha) * 61 / (61 + i)), i, c)
            chosen = set(sorted(pool, key=key)[:10])
            tally[alpha]["R@10"] += len(g & chosen) / len(g)
            tally[alpha]["AE@10"] += int(g <= chosen)
    n = len(rows)
    return {f"{a:.1f}": {"R@10": v["R@10"] / n, "AE@10": v["AE@10"] / n}
            for a, v in tally.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="A1")
    parser.add_argument("--hop", default="H1", choices=("H0", "H1"))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--accumulate", action="store_true")
    parser.add_argument("--alpha-sweep", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    gold = gold_map()
    pairs = [(a, h) for h in ("H0", "H1") for a in ("A1", "B1", "C1")] if args.all \
        else [(args.arm, args.hop)]
    report = {}
    for arm, level in pairs:
        key = f"{arm}-{level}"
        report[key] = evaluate(gold, arm, level)
        r = report[key]
        print(f"===== {key} (n={r['n']}) =====")
        for name, value in r["rates"].items():
            print(f"  {name:10s} R@10={value['R@10'] * 100:6.2f}  AE@10={value['AE@10'] * 100:6.2f}")
        print(f"  reproduction of retrieved_source_ids: {r['reproduction_matches']}/{r['n']}")
        if r["initial_pool_survival"] is not None:
            print(f"  initial pool survival at hop 1: {r['initial_pool_survival'] * 100:.1f}%")
        if args.alpha_sweep:
            report[key]["alpha_sweep"] = alpha_sweep(
                gold, arm, level, (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0)
            )
            print("  alpha sweep AE@10: " + "  ".join(
                f"{a}:{v['AE@10'] * 100:5.2f}"
                for a, v in report[key]["alpha_sweep"].items()))
        if args.accumulate and level != "H0":
            report[key]["accumulate"] = accumulate(gold, arm, level)
            print("  pool counterfactual:")
            for name, value in report[key]["accumulate"].items():
                print(f"    {name:20s} R@10={value['R@10'] * 100:6.2f} "
                      f"AE@10={value['AE@10'] * 100:6.2f} "
                      f"pool gold={value['pool_gold']:6.1f} size={value['pool_size']:6.1f}")
        print()
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
