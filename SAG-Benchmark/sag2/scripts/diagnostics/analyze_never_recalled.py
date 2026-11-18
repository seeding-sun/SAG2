#!/usr/bin/env python3
"""Gold fate: never-in-any-pool vs in-pool-not-top10 across rescore arms."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # 知识库/SAG2
if not (ROOT / "dataset" / "musique.json").exists():
    ROOT = Path("/data/machine_learning/cmh/知识库/SAG2")


def hop_bucket(qid: str) -> str:
    if qid.startswith("2hop"):
        return "2hop"
    if qid.startswith("3hop"):
        return "3hop"
    return "4hop"


def load_golds() -> tuple[dict, list]:
    corpus = json.loads((ROOT / "dataset/musique_corpus.json").read_text())
    title_text_to_id = {
        (c["title"], c["text"].strip()): i for i, c in enumerate(corpus)
    }
    ids = set(json.loads((ROOT / "dataset/musique_100_for_ab.json").read_text())["ids"])
    qs = json.loads((ROOT / "dataset/musique.json").read_text())
    golds: dict[str, list] = {}
    unmatched = 0
    for q in qs:
        if q["id"] not in ids:
            continue
        items = []
        for p in q["paragraphs"]:
            if not p.get("is_supporting"):
                continue
            key = (p["title"], p["paragraph_text"].strip())
            did = title_text_to_id.get(key)
            if did is None:
                unmatched += 1
                continue
            items.append(
                {
                    "doc_id": did,
                    "title": p["title"],
                    "idx": p["idx"],
                    "text": p["paragraph_text"][:160].replace("\n", " "),
                }
            )
        golds[q["id"]] = items
    return golds, corpus


def pools_from_trace(trace: list) -> tuple[dict[int, set[int]], set[int]]:
    hop_pools: dict[int, set[int]] = defaultdict(set)
    all_pool: set[int] = set()
    for s in trace:
        hop = int(s.get("hop") or 0)
        ids = s.get("candidate_document_ids") or s.get("candidate_ids") or []
        for x in ids:
            hop_pools[hop].add(int(x))
            all_pool.add(int(x))
        if s.get("stage") == "jev":
            for sc in s.get("scores") or []:
                hop_pools[hop].add(int(sc["id"]))
                all_pool.add(int(sc["id"]))
    return hop_pools, all_pool


def in_top10(r: dict, g: dict, corpus: list) -> bool:
    did = g["doc_id"]
    src = {str(x) for x in (r.get("retrieved_source_ids") or [])[:10]}
    if str(did) in src or did in {int(x) for x in src if str(x).isdigit()}:
        return True
    # source_id sometimes off-by-one vs document_id; match title+text prefix
    gold_full = corpus[did]["title"] + "\n" + corpus[did]["text"]
    for doc in (r.get("retrieved_docs") or [])[:10]:
        if doc == gold_full or (
            isinstance(doc, str)
            and doc.startswith(g["title"] + "\n")
            and corpus[did]["text"][:80] in doc
        ):
            return True
    return False


def grades_for(r: dict, did: int) -> list:
    out = []
    for s in r.get("trace") or []:
        if s.get("stage") != "jev":
            continue
        for sc in s.get("scores") or []:
            if int(sc["id"]) == did:
                out.append(
                    {
                        "hop": s.get("hop"),
                        "grade": sc.get("grade"),
                        "score": sc.get("score"),
                        "evidence_n": len(s.get("evidence_ids") or []),
                    }
                )
    return out


def analyze(arm_dir: Path, golds: dict, corpus: list) -> dict:
    never, miss, hit = [], [], []
    never_by_hop = Counter()
    total = 0
    for line in arm_dir.joinpath("details.jsonl").open():
        r = json.loads(line)
        qid = r["dataset_sample_id"]
        hop_pools, all_pool = pools_from_trace(r.get("trace") or [])
        for g in golds[qid]:
            total += 1
            did = g["doc_id"]
            hb = hop_bucket(qid)
            in_any = did in all_pool
            if not in_any:
                never.append({"qid": qid, "hop": hb, **g})
                never_by_hop[hb] += 1
            elif not in_top10(r, g, corpus):
                miss.append(
                    {
                        "qid": qid,
                        "hop": hb,
                        **g,
                        "in_h0": did in hop_pools.get(0, set()),
                        "grades": grades_for(r, did),
                    }
                )
            else:
                hit.append({"qid": qid, "hop": hb, **g, "in_h0": did in hop_pools.get(0, set())})
    return {
        "total": total,
        "hit": len(hit),
        "never": len(never),
        "miss": len(miss),
        "never_by_hop": dict(never_by_hop),
        "never_rows": never,
        "miss_rows": miss,
        "hit_rows": hit,
    }


def main() -> None:
    golds, corpus = load_golds()
    print(
        f"loaded golds: questions={len(golds)} passages={sum(len(v) for v in golds.values())}"
    )
    base = ROOT / "output/27b-100/H1-rescore"
    arms = {
        "A6-H2": base / "A6-H2-qwen3.8-27b",
        "B6-H2": base / "B6-H2-qwen3.8-27b",
        "C6-H2": base / "C6-H2-qwen3.8-27b",
    }
    results = {}
    for name, path in arms.items():
        results[name] = analyze(path, golds, corpus)
        r = results[name]
        print(f"\n=== {name} ===")
        print(
            f"total={r['total']}  top10={r['hit']}  "
            f"in_pool_not_top10={r['miss']}  NEVER_any_pool={r['never']}"
        )
        print(f"never_by_hop={r['never_by_hop']}")
        # miss grade summary
        last_grades = []
        for row in r["miss_rows"]:
            if row["grades"]:
                last_grades.append(row["grades"][-1].get("grade"))
            else:
                last_grades.append(None)
        print(
            "miss last_grade dist:",
            dict(Counter(last_grades)),
            "max_grade>=4:",
            sum(
                1
                for row in r["miss_rows"]
                if any((g.get("grade") or 0) >= 4 for g in row["grades"])
            ),
        )

    def keys(name: str) -> set:
        return {(x["qid"], x["doc_id"], x["title"]) for x in results[name]["never_rows"]}

    na, nb, nc = keys("A6-H2"), keys("B6-H2"), keys("C6-H2")
    print("\n=== never-pool set compare (passage-level) ===")
    print(f"A={len(na)} B={len(nb)} C={len(nc)}")
    print(f"A∩B∩C (三路都捞不到)={len(na & nb & nc)}")
    print(f"A有C无(dense救回)={len(na - nc)}  C有A无={len(nc - na)}")
    print(f"B有C无={len(nb - nc)}  C有B无={len(nc - nb)}")
    print(f"仅A={len(na - nb - nc)} 仅B={len(nb - na - nc)} 仅C={len(nc - na - nb)}")

    # Questions with any never-recalled gold on C6
    q_never = Counter(x["qid"] for x in results["C6-H2"]["never_rows"])
    print(f"\nC6-H2 questions with ≥1 never-gold: {len(q_never)}")
    print("by hop:", dict(Counter(hop_bucket(q) for q in q_never)))

    out_dir = ROOT / "output/27b-100/H1-rescore/diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    hard = sorted(na & nb & nc)
    lookup = {}
    for name in results:
        for row in results[name]["never_rows"]:
            lookup[(row["qid"], row["doc_id"], row["title"])] = row

    hard_rows = []
    for key in hard:
        row = lookup[key]
        hard_rows.append(
            {
                "qid": row["qid"],
                "hop": hop_bucket(row["qid"]),
                "doc_id": row["doc_id"],
                "title": row["title"],
                "text": row["text"],
            }
        )
    (out_dir / "never_all_retrievers.json").write_text(
        json.dumps(hard_rows, ensure_ascii=False, indent=2)
    )
    (out_dir / "never_C6_H2.json").write_text(
        json.dumps(results["C6-H2"]["never_rows"], ensure_ascii=False, indent=2)
    )
    (out_dir / "miss_C6_H2.json").write_text(
        json.dumps(results["C6-H2"]["miss_rows"], ensure_ascii=False, indent=2)
    )

    print("\n=== 三路都捞不到 (sample 20) ===")
    for row in hard_rows[:20]:
        print(f"  {row['hop']:4s} {row['qid']}")
        print(f"       [{row['doc_id']}] {row['title'][:60]}")
        print(f"       {row['text'][:120]}")

    print("\n=== C6-H2 池内未进 top10 且全程 grade<4 (sample 15) ===")
    hard_miss = [
        row
        for row in results["C6-H2"]["miss_rows"]
        if not any((g.get("grade") or 0) >= 4 for g in row["grades"])
    ]
    print(f"count={len(hard_miss)} / miss={results['C6-H2']['miss']}")
    for row in hard_miss[:15]:
        gs = ",".join(
            f"h{g['hop']}:g{g['grade']}(ev{g['evidence_n']})" for g in row["grades"][:4]
        )
        print(f"  {row['hop']:4s} {row['qid']} in_h0={row['in_h0']} | {row['title'][:50]}")
        print(f"       grades: {gs}")
        print(f"       {row['text'][:110]}")

    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
