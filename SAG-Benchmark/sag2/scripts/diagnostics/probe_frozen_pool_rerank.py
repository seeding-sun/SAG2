#!/usr/bin/env python3
"""Frozen-pool rerank probe (never11): bge-reranker vs yesterday's JEV three-arms.

Same freeze as probe_frozen_pool_three_arms.py. No LLM JEV calls — only HTTP rerank.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import aiohttp

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from hop_anchor import fill_placeholders, has_open_placeholder  # noqa: E402
from probe_frozen_pool_three_arms import (  # noqa: E402
    chunks_for_ids,
    fill_gold_decomp,
    load_jsonl,
    pool_hit,
    recall_at_k,
    self_fill_anchors,
)
from run_hybrid_compare import load_subset  # noqa: E402
from run_sag2_benchmark import load_dataset, write_json  # noqa: E402
from sag2 import DEFAULT_CHUNK_CHARS, CorpusIndex, corpus_fingerprint  # noqa: E402


def doc_text(c: dict, *, max_chars: int) -> str:
    text = (c.get("full_doc") or f"{c.get('title','')}\n{c.get('content','')}").strip()
    if len(text) > max_chars:
        return text[:max_chars]
    return text


async def rerank_once(
    session: aiohttp.ClientSession,
    *,
    url: str,
    model: str,
    query: str,
    documents: list[str],
) -> list[dict]:
    payload = {"model": model, "query": query, "documents": documents}
    async with session.post(url, json=payload) as resp:
        body = await resp.text()
        if resp.status >= 400:
            raise RuntimeError(f"rerank HTTP {resp.status}: {body[:400]}")
        data = json.loads(body)
    return list(data.get("results") or [])


async def rerank_pool(
    session: aiohttp.ClientSession,
    *,
    url: str,
    model: str,
    query: str,
    candidates: list[dict],
    max_chars: int,
    batch_size: int,
) -> list[dict]:
    """Return candidates sorted by max relevance_score (stable id tiebreak)."""
    if not candidates:
        return []
    best: dict[int, float] = {int(c["id"]): float("-inf") for c in candidates}
    texts = [doc_text(c, max_chars=max_chars) for c in candidates]
    for start in range(0, len(candidates), batch_size):
        chunk_c = candidates[start : start + batch_size]
        chunk_t = texts[start : start + batch_size]
        results = await rerank_once(
            session, url=url, model=model, query=query, documents=chunk_t
        )
        for item in results:
            local = int(item["index"])
            cid = int(chunk_c[local]["id"])
            score = float(item["relevance_score"])
            if score > best[cid]:
                best[cid] = score
    id2c = {int(c["id"]): c for c in candidates}
    ranked = []
    for cid, score in best.items():
        row = dict(id2c[cid])
        row["rerank_score"] = score if score != float("-inf") else 0.0
        ranked.append(row)
    ranked.sort(key=lambda c: (-c["rerank_score"], c["id"]))
    return ranked


async def rerank_max_queries(
    session: aiohttp.ClientSession,
    *,
    url: str,
    model: str,
    queries: list[str],
    candidates: list[dict],
    max_chars: int,
    batch_size: int,
) -> list[dict]:
    """Per-doc max score across queries (dual-style)."""
    if not candidates:
        return []
    best: dict[int, float] = {int(c["id"]): float("-inf") for c in candidates}
    id2c = {int(c["id"]): c for c in candidates}
    for q in queries:
        ranked = await rerank_pool(
            session,
            url=url,
            model=model,
            query=q,
            candidates=candidates,
            max_chars=max_chars,
            batch_size=batch_size,
        )
        for c in ranked:
            cid = int(c["id"])
            best[cid] = max(best[cid], float(c["rerank_score"]))
    out = []
    for cid, score in best.items():
        row = dict(id2c[cid])
        row["rerank_score"] = score if score != float("-inf") else 0.0
        out.append(row)
    out.sort(key=lambda c: (-c["rerank_score"], c["id"]))
    return out


async def run(args: argparse.Namespace) -> None:
    mq_rows = load_jsonl(args.mq_run / "details.jsonl")
    dual_rows = load_jsonl(args.dual_run / "details.jsonl")
    musique = {
        row["id"]: row
        for row in json.loads(args.musique_json.read_text(encoding="utf-8"))
    }
    corpus, records, unit = load_dataset("musique", args.dataset_dir)
    selected = load_subset(args.sample_file, records)
    by_id = {r["id"]: r for r in selected}
    corpus_sha = corpus_fingerprint(args.dataset_dir / "musique_corpus.json")
    index = CorpusIndex(
        args.cache_dir / "musique.sqlite", corpus, corpus_sha, DEFAULT_CHUNK_CHARS
    )

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    details = out / "details.jsonl"
    if details.exists():
        details.unlink()

    results = []
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=args.timeout)
    ) as session:
        ordered_ids = [r["id"] for r in selected]
        for i, qid in enumerate(ordered_ids, start=1):
            mq = mq_rows[qid]
            dual = dual_rows.get(qid) or {}
            record = by_id[qid]
            gold = record["gold_docs"]
            question = record["question"]

            pool_ids = list(mq.get("initial_candidate_ids") or [])
            for t in mq.get("trace") or []:
                if t.get("stage") == "jev":
                    for s in t.get("scores") or []:
                        cid = int(s["id"])
                        if cid not in pool_ids:
                            pool_ids.append(cid)
            candidates = chunks_for_ids(index, pool_ids)

            self_anchors = self_fill_anchors(
                dual.get("hop_anchor_templates"),
                {
                    int(k): v
                    for k, v in (dual.get("hop_anchor_answers") or {}).items()
                },
            )
            sample = musique.get(qid) or {}
            oracle_anchors = fill_gold_decomp(
                sample.get("question_decomposition") or []
            )

            print(
                f"[{i}/{len(ordered_ids)}] {qid} pool={len(candidates)} "
                f"self_n={len(self_anchors)} oracle_n={len(oracle_anchors)}"
            )

            rr_orig = await rerank_pool(
                session,
                url=args.rerank_url,
                model=args.rerank_model,
                query=question,
                candidates=candidates,
                max_chars=args.max_chars,
                batch_size=args.batch_size,
            )
            rr_self = await rerank_max_queries(
                session,
                url=args.rerank_url,
                model=args.rerank_model,
                queries=[question, *self_anchors] if self_anchors else [question],
                candidates=candidates,
                max_chars=args.max_chars,
                batch_size=args.batch_size,
            )
            rr_oracle = await rerank_max_queries(
                session,
                url=args.rerank_url,
                model=args.rerank_model,
                queries=[question, *oracle_anchors]
                if oracle_anchors
                else [question],
                candidates=candidates,
                max_chars=args.max_chars,
                batch_size=args.batch_size,
            )

            pool_docs = {c["full_doc"] for c in candidates}
            row = {
                "dataset_sample_id": qid,
                "question": question,
                "pool_size": len(candidates),
                "pool_gold_recall": pool_hit(pool_docs, gold),
                "self_anchors": self_anchors,
                "oracle_anchors": oracle_anchors,
                "metrics": {
                    "rr_orig": {
                        "Recall@10": recall_at_k(
                            [c["full_doc"] for c in rr_orig], gold, 10
                        )
                    },
                    "rr_self": {
                        "Recall@10": recall_at_k(
                            [c["full_doc"] for c in rr_self], gold, 10
                        )
                    },
                    "rr_oracle": {
                        "Recall@10": recall_at_k(
                            [c["full_doc"] for c in rr_oracle], gold, 10
                        )
                    },
                },
                "top10_titles": {
                    "rr_orig": [c["title"] for c in rr_orig[:10]],
                    "rr_self": [c["title"] for c in rr_self[:10]],
                    "rr_oracle": [c["title"] for c in rr_oracle[:10]],
                },
            }
            results.append(row)
            print(
                f"  pool={row['pool_gold_recall']:.2f} "
                f"rr_orig={row['metrics']['rr_orig']['Recall@10']:.2f} "
                f"rr_self={row['metrics']['rr_self']['Recall@10']:.2f} "
                f"rr_oracle={row['metrics']['rr_oracle']['Recall@10']:.2f}"
            )
            details.open("a", encoding="utf-8").write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    n = len(results) or 1
    # yesterday JEV baseline on same freeze
    jev_path = (
        args.mq_run.parent / "frozen-pool-three-arms" / "summary.json"
    )
    jev = {}
    if jev_path.exists():
        jev = json.loads(jev_path.read_text(encoding="utf-8")).get("metrics") or {}

    summary = {
        "status": "complete",
        "question_count": len(results),
        "unit": unit,
        "rerank": {"url": args.rerank_url, "model": args.rerank_model},
        "freeze": "MQ hop0 initial ∪ jev-scored ids (same as frozen three-arms)",
        "metrics": {
            "pool_gold_recall": sum(r["pool_gold_recall"] for r in results) / n,
            "rr_orig_Recall@10": sum(
                r["metrics"]["rr_orig"]["Recall@10"] for r in results
            )
            / n,
            "rr_self_Recall@10": sum(
                r["metrics"]["rr_self"]["Recall@10"] for r in results
            )
            / n,
            "rr_oracle_Recall@10": sum(
                r["metrics"]["rr_oracle"]["Recall@10"] for r in results
            )
            / n,
        },
        "jev_frozen_baseline": jev,
        "per_question": [
            {
                "qid": r["dataset_sample_id"],
                "pool_gold_recall": r["pool_gold_recall"],
                "rr_orig": r["metrics"]["rr_orig"]["Recall@10"],
                "rr_self": r["metrics"]["rr_self"]["Recall@10"],
                "rr_oracle": r["metrics"]["rr_oracle"]["Recall@10"],
            }
            for r in results
        ],
    }
    write_json(out / "summary.json", summary)
    gate = out / "GATE_rerank.md"
    gate.write_text(
        "\n".join(
            [
                "# 冻结池重排探针（never11）",
                "",
                f"模型：`{args.rerank_model}` @ `{args.rerank_url}`",
                "冻结：与昨天三枪相同（MQ 池，不重搜）。",
                "",
                "| | R@10 |",
                "|---|---:|",
                f"| 池内金覆盖 | {summary['metrics']['pool_gold_recall']:.3f} |",
                f"| JEV 原问（昨） | {jev.get('orig_Recall@10', float('nan')):.3f} |",
                f"| JEV 自填（昨） | {jev.get('self_Recall@10', float('nan')):.3f} |",
                f"| JEV oracle（昨） | {jev.get('oracle_Recall@10', float('nan')):.3f} |",
                f"| **rerank 原问** | **{summary['metrics']['rr_orig_Recall@10']:.3f}** |",
                f"| **rerank 自填max** | **{summary['metrics']['rr_self_Recall@10']:.3f}** |",
                f"| **rerank oracle max** | **{summary['metrics']['rr_oracle_Recall@10']:.3f}** |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"metrics": summary["metrics"], "jev": jev}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[4]
    sag2 = Path(__file__).resolve().parents[2]
    p.add_argument(
        "--sample-file",
        type=Path,
        default=repo / "dataset" / "musique_never11_for_mq.json",
    )
    p.add_argument(
        "--mq-run",
        type=Path,
        default=repo / "output" / "probe-never-multiquery-v2-27b" / "C6-MQ-H1-never11",
    )
    p.add_argument(
        "--dual-run",
        type=Path,
        default=repo
        / "output"
        / "probe-never-multiquery-v2-27b"
        / "C6-MQ-H1-dual-never11",
    )
    p.add_argument("--musique-json", type=Path, default=repo / "dataset" / "musique.json")
    p.add_argument("--dataset-dir", type=Path, default=repo / "dataset")
    p.add_argument("--cache-dir", type=Path, default=sag2 / ".cache" / "sag2")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=repo
        / "output"
        / "probe-never-multiquery-v2-27b"
        / "frozen-pool-rerank",
    )
    p.add_argument("--rerank-url", default="http://192.168.2.108:8014/v1/rerank")
    p.add_argument("--rerank-model", default="bge-reranker-v2-m3")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-chars", type=int, default=1500)
    p.add_argument("--timeout", type=int, default=120)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
