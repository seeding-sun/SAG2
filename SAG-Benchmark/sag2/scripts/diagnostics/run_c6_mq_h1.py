#!/usr/bin/env python3
"""Run dense C6-H1 with multi-query hop0 + evidence follow-up on hop1."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import aiohttp

SCRIPTS = Path(__file__).resolve().parents[1]
SAG2_ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SCRIPTS))

from embedding_client import EmbeddingClient  # noqa: E402
from hop_anchor import DualAnchorScorer, HopAnchorPlanner  # noqa: E402
from multi_query_retriever import MultiQueryDenseRetriever  # noqa: E402
from qwen_evidence_scorer import QwenEvidenceScorer, chat_completions_endpoint  # noqa: E402
from run_hybrid_compare import load_subset  # noqa: E402
from run_sag2_benchmark import load_dataset, unique_documents, write_json  # noqa: E402
from sample_identity import index_unique_records  # noqa: E402
from sag2 import (  # noqa: E402
    DEFAULT_CHUNK_CHARS,
    CorpusIndex,
    corpus_fingerprint,
    search_sag2,
)
from vector_retriever import VectorCache, VectorRetriever  # noqa: E402


def hop_recall_from_trace(
    trace: list | None,
    id_to_doc: dict[int, str],
    gold: list[str],
    *,
    top_k: int = 10,
) -> dict[str, float]:
    """H0/H1/H2 R@k from one H2 run's jev stages (no separate H0/H1 re-runs)."""
    out: dict[str, float] = {}
    if not gold or not trace:
        return out
    gold_set = set(gold)
    for stage in trace:
        if not isinstance(stage, dict) or stage.get("stage") != "jev":
            continue
        hop = stage.get("hop")
        if hop is None:
            continue
        scores = stage.get("scores") or []
        ranked_docs: list[str] = []
        seen: set[str] = set()
        for item in scores:
            doc = id_to_doc.get(item.get("id")) or id_to_doc.get(item.get("document_id"))
            if not doc or doc in seen:
                continue
            seen.add(doc)
            ranked_docs.append(doc)
            if len(ranked_docs) >= top_k:
                break
        hit = sum(1 for g in gold if g in set(ranked_docs[:top_k]))
        out[f"H{hop}_Recall@{top_k}"] = hit / len(gold)
    return out


async def run(args: argparse.Namespace) -> None:
    corpus, records, unit = load_dataset("musique", args.dataset_dir)
    selected = load_subset(args.sample_file, records)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    corpus_sha = corpus_fingerprint(args.dataset_dir / "musique_corpus.json")
    cache_path = args.cache_dir / "musique.sqlite"
    vector_cache = VectorCache.load(args.cache_dir, "musique")
    index = CorpusIndex(cache_path, corpus, corpus_sha, DEFAULT_CHUNK_CHARS)
    vector_cache.validate_index(index, corpus_sha)

    evidence_model = args.evidence_model
    rewrite_model = args.rewrite_model or evidence_model
    config = {
        "dataset": "musique",
        "retrieval_mode": "dense-mq",
        "algorithm": "SAG2 multi-query + followup",
        "metric_unit": unit,
        "sample_file": str(args.sample_file.resolve()),
        "sample_count": len(selected),
        "evidence_model": evidence_model,
        "rewrite_model": rewrite_model,
        "max_hops": args.max_hops,
        "scoring_scale": args.scoring_scale,
        "grade_threshold": args.grade_threshold,
        "candidate_k": args.candidate_k,
        "seed_k": args.seed_k,
        "top_k": args.top_k,
        "followup": True,
        "dual_anchor": bool(args.dual_anchor),
        "dual_anchor_mode": args.dual_anchor_mode,
        "dual_anchor_alpha": args.dual_anchor_alpha,
        "bridge_gate": bool(args.bridge_gate),
        "entity_tau": args.entity_tau,
        "verify_retrieve": bool(args.verify_retrieve),
        "slot_conflict": bool(args.slot_conflict),
        "passage_verify": bool(args.passage_verify),
        "slot_semantic": bool(args.slot_semantic),
        "uncertain_scale": args.uncertain_scale,
        "verify_top_n": args.verify_top_n,
    }
    write_json(output / "config.json", config)

    checkpoint_path = output / "details.jsonl"
    completed = []
    if checkpoint_path.exists() and args.resume:
        completed = [
            json.loads(line)
            for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    done = index_unique_records(completed, "dataset_sample_id", "checkpoint") if completed else {}
    pending = [r for r in selected if r["id"] not in done]
    print(f"pending {len(pending)}/{len(selected)}")

    evidence_base = os.environ["SAG_EVIDENCE_BASE_URL"]
    evidence_key = os.environ["SAG_EVIDENCE_API_KEY"]
    emb_base = os.environ["SAG_EMBEDDING_BASE_URL"]
    emb_model = os.environ["SAG_EMBEDDING_MODEL"]
    emb_key = os.environ.get("SAG_EMBEDDING_API_KEY", "")
    endpoint = chat_completions_endpoint(evidence_base)

    started = time.perf_counter()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=args.timeout)
    ) as session:
        emb_client = EmbeddingClient(
            session,
            emb_base,
            emb_model,
            emb_key,
            journal_dir=args.cache_dir / "request-cache" / "query-embedding",
            batch_size=32,
            concurrency=2,
            normalize=True,
            expected_dimension=1024,
        )
        vector = VectorRetriever(emb_client, vector_cache)
        mq = MultiQueryDenseRetriever(
            vector,
            session=session,
            endpoint=endpoint,
            model=rewrite_model,
            api_key=evidence_key,
            per_query_k=args.per_query_k,
        )

        with checkpoint_path.open("a", encoding="utf-8") as out_f:
            for i, record in enumerate(pending, start=1):
                base_scorer = QwenEvidenceScorer(
                    session,
                    evidence_base,
                    evidence_model,
                    evidence_key,
                    journal_dir=output / "requests" / str(record["id"]) / "qwen",
                    batch_size=args.evidence_batch_size,
                    concurrency=1,
                    response_format=True,
                    scoring_scale=args.scoring_scale,
                    fill_missing_score=0,
                    max_output_tokens_per_candidate=64,
                    strict_model=True,
                )
                planner = None
                scorer = base_scorer
                if args.dual_anchor:
                    planner = HopAnchorPlanner(
                        session=session,
                        endpoint=endpoint,
                        model=rewrite_model,
                        api_key=evidence_key,
                        vector_retriever=vector,
                        per_query_k=min(40, args.per_query_k),
                        verify=bool(args.verify_retrieve),
                        gate=bool(args.bridge_gate),
                        entity_tau=args.entity_tau,
                        slot_conflict=bool(args.slot_conflict),
                        passage_verify=bool(args.passage_verify),
                        slot_semantic=bool(args.slot_semantic),
                        uncertain_scale=args.uncertain_scale,
                        verify_top_n=args.verify_top_n,
                    )
                    scorer = DualAnchorScorer(
                        base_scorer,
                        planner,
                        mode=args.dual_anchor_mode,
                        alpha=args.dual_anchor_alpha,
                    )
                result = await search_sag2(
                    index,
                    scorer,
                    record["question"],
                    candidate_k=args.candidate_k,
                    seed_k=args.seed_k,
                    max_hops=args.max_hops,
                    grade_threshold=args.grade_threshold,
                    initial_retriever=mq,
                    followup_rewriter=mq,
                    followup_vector=vector,
                    followup_per_query_k=args.per_query_k,
                    hop_planner=planner,
                )
                initial = result.pop("baseline")
                all_ranked = result.pop("candidates")
                initial_docs = unique_documents(initial, args.candidate_k)
                retrieved = unique_documents(all_ranked, args.top_k)
                gold = record["gold_docs"]
                gold_set = set(gold)

                metrics = {
                    f"Recall@{k}": (
                        sum(1 for g in gold if g in {c["full_doc"] for c in retrieved[:k]})
                        / len(gold)
                        if gold
                        else 0.0
                    )
                    for k in (1, 2, 5, 10)
                }
                id_to_doc = {
                    c["id"]: c["full_doc"]
                    for c in [*initial_docs, *all_ranked]
                    if "id" in c and "full_doc" in c
                }
                # Resolve any hop-score ids that fell out of final top-candidate_k.
                missing: set[int] = set()
                for stage in result.get("trace") or []:
                    if not isinstance(stage, dict) or stage.get("stage") != "jev":
                        continue
                    for item in stage.get("scores") or []:
                        cid = item.get("id")
                        if isinstance(cid, int) and cid not in id_to_doc:
                            missing.add(cid)
                if missing:
                    db = getattr(index, "db", None)
                    if db is not None:
                        try:
                            rows_db = db.execute(
                                f"SELECT id, full_doc FROM chunks WHERE id IN ({','.join('?' * len(missing))})",
                                tuple(missing),
                            ).fetchall()
                            for row in rows_db:
                                id_to_doc[int(row[0] if not hasattr(row, "keys") else row["id"])] = (
                                    row[1] if not hasattr(row, "keys") else row["full_doc"]
                                )
                        except Exception as exc:  # noqa: BLE001 — hop_metrics best-effort
                            print(f"warn: hop id resolve failed ({len(missing)} ids): {exc}")
                hop_metrics = hop_recall_from_trace(
                    result.get("trace"), id_to_doc, gold, top_k=args.top_k
                )
                # Final ranking after max_hops (includes DPP on top-10) is the Hn number.
                hop_metrics[f"H{args.max_hops}_Recall@{args.top_k}"] = metrics[
                    f"Recall@{args.top_k}"
                ]
                row = {
                    "dataset_sample_id": record["id"],
                    "question": record["question"],
                    "retrieved_docs": [c["full_doc"] for c in retrieved],
                    "retrieved_source_ids": [c["source_id"] for c in retrieved],
                    "initial_docs": [c["full_doc"] for c in initial_docs],
                    "initial_candidate_ids": [c["id"] for c in initial_docs],
                    "initial_candidates": [
                        {
                            "id": c["id"],
                            "document_id": c["document_id"],
                            "source_id": c["source_id"],
                            "title": c["title"],
                            "is_gold": c["full_doc"] in gold_set,
                        }
                        for c in initial_docs
                    ],
                    "metrics": metrics,
                    "hop_metrics": hop_metrics,
                    "initial_metrics": {
                        f"InitialRecall@{k}": (
                            sum(
                                1
                                for g in gold
                                if g in {c["full_doc"] for c in initial_docs[:k]}
                            )
                            / len(gold)
                            if gold
                            else 0.0
                        )
                        for k in (1, 2, 5, 10)
                    },
                    "trace": result.get("trace"),
                    "jev_usage": result.get("jev_usage"),
                    "elapsed_ms": result.get("elapsed_ms"),
                    "algorithm": result.get("algorithm"),
                    "initial_retriever": result.get("initial_retriever"),
                    "retrieval_usage": result.get("retrieval_usage"),
                    "hop_anchor_trace": result.get("hop_anchor_trace"),
                    "hop_anchor_answers": result.get("hop_anchor_answers"),
                    "hop_anchor_templates": result.get("hop_anchor_templates"),
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                print(
                    f"[{i}/{len(pending)}] {record['id']} R@10={metrics['Recall@10']:.2f}"
                )

    # aggregate
    rows = [
        json.loads(l)
        for l in checkpoint_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    metrics = {
        k: sum(r["metrics"][k] for r in rows) / len(rows)
        for k in ("Recall@1", "Recall@2", "Recall@5", "Recall@10")
    }
    hop_keys = sorted(
        {k for r in rows for k in (r.get("hop_metrics") or {})},
        key=lambda x: (int(x.split("_")[0][1:]), x),
    )
    hop_metrics_agg = {
        k: sum((r.get("hop_metrics") or {}).get(k, 0.0) for r in rows) / len(rows)
        for k in hop_keys
    }
    write_json(
        output / "metrics.json",
        {
            "status": "complete",
            "question_count": len(rows),
            "config": config,
            "metrics": metrics,
            "hop_metrics": hop_metrics_agg,
            "wall_seconds": round(time.perf_counter() - started, 1),
        },
    )
    print(json.dumps({"metrics": metrics, "hop_metrics": hop_metrics_agg}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset-dir", type=Path, default=REPO / "dataset")
    p.add_argument("--cache-dir", type=Path, default=SAG2_ROOT / ".cache" / "sag2")
    p.add_argument("--evidence-model", default="qwen3.8-27b")
    p.add_argument("--rewrite-model", default="qwen3.8-27b")
    p.add_argument("--max-hops", type=int, default=1)
    p.add_argument(
        "--scoring-scale",
        choices=("binary", "grade"),
        default="grade",
        help="binary=0/1; grade=six-level rubric (0/2/4/6/8/10)",
    )
    p.add_argument("--grade-threshold", type=float, default=0.4)
    p.add_argument("--candidate-k", type=int, default=100)
    p.add_argument("--seed-k", type=int, default=10)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--per-query-k", type=int, default=80)
    p.add_argument("--evidence-batch-size", type=int, default=10)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--dual-anchor",
        action="store_true",
        help="Score against hop sub-questions as well as the original question",
    )
    p.add_argument(
        "--dual-anchor-mode",
        choices=("max", "blend"),
        default="max",
        help="How to merge orig vs hop grades",
    )
    p.add_argument("--dual-anchor-alpha", type=float, default=0.7)
    p.add_argument(
        "--verify-retrieve",
        action="store_true",
        default=True,
        help="After extracting an entity, force relation second-pass dense retrieve",
    )
    p.add_argument("--no-verify-retrieve", action="store_false", dest="verify_retrieve")
    p.add_argument(
        "--bridge-gate",
        action="store_true",
        default=True,
        help="Enable §5.8 admission gates for structured bridge scoring",
    )
    p.add_argument("--no-bridge-gate", action="store_false", dest="bridge_gate")
    p.add_argument(
        "--entity-tau",
        type=float,
        default=0.6,
        help="Min prior-hop jev_score for entity E before bridge scoring",
    )
    p.add_argument(
        "--slot-conflict",
        action="store_true",
        default=True,
        help="Close bridge scoring when same E+R yields ≥2 distinct answers",
    )
    p.add_argument("--no-slot-conflict", action="store_false", dest="slot_conflict")
    p.add_argument(
        "--passage-verify",
        action="store_true",
        default=False,
        help="Per-passage valid/invalid/uncertain gate on s2 for HQ/birth_place",
    )
    p.add_argument("--no-passage-verify", action="store_false", dest="passage_verify")
    p.add_argument(
        "--slot-semantic",
        action="store_true",
        default=False,
        help="Require strong primary-fact support for HQ/birth_place slot values before bridge",
    )
    p.add_argument("--no-slot-semantic", action="store_false", dest="slot_semantic")
    p.add_argument(
        "--uncertain-scale",
        type=float,
        default=0.2,
        help="Scale factor for s2 when verifier returns uncertain",
    )
    p.add_argument(
        "--verify-top-n",
        type=int,
        default=20,
        help="Verify only top-N candidates by raw bridge grade; drop unchecked s2",
    )
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
