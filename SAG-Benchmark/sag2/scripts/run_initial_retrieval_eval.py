#!/usr/bin/env python3
"""Evaluate BM25, BGE-M3 and hybrid initial retrieval without a scorer."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

from embedding_client import EmbeddingClient
from hybrid_retriever import HybridRetriever
from run_hybrid_compare import load_subset
from run_sag2_benchmark import (
    ROOT,
    corpus_fingerprint,
    load_dataset,
    recall_metrics,
    unique_documents,
    write_json,
)
from sag2 import DEFAULT_CHUNK_CHARS, CorpusIndex
from sample_identity import index_unique_records
from vector_retriever import VectorCache, VectorRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--sample-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "sag2")
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k-list", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--index-workers", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--pool-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-api-key-env", default="SAG_EMBEDDING_API_KEY")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-concurrency", type=int, default=2)
    parser.add_argument("--embedding-dimension", type=int, default=1024)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    return parser


def evaluate(candidates: list[dict], gold: list[str], args: argparse.Namespace) -> dict:
    docs = unique_documents(candidates, args.candidate_k)
    returned = unique_documents(candidates, args.top_k)
    gold_set = set(gold)
    docs_set = {row["full_doc"] for row in docs}
    return {
        "candidate_ids": [row["id"] for row in docs],
        "candidate_docs": [row["full_doc"] for row in docs],
        "returned_docs": [row["full_doc"] for row in returned],
        "metrics": recall_metrics(
            gold, [row["full_doc"] for row in returned], args.k_list
        ),
        "all_evidence_metrics": {
            f"AllEvidence@{k}": int(
                gold_set <= {row["full_doc"] for row in returned[:k]}
            )
            for k in args.k_list
        },
        "entry_recall": int(bool(gold_set & docs_set)),
        "full_recall": int(gold_set <= docs_set),
    }


async def run(args: argparse.Namespace) -> None:
    load_dotenv(args.env_file)
    corpus, records, unit = load_dataset(args.dataset, args.dataset_dir)
    selected = load_subset(args.sample_file, records)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    corpus_sha256 = corpus_fingerprint(
        args.dataset_dir / f"{args.dataset}_corpus.json"
    )
    vector_cache = VectorCache.load(args.cache_dir, args.dataset)
    config = {
        "dataset": args.dataset,
        "algorithm": "initial retrieval only",
        "metric_unit": unit,
        "sample_file": str(args.sample_file.resolve()),
        "sample_count": len(selected),
        "candidate_k": args.candidate_k,
        "top_k": args.top_k,
        "k_list": args.k_list,
        "pool_k": args.pool_k,
        "rrf_k": args.rrf_k,
        "chunk_chars": args.chunk_chars,
        "corpus_sha256": corpus_sha256,
        "embedding_model": args.embedding_model or os.getenv("SAG_EMBEDDING_MODEL"),
        "embedding_dimension": args.embedding_dimension,
    }
    write_json(output / "config.json", config)
    index = CorpusIndex(
        args.cache_dir / f"{args.dataset}.sqlite",
        corpus,
        corpus_sha256,
        args.chunk_chars,
        index_workers=args.index_workers,
    )
    vector_cache.validate_index(index, corpus_sha256)
    base_url = args.embedding_base_url or os.getenv("SAG_EMBEDDING_BASE_URL")
    model = args.embedding_model or os.getenv("SAG_EMBEDDING_MODEL")
    if not base_url or not model:
        raise ValueError("Embedding base URL and model must be configured")
    started = time.perf_counter()
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=args.timeout)
        ) as session:
            client = EmbeddingClient(
                session,
                base_url,
                model,
                os.getenv(args.embedding_api_key_env, ""),
                journal_dir=args.cache_dir / "request-cache" / "query-embedding",
                batch_size=args.embedding_batch_size,
                concurrency=args.embedding_concurrency,
                expected_dimension=args.embedding_dimension,
            )
            dense = VectorRetriever(client, vector_cache)
            hybrid = HybridRetriever(dense, pool_k=args.pool_k, rrf_k=args.rrf_k)
            semaphore = asyncio.Semaphore(args.concurrency)
            rows = []

            async def evaluate_question(record: dict) -> dict:
                async with semaphore:
                    expression = index.match_query(record["question"])
                    bm25_hits = (
                        index.retrieve(expression, set(), args.candidate_k)
                        if expression
                        else []
                    )
                    dense_hits, _ = await dense.search(
                        index,
                        record["question"],
                        limit=args.candidate_k,
                        excluded=set(),
                    )
                    hybrid_hits, _ = await hybrid.search(
                        index,
                        record["question"],
                        excluded=set(),
                        limit=args.candidate_k,
                    )
                    bm25 = evaluate(bm25_hits, record["gold_docs"], args)
                    dense_result = evaluate(dense_hits, record["gold_docs"], args)
                    hybrid_result = evaluate(hybrid_hits, record["gold_docs"], args)
                    added = set(hybrid_result["candidate_ids"]) - set(bm25["candidate_ids"])
                    hybrid_gold = {
                        row["id"]
                        for row in unique_documents(hybrid_hits, args.candidate_k)
                        if row["full_doc"] in set(record["gold_docs"])
                    }
                    added_gold = len(added & hybrid_gold)
                    return {
                        "dataset_sample_id": record["id"],
                        "question": record["question"],
                        "bm25": bm25,
                        "dense": dense_result,
                        "hybrid": hybrid_result,
                        "hybrid_added_candidates": len(added),
                        "hybrid_added_gold": added_gold,
                        "hybrid_noise_rate": (
                            (len(added) - added_gold) / len(added) if added else 0.0
                        ),
                    }

            tasks = [evaluate_question(record) for record in selected]
            for future in asyncio.as_completed(tasks):
                row = await future
                rows.append(row)
                if len(rows) % 10 == 0 or len(rows) == len(selected):
                    print(
                        f"Completed {len(rows)}/{len(selected)} in "
                        f"{time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
    finally:
        index.close()
    by_id = index_unique_records(rows, "dataset_sample_id", "retrieval rows")
    ordered = [by_id[record["id"]] for record in selected]
    with (output / "details.jsonl").open("w", encoding="utf-8") as stream:
        for row in ordered:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "status": "completed",
        "question_count": len(ordered),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "modes": {},
    }
    for mode in ("bm25", "dense", "hybrid"):
        summary["modes"][mode] = {
            "metrics": {
                key: sum(row[mode]["metrics"][key] for row in ordered) / len(ordered)
                for key in ordered[0][mode]["metrics"]
            },
            "all_evidence_metrics": {
                key: sum(
                    row[mode]["all_evidence_metrics"][key] for row in ordered
                )
                / len(ordered)
                for key in ordered[0][mode]["all_evidence_metrics"]
            },
            "entry_recall": sum(row[mode]["entry_recall"] for row in ordered)
            / len(ordered),
            "full_recall": sum(row[mode]["full_recall"] for row in ordered)
            / len(ordered),
        }
    summary["hybrid_initial_delta"] = {
        "added_candidates": sum(row["hybrid_added_candidates"] for row in ordered),
        "added_gold": sum(row["hybrid_added_gold"] for row in ordered),
        "mean_noise_rate": sum(row["hybrid_noise_rate"] for row in ordered)
        / len(ordered),
    }
    write_json(output / "metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
