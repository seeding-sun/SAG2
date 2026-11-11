#!/usr/bin/env python3
"""Run one arm of the paired BM25 versus BM25+BGE-M3 SAG2 experiment."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

from embedding_client import EmbeddingClient
from hybrid_retriever import HybridRetriever
from qwen_evidence_scorer import QwenEvidenceScorer
from run_sag2_benchmark import (
    ROOT,
    corpus_fingerprint,
    load_dataset,
    recall_metrics,
    unique_documents,
    write_json,
)
from sag2 import DEFAULT_CHUNK_CHARS, CorpusIndex, search_sag2
from sample_identity import (
    index_unique_records,
    validate_identity_coverage,
    validate_question_identity,
)
from vector_retriever import VectorCache, VectorRetriever


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_subset(path: Path, records: list[dict]) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("ids") if isinstance(payload, dict) else None
    if not isinstance(ids, list) or any(not isinstance(value, str) for value in ids):
        raise ValueError("Sample file must contain a string list in 'ids'")
    if len(ids) != len(set(ids)):
        raise ValueError("Sample file contains duplicate ids")
    by_id = index_unique_records(records, "id", "dataset questions")
    validate_identity_coverage(set(ids), set(ids) & set(by_id), "sample file")
    return [by_id[question_id] for question_id in ids]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval-mode",
        choices=["bm25", "dense", "hybrid"],
        required=True,
    )
    parser.add_argument("--dataset", default="musique")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(r"D:\learn\知识库\RAG\SAG2\dataset"),
    )
    parser.add_argument("--sample-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--seed-k", type=int, default=10)
    parser.add_argument("--max-hops", type=int, default=1)
    parser.add_argument("--expansion-term-limit", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k-list", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--chunk-chars", type=int)
    parser.add_argument("--index-workers", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "sag2")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-api-key-env", default="SAG_EMBEDDING_API_KEY")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-concurrency", type=int, default=2)
    parser.add_argument("--embedding-dimension", type=int, default=1024)
    parser.add_argument("--evidence-base-url")
    parser.add_argument("--evidence-model")
    parser.add_argument("--evidence-api-key-env", default="SAG_EVIDENCE_API_KEY")
    parser.add_argument("--evidence-batch-size", type=int, default=10)
    parser.add_argument("--evidence-concurrency", type=int, default=4)
    parser.add_argument("--no-response-format", action="store_true")
    parser.add_argument("--pool-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for key in (
        "candidate_k",
        "seed_k",
        "top_k",
        "expansion_term_limit",
        "index_workers",
        "concurrency",
        "pool_k",
        "embedding_batch_size",
        "embedding_concurrency",
        "evidence_batch_size",
        "evidence_concurrency",
    ):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if not 0 <= args.max_hops <= 2:
        parser.error("--max-hops must be between 0 and 2")
    if args.seed_k > args.candidate_k or args.top_k > args.candidate_k:
        parser.error("--seed-k and --top-k cannot exceed --candidate-k")
    if not args.k_list or min(args.k_list) <= 0 or max(args.k_list) > args.top_k:
        parser.error("--k-list values must be positive and cannot exceed --top-k")
    if args.rrf_k < 0:
        parser.error("--rrf-k must be non-negative")
    if args.bm25_weight <= 0 or args.dense_weight <= 0:
        parser.error("RRF route weights must be positive")
    if args.output_dir.exists() and not args.resume:
        parser.error("--output-dir already exists; use --resume or choose a new directory")


def candidate_summary(candidates: list[dict], gold: set[str]) -> list[dict]:
    return [
        {
            key: candidate.get(key)
            for key in (
                "id",
                "document_id",
                "source_id",
                "title",
                "bm25_rank",
                "bm25_score",
                "dense_rank",
                "dense_score",
                "rrf_score",
                "initial_rank",
                "retrieved_by",
            )
        }
        | {"is_gold": candidate["full_doc"] in gold}
        for candidate in candidates
    ]


async def run(args: argparse.Namespace) -> None:
    validate_args(build_parser(), args)
    load_dotenv(args.env_file)
    corpus, records, unit = load_dataset(args.dataset, args.dataset_dir)
    selected = load_subset(args.sample_file, records)
    if not selected:
        raise ValueError("Sample file is empty")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    corpus_sha256 = corpus_fingerprint(
        args.dataset_dir / f"{args.dataset}_corpus.json"
    )
    cache_path = args.cache_dir / f"{args.dataset}.sqlite"
    vector_cache = None
    if args.retrieval_mode in {"dense", "hybrid"}:
        vector_cache = VectorCache.load(args.cache_dir, args.dataset)
    config = {
        "dataset": args.dataset,
        "retrieval_mode": args.retrieval_mode,
        "algorithm": "SAG2 with Qwen evidence scorer",
        "metric_unit": unit,
        "sample_file": str(args.sample_file.resolve()),
        "sample_sha256": file_sha256(args.sample_file),
        "sample_count": len(selected),
        "corpus_sha256": corpus_sha256,
        "sag2_code_sha256": file_sha256(ROOT / "scripts" / "sag2.py"),
        "runner_code_sha256": file_sha256(Path(__file__)),
        "qwen_scorer_sha256": file_sha256(ROOT / "scripts" / "qwen_evidence_scorer.py"),
        "candidate_k": args.candidate_k,
        "seed_k": args.seed_k,
        "max_hops": args.max_hops,
        "expansion_term_limit": args.expansion_term_limit,
        "top_k": args.top_k,
        "k_list": args.k_list,
        "chunk_chars": args.chunk_chars or DEFAULT_CHUNK_CHARS,
        "embedding_model": args.embedding_model
        or os.getenv("SAG_EMBEDDING_MODEL"),
        "embedding_dimension": args.embedding_dimension,
        "evidence_model": args.evidence_model
        or os.getenv("SAG_EVIDENCE_MODEL"),
        "evidence_batch_size": args.evidence_batch_size,
        "evidence_concurrency": args.evidence_concurrency,
        "response_format": not args.no_response_format,
        "pool_k": args.pool_k if args.retrieval_mode == "hybrid" else None,
        "rrf_k": args.rrf_k if args.retrieval_mode == "hybrid" else None,
        "bm25_weight": args.bm25_weight if args.retrieval_mode == "hybrid" else None,
        "dense_weight": args.dense_weight if args.retrieval_mode == "hybrid" else None,
        "vector_manifest_sha256": file_sha256(
            args.cache_dir / f"{args.dataset}-bge-m3.manifest.json"
        )
        if vector_cache is not None
        else None,
    }
    config_path = output / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError(f"Resume configuration differs from {config_path}")
    else:
        write_json(config_path, config)

    index = CorpusIndex(
        cache_path,
        corpus,
        corpus_sha256,
        args.chunk_chars or DEFAULT_CHUNK_CHARS,
        index_workers=args.index_workers,
    )
    if vector_cache is not None:
        vector_cache.validate_index(index, corpus_sha256)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    checkpoint_path = output / "details.jsonl"
    completed = []
    if checkpoint_path.exists():
        completed = [
            json.loads(line)
            for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    done = index_unique_records(completed, "dataset_sample_id", "checkpoint")
    expected_ids = {record["id"] for record in selected}
    if set(done) - expected_ids:
        raise ValueError("Checkpoint contains questions outside the fixed sample")
    for record in selected:
        if record["id"] in done:
            validate_question_identity(
                record["question"], done[record["id"]]["question"], record["id"], "checkpoint"
            )
    pending = iter(record for record in selected if record["id"] not in done)
    started = time.perf_counter()
    try:
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=args.timeout),
        ) as session:
            vector_retriever = None
            if args.retrieval_mode in {"dense", "hybrid"}:
                embedding_base_url = args.embedding_base_url or os.getenv(
                    "SAG_EMBEDDING_BASE_URL"
                )
                embedding_model = args.embedding_model or os.getenv(
                    "SAG_EMBEDDING_MODEL"
                )
                if not embedding_base_url or not embedding_model:
                    raise ValueError("Hybrid mode requires embedding base URL and model")
                embedding_client = EmbeddingClient(
                    session,
                    embedding_base_url,
                    embedding_model,
                    os.getenv(args.embedding_api_key_env, ""),
                    journal_dir=args.cache_dir / "request-cache" / "query-embedding",
                    batch_size=args.embedding_batch_size,
                    concurrency=args.embedding_concurrency,
                    normalize=True,
                    expected_dimension=args.embedding_dimension,
                )
                vector_retriever = VectorRetriever(embedding_client, vector_cache)
            evidence_base_url = args.evidence_base_url or os.getenv(
                "SAG_EVIDENCE_BASE_URL"
            )
            evidence_model = args.evidence_model or os.getenv("SAG_EVIDENCE_MODEL")
            if not evidence_base_url or not evidence_model:
                raise ValueError("Evidence base URL and model must be configured")
            evidence_key = os.getenv(args.evidence_api_key_env, "")

            async def worker() -> None:
                for record in pending:
                    retriever = None
                    if args.retrieval_mode == "dense":
                        retriever = vector_retriever
                    elif args.retrieval_mode == "hybrid":
                        retriever = HybridRetriever(
                            vector_retriever,
                            pool_k=args.pool_k,
                            rrf_k=args.rrf_k,
                            bm25_weight=args.bm25_weight,
                            dense_weight=args.dense_weight,
                        )
                    scorer = QwenEvidenceScorer(
                        session,
                        evidence_base_url,
                        evidence_model,
                        evidence_key,
                        journal_dir=output / "requests" / str(record["id"]) / "qwen",
                        batch_size=args.evidence_batch_size,
                        concurrency=args.evidence_concurrency,
                        response_format=not args.no_response_format,
                    )
                    result = await search_sag2(
                        index,
                        scorer,
                        record["question"],
                        candidate_k=args.candidate_k,
                        seed_k=args.seed_k,
                        max_hops=args.max_hops,
                        expansion_term_limit=args.expansion_term_limit,
                        initial_retriever=retriever,
                    )
                    initial = result.pop("baseline")
                    initial_docs = unique_documents(initial, args.candidate_k)
                    retrieved = unique_documents(result.pop("candidates"), args.top_k)
                    gold = record["gold_docs"]
                    gold_set = set(gold)
                    initial_set = {candidate["full_doc"] for candidate in initial_docs}
                    final_set = {candidate["full_doc"] for candidate in retrieved}
                    row = {
                        "dataset_sample_id": record["id"],
                        "question": record["question"],
                        "retrieved_docs": [candidate["full_doc"] for candidate in retrieved],
                        "retrieved_source_ids": [
                            candidate["source_id"] for candidate in retrieved
                        ],
                        "initial_docs": [
                            candidate["full_doc"] for candidate in initial_docs
                        ],
                        "initial_candidate_ids": [
                            candidate["id"] for candidate in initial_docs
                        ],
                        "initial_candidates": candidate_summary(initial, gold_set),
                        "metrics": recall_metrics(
                            gold,
                            [candidate["full_doc"] for candidate in retrieved],
                            args.k_list,
                        ),
                        "all_evidence_metrics": {
                            f"AllEvidence@{k}": int(
                                gold_set <= final_set
                                if k == args.top_k
                                else gold_set
                                <= {candidate["full_doc"] for candidate in retrieved[:k]}
                            )
                            for k in args.k_list
                        },
                        "initial_metrics": recall_metrics(
                            gold,
                            [candidate["full_doc"] for candidate in initial_docs],
                            args.k_list,
                            prefix="InitialRecall",
                        ),
                        "initial_entry_recall": int(bool(gold_set & initial_set)),
                        "initial_full_recall": int(gold_set <= initial_set),
                        **result,
                    }
                    with checkpoint_path.open("a", encoding="utf-8") as stream:
                        stream.write(
                            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                        )
                        stream.flush()
                    done[record["id"]] = row
                    if len(done) % 10 == 0 or len(done) == len(selected):
                        print(
                            f"Completed {len(done)}/{len(selected)} in "
                            f"{time.perf_counter() - started:.1f}s",
                            flush=True,
                        )

            async with asyncio.TaskGroup() as group:
                for _ in range(args.concurrency):
                    group.create_task(worker())
    finally:
        index.close()
    ordered = [done[record["id"]] for record in selected]
    metric_names = list(ordered[0]["metrics"])
    initial_metric_names = list(ordered[0]["initial_metrics"])
    all_evidence_names = list(ordered[0]["all_evidence_metrics"])
    summary = {
        "status": "completed",
        "question_count": len(ordered),
        "config": config,
        "metrics": {
            key: sum(row["metrics"][key] for row in ordered) / len(ordered)
            for key in metric_names
        },
        "initial_metrics": {
            key: sum(row["initial_metrics"][key] for row in ordered) / len(ordered)
            for key in initial_metric_names
        },
        "initial_entry_recall": sum(
            row["initial_entry_recall"] for row in ordered
        )
        / len(ordered),
        "initial_full_recall": sum(
            row["initial_full_recall"] for row in ordered
        )
        / len(ordered),
        "all_evidence_metrics": {
            key: sum(row["all_evidence_metrics"][key] for row in ordered)
            / len(ordered)
            for key in all_evidence_names
        },
        "mean_query_ms": round(
            sum(row["elapsed_ms"] for row in ordered) / len(ordered)
        ),
        "judgments": sum(row["jev_usage"]["judgments"] for row in ordered),
        "requests": sum(row["jev_usage"]["requests"] for row in ordered),
        "input_tokens": sum(row["jev_usage"]["input_tokens"] for row in ordered),
        "output_tokens": sum(row["jev_usage"]["output_tokens"] for row in ordered),
        "wall_seconds_this_invocation": round(time.perf_counter() - started, 3),
        "completed_at": datetime.now(UTC).isoformat(),
    }
    write_json(output / "metrics.json", summary)
    write_json(
        output / "search_results.json",
        [
            {
                "dataset_sample_id": row["dataset_sample_id"],
                "question": row["question"],
                "retrieved_docs": row["retrieved_docs"],
                "metrics": row["metrics"],
                "all_evidence_metrics": row["all_evidence_metrics"],
            }
            for row in ordered
        ],
    )
    print(json.dumps(summary["metrics"], ensure_ascii=False), flush=True)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
