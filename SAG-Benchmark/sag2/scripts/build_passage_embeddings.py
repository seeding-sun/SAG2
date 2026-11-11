#!/usr/bin/env python3
"""Build a reusable BGE-M3 embedding cache for the current BM25 chunk index."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import aiohttp
import numpy as np
from dotenv import load_dotenv

from embedding_client import EmbeddingClient
from sag2 import CHUNK_POLICY, DEFAULT_CHUNK_CHARS, corpus_fingerprint

ROOT = Path(__file__).resolve().parents[1]
TEXT_TEMPLATE = "title\\nheading\\ncontent"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


def expected_manifest(
    *,
    dataset: str,
    corpus_sha256: str,
    index_signature: str,
    model: str,
    base_url: str,
    dimension: int,
    count: int,
    batch_size: int,
    concurrency: int,
) -> dict:
    return {
        "schema_version": 1,
        "dataset": dataset,
        "corpus_sha256": corpus_sha256,
        "index_signature": index_signature,
        "chunk_policy": CHUNK_POLICY,
        "chunk_chars": DEFAULT_CHUNK_CHARS,
        "retrieval_unit": "chunk",
        "embedding_model": model,
        "embedding_base_url": base_url.rstrip("/"),
        "dimension": dimension,
        "count": count,
        "normalized": True,
        "batch_size": batch_size,
        "concurrency": concurrency,
        "text_template": TEXT_TEMPLATE,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def load_chunks(index_path: Path) -> tuple[list[dict], str]:
    database = sqlite3.connect(index_path)
    database.row_factory = sqlite3.Row
    try:
        row = database.execute("SELECT signature FROM metadata LIMIT 1").fetchone()
        if row is None:
            raise ValueError("BM25 index metadata has no signature")
        signature = row["signature"]
        columns = {
            row["name"]
            for row in database.execute("PRAGMA table_info(chunks)")
        }
        if "content" not in columns or "full_doc" not in columns:
            raise ValueError(
                "Embedding builder requires a plain-text index with content and full_doc"
            )
        chunks = [
            dict(row)
            for row in database.execute(
                "SELECT id,title,heading,content FROM chunks ORDER BY id"
            )
        ]
        return chunks, signature
    finally:
        database.close()


def embedding_text(chunk: dict) -> str:
    parts = [chunk["title"].strip()]
    if chunk["heading"].strip():
        parts.append(chunk["heading"].strip())
    parts.append(chunk["content"])
    return "\n".join(parts)


def cache_is_current(
    stem: Path,
    manifest: dict,
    ids: list[int],
) -> bool:
    vectors_path = stem.with_suffix(".npy")
    ids_path = stem.with_suffix(".ids.json")
    manifest_path = stem.with_suffix(".manifest.json")
    if not all(path.is_file() for path in (vectors_path, ids_path, manifest_path)):
        return False
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    comparable = {
        key: existing.get(key)
        for key in manifest
        if key != "code_sha256"
    }
    expected = {key: value for key, value in manifest.items() if key != "code_sha256"}
    if comparable != expected:
        return False
    existing_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if existing_ids != ids:
        return False
    vectors = np.load(vectors_path, mmap_mode="r")
    return vectors.shape == (len(ids), manifest["dimension"])


async def build(args: argparse.Namespace) -> dict:
    load_dotenv(args.env_file)
    base_url = args.embedding_base_url or os.getenv("SAG_EMBEDDING_BASE_URL")
    model = args.embedding_model or os.getenv("SAG_EMBEDDING_MODEL")
    api_key = os.getenv(args.embedding_api_key_env, "")
    if not base_url or not model:
        raise ValueError("Embedding base URL and model must be configured")
    chunks, index_signature = load_chunks(args.index)
    ids = [chunk["id"] for chunk in chunks]
    corpus_path = args.dataset_dir / f"{args.dataset}_corpus.json"
    corpus_sha256 = corpus_fingerprint(corpus_path)
    cache_dir = args.cache_dir
    stem = cache_dir / f"{args.dataset}-bge-m3"
    initial_manifest = expected_manifest(
        dataset=args.dataset,
        corpus_sha256=corpus_sha256,
        index_signature=index_signature,
        model=model,
        base_url=base_url,
        dimension=args.dimension,
        count=len(chunks),
        batch_size=args.batch_size,
        concurrency=args.concurrency,
    )
    if not args.force and cache_is_current(stem, initial_manifest, ids):
        print(f"Embedding cache is current: {stem}.npy")
        return initial_manifest
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        client = EmbeddingClient(
            session,
            base_url,
            model,
            api_key,
            journal_dir=cache_dir / "request-cache" / "embedding",
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            normalize=True,
            expected_dimension=args.dimension,
        )
        vectors, usage = await client.embed(
            [embedding_text(chunk) for chunk in chunks],
            input_type="passage",
        )
    if vectors.shape != (len(chunks), args.dimension):
        raise ValueError(f"Unexpected embedding matrix shape: {vectors.shape}")
    manifest = {
        **initial_manifest,
        "created_at": datetime.now(UTC).isoformat(),
        "embedding_usage": usage,
    }
    atomic_npy(stem.with_suffix(".npy"), vectors)
    atomic_json(stem.with_suffix(".ids.json"), ids)
    atomic_json(stem.with_suffix(".manifest.json"), manifest)
    print(
        f"Wrote {len(ids)} vectors of dimension {vectors.shape[1]} to "
        f"{stem.with_suffix('.npy')}"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "sag2")
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-api-key-env", default="SAG_EMBEDDING_API_KEY")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    asyncio.run(build(build_parser().parse_args()))


if __name__ == "__main__":
    main()
