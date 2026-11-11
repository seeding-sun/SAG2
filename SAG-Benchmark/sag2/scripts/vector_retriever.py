"""NumPy dense retrieval over a validated chunk embedding cache."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embedding_client import EmbeddingClient
from sag2 import CorpusIndex, compact_json


@dataclass(frozen=True)
class VectorCache:
    vectors: np.ndarray
    ids: list[int]
    manifest: dict

    @classmethod
    def load(cls, cache_dir: Path, dataset: str) -> "VectorCache":
        stem = cache_dir / f"{dataset}-bge-m3"
        vectors_path = stem.with_suffix(".npy")
        ids_path = stem.with_suffix(".ids.json")
        manifest_path = stem.with_suffix(".manifest.json")
        for path in (vectors_path, ids_path, manifest_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing vector cache file: {path}")
        vectors = np.load(vectors_path, mmap_mode="r")
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if vectors.ndim != 2 or vectors.shape[0] != len(ids):
            raise ValueError(
                f"Vector cache shape {vectors.shape} does not match {len(ids)} ids"
            )
        if len(ids) != len(set(ids)):
            raise ValueError("Vector cache contains duplicate chunk ids")
        if manifest.get("count") != len(ids) or manifest.get("dimension") != vectors.shape[1]:
            raise ValueError("Vector cache manifest does not match ids or dimensions")
        if not manifest.get("normalized"):
            raise ValueError("Vector cache must be L2-normalized for cosine retrieval")
        sample = np.asarray(vectors[: min(32, len(ids))], dtype=np.float32)
        if not np.isfinite(sample).all():
            raise ValueError("Vector cache contains invalid values")
        return cls(vectors=vectors, ids=ids, manifest=manifest)

    def validate_index(self, index: CorpusIndex, corpus_sha256: str | None) -> None:
        if corpus_sha256 and self.manifest.get("corpus_sha256") != corpus_sha256:
            raise ValueError("Vector cache corpus hash does not match the current corpus")
        count = index.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if count != len(self.ids):
            raise ValueError(
                f"Vector cache has {len(self.ids)} ids but index has {count} chunks"
            )
        missing = index.db.execute(
            "SELECT COUNT(*) FROM chunks WHERE id NOT IN "
            "(SELECT value FROM json_each(?))",
            (compact_json(sorted(self.ids)),),
        ).fetchone()[0]
        if missing:
            raise ValueError("Vector cache ids do not match the current BM25 index")


class VectorRetriever:
    name = "dense"

    def __init__(self, client: EmbeddingClient, cache: VectorCache):
        self.client = client
        self.cache = cache
        self.id_to_position = {chunk_id: i for i, chunk_id in enumerate(cache.ids)}

    @staticmethod
    def _fetch_chunks(index: CorpusIndex, ids: list[int]) -> list[dict]:
        if not ids:
            return []
        records = {
            row["id"]: dict(row)
            for row in index.db.execute(
                "SELECT * FROM chunks WHERE id IN (SELECT value FROM json_each(?))",
                (compact_json(ids),),
            )
        }
        missing = set(ids) - set(records)
        if missing:
            raise ValueError(f"Chunk ids missing from index: {sorted(missing)[:10]}")
        if records and "content" not in records[ids[0]]:
            raise ValueError(
                "Vector retrieval currently requires a plain-text index with stored chunk content"
            )
        return [records[chunk_id] for chunk_id in ids]

    async def search(
        self,
        index: CorpusIndex,
        query: str,
        *,
        limit: int,
        excluded: set[int] | None = None,
    ) -> tuple[list[dict], dict]:
        if limit < 1:
            raise ValueError("Vector retrieval limit must be positive")
        excluded = set(excluded or ())
        query_vector, embedding_usage = await self.client.embed_query(query)
        if query_vector.shape[0] != self.cache.vectors.shape[1]:
            raise ValueError("Query embedding dimension does not match vector cache")
        scores = np.asarray(self.cache.vectors @ query_vector, dtype=np.float32)
        if excluded:
            blocked = np.fromiter(
                (chunk_id in excluded for chunk_id in self.cache.ids),
                dtype=bool,
                count=len(self.cache.ids),
            )
            scores[blocked] = -np.inf
        eligible = int(np.isfinite(scores).sum())
        if eligible == 0:
            return [], {
                "retriever": self.name,
                "candidate_count": 0,
                "embedding": embedding_usage,
            }
        take = min(limit, eligible)
        if take == len(scores):
            positions = np.argsort(-scores, kind="stable")
        else:
            partial = np.argpartition(-scores, take - 1)[:take]
            positions = partial[np.argsort(-scores[partial], kind="stable")]
        positions = [int(position) for position in positions[:take]]
        ids = [self.cache.ids[position] for position in positions]
        rows = self._fetch_chunks(index, ids)
        hits = []
        for rank, (row, position) in enumerate(zip(rows, positions, strict=True), start=1):
            score = float(scores[position])
            hits.append(
                {
                    **row,
                    "score": score,
                    "dense_score": score,
                    "dense_rank": rank,
                    "retrieval_route": "dense",
                }
            )
        return hits, {
            "retriever": self.name,
            "candidate_count": len(hits),
            "eligible_count": eligible,
            "embedding": embedding_usage,
        }
