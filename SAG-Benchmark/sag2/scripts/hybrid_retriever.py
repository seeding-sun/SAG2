"""RRF fusion of the existing BM25 retriever and BGE-M3 dense retrieval."""

from __future__ import annotations

from sag2 import CorpusIndex
from vector_retriever import VectorRetriever


class HybridRetriever:
    name = "hybrid"

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        *,
        pool_k: int = 100,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ):
        if pool_k < 1 or rrf_k < 0:
            raise ValueError("pool_k must be positive and rrf_k must be non-negative")
        if bm25_weight <= 0 or dense_weight <= 0:
            raise ValueError("RRF route weights must be positive")
        self.vector_retriever = vector_retriever
        self.pool_k = pool_k
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    async def search(
        self,
        index: CorpusIndex,
        query: str,
        *,
        excluded: set[int],
        limit: int,
    ) -> tuple[list[dict], dict]:
        expression = index.match_query(query)
        bm25_hits = (
            index.retrieve(expression, excluded, self.pool_k) if expression else []
        )
        dense_hits, dense_usage = await self.vector_retriever.search(
            index,
            query,
            limit=self.pool_k,
            excluded=excluded,
        )
        records: dict[int, dict] = {}
        bm25_ranks: dict[int, int] = {}
        dense_ranks: dict[int, int] = {}
        for rank, hit in enumerate(bm25_hits, start=1):
            chunk_id = hit["id"]
            bm25_ranks[chunk_id] = rank
            records[chunk_id] = {
                **hit,
                "bm25_rank": rank,
                "bm25_score": hit.get("score"),
            }
        for rank, hit in enumerate(dense_hits, start=1):
            chunk_id = hit["id"]
            dense_ranks[chunk_id] = rank
            records.setdefault(chunk_id, dict(hit))
            records[chunk_id].update(
                {
                    "dense_rank": rank,
                    "dense_score": hit["dense_score"],
                }
            )
        fused = []
        for chunk_id, row in records.items():
            bm25_rank = bm25_ranks.get(chunk_id)
            dense_rank = dense_ranks.get(chunk_id)
            score = 0.0
            if bm25_rank is not None:
                score += self.bm25_weight / (self.rrf_k + bm25_rank)
            if dense_rank is not None:
                score += self.dense_weight / (self.rrf_k + dense_rank)
            routes = []
            if bm25_rank is not None:
                routes.append("bm25")
            if dense_rank is not None:
                routes.append("dense")
            route_rank = min(
                rank
                for rank in (bm25_rank, dense_rank)
                if rank is not None
            )
            fused.append(
                {
                    **row,
                    "score": score,
                    "rrf_score": score,
                    "bm25_rank": bm25_rank,
                    "dense_rank": dense_rank,
                    "retrieved_by": routes,
                    "retrieval_route": "hybrid",
                    "_route_rank": route_rank,
                }
            )
        fused.sort(key=lambda row: (-row["rrf_score"], row["_route_rank"], row["id"]))
        fused = fused[:limit]
        for rank, row in enumerate(fused, start=1):
            row["initial_rank"] = rank
            row.pop("_route_rank", None)
        return fused, {
            "retriever": self.name,
            "pool_k": self.pool_k,
            "rrf_k": self.rrf_k,
            "bm25_weight": self.bm25_weight,
            "dense_weight": self.dense_weight,
            "bm25_count": len(bm25_hits),
            "dense_count": len(dense_hits),
            "union_count": len(records),
            "candidate_count": len(fused),
            "dense": dense_usage,
        }
