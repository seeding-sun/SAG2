import asyncio

import sag2
from hybrid_retriever import HybridRetriever


class FixedScorer:
    async def score(self, query, candidates, evidence, *, context=None):
        return [
            {**candidate, "jev_score": 1.0, "jev_model": "fixed"}
            for candidate in candidates
        ], {
            "requests": 1,
            "judgments": len(candidates),
            "input_tokens": 0,
            "output_tokens": 0,
            "batches": [],
        }


class FixedInitialRetriever:
    name = "fixed-initial"

    async def search(self, index, query, *, excluded, limit):
        rows = index.retrieve(index.match_query(query), excluded, limit)
        rows.reverse()
        return rows, {"retriever": self.name, "candidate_count": len(rows)}


class FixedVectorRetriever:
    async def search(self, index, query, *, limit, excluded=None):
        rows = []
        excluded = excluded or set()
        for document_id in (2, 1):
            row = index.db.execute(
                "SELECT * FROM chunks WHERE document_id=? AND id NOT IN "
                "(SELECT value FROM json_each(?)) LIMIT 1",
                (document_id, sag2.compact_json(sorted(excluded))),
            ).fetchone()
            if row is not None:
                rows.append(
                    {
                        **dict(row),
                        "score": 1.0 / (document_id),
                        "dense_score": 1.0 / document_id,
                        "dense_rank": len(rows) + 1,
                    }
                )
        return rows[:limit], {"retriever": "dense"}


def make_index(tmp_path):
    return sag2.CorpusIndex(
        tmp_path / "hybrid.sqlite",
        [
            {"title": "Anchor", "text": "anchor first"},
            {"title": "Bridge", "text": "anchor second"},
        ],
        "hybrid-fixture",
    )


def test_search_sag2_uses_initial_retriever_only_at_hop_zero(tmp_path):
    index = make_index(tmp_path)
    try:
        result = asyncio.run(
            sag2.search_sag2(
                index,
                FixedScorer(),
                "anchor",
                candidate_k=2,
                seed_k=1,
                max_hops=0,
                initial_retriever=FixedInitialRetriever(),
            )
        )
        assert result["initial_retriever"] == "fixed-initial"
        assert result["retrieval_usage"]["retriever"] == "fixed-initial"
        assert result["trace"][0]["stage"] == "fixed-initial"
        assert result["baseline"][0]["title"] == "Bridge"
    finally:
        index.close()


def test_hybrid_retriever_fuses_routes_and_keeps_rank_metadata(tmp_path):
    index = make_index(tmp_path)
    try:
        retriever = HybridRetriever(
            FixedVectorRetriever(),
            pool_k=2,
            rrf_k=60,
        )
        hits, usage = asyncio.run(
            retriever.search(index, "anchor", excluded=set(), limit=2)
        )
        assert len(hits) == 2
        assert usage["bm25_count"] == 2
        assert usage["dense_count"] == 2
        assert usage["union_count"] == 2
        assert {row["title"] for row in hits} == {"Anchor", "Bridge"}
        for row in hits:
            assert set(row["retrieved_by"]) == {"bm25", "dense"}
            assert row["bm25_rank"] in (1, 2)
            assert row["dense_rank"] in (1, 2)
            assert row["rrf_score"] > 0
    finally:
        index.close()
