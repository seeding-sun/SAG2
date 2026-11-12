"""Seed-selection behaviour under a grade threshold.

The binary scorer used to score almost everything 0.0, so the pipeline padded the
seed set to `seed_k` with unscored candidates. `grade_threshold` changes that: only
candidates at or above the threshold may become seeds, and a short (or empty) seed
set is accepted rather than padded.
"""

import asyncio

import sag2


class RankScorer:
    """Gives fixed scores by document id so seed membership is predictable."""

    def __init__(self, scores: dict[int, float]):
        self.scores = scores
        self.pools: list[list[int]] = []

    async def score(self, query, candidates, evidence, *, context=None):
        self.pools.append([candidate["id"] for candidate in candidates])
        return [
            {
                **candidate,
                "jev_score": self.scores.get(candidate["id"], 0.0),
                "jev_model": "fixed",
            }
            for candidate in candidates
        ], {
            "requests": 1,
            "judgments": len(candidates),
            "input_tokens": 0,
            "output_tokens": 0,
            "batches": [],
        }


def make_index(tmp_path):
    corpus = [
        {"title": f"Doc{number}", "text": f"anchor passage number {number}"}
        for number in range(1, 13)
    ]
    return sag2.CorpusIndex(tmp_path / "grades.sqlite", corpus, "grade-fixture")


def seed_ids(result):
    hop_zero = next(step for step in result["trace"] if step["stage"] == "jev")
    return hop_zero["seed_ids"]


def test_default_threshold_keeps_the_padded_seed_set(tmp_path):
    """grade_threshold defaults to 0.0, so the legacy fill-to-seed_k behaviour survives."""
    index = make_index(tmp_path)
    dataset = index.retrieve(index.match_query("anchor"), set(), 10)
    doc_ids = [row["id"] for row in dataset]
    # Only the top two are scored; everything else is 0.0.
    scorer = RankScorer({doc_ids[0]: 1.0, doc_ids[1]: 0.4})
    result = asyncio.run(
        sag2.search_sag2(
            index, scorer, "anchor", candidate_k=10, seed_k=5, max_hops=0
        )
    )
    assert len(seed_ids(result)) == 5
    assert seed_ids(result)[:2] == doc_ids[:2]


def test_threshold_drops_unsupported_seeds_instead_of_padding(tmp_path):
    index = make_index(tmp_path)
    dataset = index.retrieve(index.match_query("anchor"), set(), 10)
    doc_ids = [row["id"] for row in dataset]
    scorer = RankScorer({doc_ids[0]: 1.0, doc_ids[1]: 0.4})
    result = asyncio.run(
        sag2.search_sag2(
            index,
            scorer,
            "anchor",
            candidate_k=10,
            seed_k=5,
            max_hops=0,
            grade_threshold=0.4,
        )
    )
    # Two candidates clear the bar; the remaining three slots stay empty.
    assert seed_ids(result) == doc_ids[:2]
    hop_zero = next(step for step in result["trace"] if step["stage"] == "jev")
    assert hop_zero["grade_threshold"] == 0.4
    assert hop_zero["eligible_count"] == 2


def test_threshold_above_every_score_skips_expansion_entirely(tmp_path):
    index = make_index(tmp_path)
    dataset = index.retrieve(index.match_query("anchor"), set(), 10)
    doc_ids = [row["id"] for row in dataset]
    scorer = RankScorer({doc_ids[0]: 1.0})
    result = asyncio.run(
        sag2.search_sag2(
            index,
            scorer,
            "anchor",
            candidate_k=10,
            seed_k=5,
            max_hops=1,
            grade_threshold=1.1,
        )
    )
    stages = [step["stage"] for step in result["trace"]]
    assert stages == ["bm25", "jev"]
    assert seed_ids(result) == []
    # Without seeds there is no expansion round, so the scorer ran exactly once.
    assert len(scorer.pools) == 1


def test_eligible_count_matches_the_scored_pool_when_everything_passes(tmp_path):
    """Sanity guard: eligible_count cannot exceed the pool that was actually scored."""
    index = make_index(tmp_path)
    dataset = index.retrieve(index.match_query("anchor"), set(), 10)
    doc_ids = [row["id"] for row in dataset]
    scorer = RankScorer({doc_id: 0.4 for doc_id in doc_ids})
    result = asyncio.run(
        sag2.search_sag2(
            index,
            scorer,
            "anchor",
            candidate_k=10,
            seed_k=10,
            max_hops=0,
            grade_threshold=0.4,
        )
    )
    hop_zero = next(step for step in result["trace"] if step["stage"] == "jev")
    assert hop_zero["eligible_count"] == len(hop_zero["scores"])
    assert len(seed_ids(result)) == 10


def test_hop_two_previous_candidates_respect_the_threshold(tmp_path):
    """hop-2 used to backfill 80 slots from ranks that were all 0.0."""
    index = make_index(tmp_path)
    dataset = index.retrieve(index.match_query("anchor"), set(), 10)
    doc_ids = [row["id"] for row in dataset]
    scorer = RankScorer({doc_ids[0]: 1.0})
    result = asyncio.run(
        sag2.search_sag2(
            index,
            scorer,
            "anchor",
            candidate_k=10,
            seed_k=2,
            max_hops=2,
            grade_threshold=0.4,
        )
    )
    hop_two = [step for step in result["trace"] if step["stage"] == "jev"][-1]
    # Only the single candidate above the threshold may be carried into hop 2.
    assert hop_two["carried_seed_count"] == 1
    assert hop_two["eligible_count"] == 1
