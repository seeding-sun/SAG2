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


class RuleScorer:
    """Scores by document id so every hop has qualifying seeds and non-qualifying filler."""

    def __init__(self, qualifies):
        self.qualifies = qualifies

    async def score(self, query, candidates, evidence, *, context=None):
        return [
            {
                **candidate,
                "jev_score": 1.0 if self.qualifies(candidate["id"]) else 0.0,
                "jev_model": "rule",
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


def make_deep_index(tmp_path):
    """Enough unseen passages that both expansion rounds retrieve new candidates.

    `search_sag2` breaks out of the hop loop when an expansion round returns nothing,
    so a 12-document corpus can never reach hop 2. Forty documents leave room for
    hop-1 and hop-2 to each pull ten fresh passages.
    """
    corpus = [
        {"title": f"Doc{number}", "text": f"anchor passage number {number}"}
        for number in range(1, 41)
    ]
    return sag2.CorpusIndex(tmp_path / "deep.sqlite", corpus, "deep-fixture")


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
    index = make_deep_index(tmp_path)

    def qualifies(document_id: int) -> bool:
        return document_id % 3 == 0

    result = asyncio.run(
        sag2.search_sag2(
            index,
            RuleScorer(qualifies),
            "anchor",
            candidate_k=10,
            seed_k=2,
            max_hops=2,
            grade_threshold=0.4,
        )
    )
    steps = [step for step in result["trace"] if step["stage"] == "jev"]
    hop_two = steps[-1]
    assert hop_two["hop"] == 2, "hop 2 must actually run for this assertion to mean anything"

    # `latest` holds every judgement so far; only the qualifying ones may resurface.
    seen = {entry["id"] for step in steps[:2] for entry in step["scores"]}
    expected = sum(1 for document_id in seen if qualifies(document_id))
    assert hop_two["previous_eligible_count"] == expected
    assert hop_two["previous_eligible_count"] < len(seen), "the filter must actually drop rows"


def test_hop_two_history_is_unfiltered_at_the_default_threshold(tmp_path):
    """At grade_threshold 0.0 every scored candidate still backfills hop-2 history."""
    index = make_deep_index(tmp_path)
    result = asyncio.run(
        sag2.search_sag2(
            index,
            RuleScorer(lambda document_id: document_id % 3 == 0),
            "anchor",
            candidate_k=10,
            seed_k=2,
            max_hops=2,
        )
    )
    steps = [step for step in result["trace"] if step["stage"] == "jev"]
    hop_two = steps[-1]
    assert hop_two["hop"] == 2
    seen = {entry["id"] for step in steps[:2] for entry in step["scores"]}
    # Nothing is filtered, so the whole accumulated history is eligible.
    assert hop_two["previous_eligible_count"] == len(seen)
    assert hop_two["previous_eligible_count"] > 1


def test_a_threshold_inside_a_grade_gap_is_reported_as_a_no_op(caplog):
    """Grades are quantised to {0,2,4,6,8,10}, so tau=0.3 selects exactly the tau=0.4 set.

    Silently accepting 0.3 would let an experiment claim a recalibration that never
    happened, which is exactly the mistake this warning exists to catch.
    """
    import logging

    import run_hybrid_compare as runner

    parser = runner.build_parser()
    args = parser.parse_args(
        [
            "--retrieval-mode", "bm25",
            "--sample-file", "sample.json",
            "--output-dir", "out",
            "--scoring-scale", "grade",
            "--grade-threshold", "0.3",
        ]
    )
    with caplog.at_level(logging.WARNING):
        runner.validate_args(parser, args)
    assert any("same candidates" in record.getMessage() for record in caplog.records)


def test_a_threshold_on_a_grade_anchor_is_not_warned(caplog):
    import logging

    import run_hybrid_compare as runner

    parser = runner.build_parser()
    args = parser.parse_args(
        [
            "--retrieval-mode", "bm25",
            "--sample-file", "sample.json",
            "--output-dir", "out",
            "--scoring-scale", "grade",
            "--grade-threshold", "0.2",
        ]
    )
    with caplog.at_level(logging.WARNING):
        runner.validate_args(parser, args)
    assert not [r for r in caplog.records if "same candidates" in r.getMessage()]
