import asyncio
import json

import pytest

from qwen_evidence_scorer import (
    QwenEvidenceScorer,
    QwenProtocolError,
    parse_grades,
    parse_scores,
)


def test_parse_scores_accepts_one_json_fence():
    scores = parse_scores(
        '```json\n{"scores":{"candidate_1":0.8,"candidate_2":0.1}}\n```',
        {"candidate_1", "candidate_2"},
    )
    assert scores == {"candidate_1": 0.8, "candidate_2": 0.1}


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"scores":{"candidate_1":0.8}}',
        '{"scores":{"candidate_1":"0.8","candidate_2":0.1}}',
        '{"scores":{"candidate_1":1.2,"candidate_2":0.1}}',
    ],
)
def test_parse_scores_rejects_invalid_protocol(content):
    with pytest.raises(QwenProtocolError):
        parse_scores(content, {"candidate_1", "candidate_2"})


def test_parse_scores_repairs_one_deleted_digit_in_candidate_key():
    repairs: list[tuple[str, str]] = []
    scores = parse_scores(
        '{"scores":{"candidate_2018":0.4,"candidate_1017":0.9}}',
        {"candidate_2018", "candidate_10197"},
        repairs,
    )
    assert scores == {"candidate_2018": 0.4, "candidate_10197": 0.9}
    assert repairs == [("candidate_1017", "candidate_10197")]


@pytest.mark.parametrize(
    "scores_json,expected",
    [
        # no key at all: one expected key missing, nothing unexpected to map
        ('{}', {"candidate_10197"}),
        # unexpected key is not a one-character deletion of the missing key
        ('{"candidate_9999":0.9}', {"candidate_10197"}),
        # two expected keys missing: ambiguous
        ('{"candidate_1017":0.9}', {"candidate_10197", "candidate_5555"}),
    ],
)
def test_parse_scores_does_not_repair_ambiguous_keys(scores_json, expected):
    with pytest.raises(QwenProtocolError):
        parse_scores('{"scores":%s}' % scores_json, expected)


def test_parse_grades_accepts_every_rubric_anchor():
    grades = parse_grades(
        '{"grades":{"candidate_1":0,"candidate_2":4,"candidate_3":10}}',
        {"candidate_1", "candidate_2", "candidate_3"},
    )
    assert grades == {"candidate_1": 0, "candidate_2": 4, "candidate_3": 10}


@pytest.mark.parametrize(
    "content",
    [
        # off-anchor integer
        '{"grades":{"candidate_1":5,"candidate_2":4}}',
        # float, even when numerically on an anchor
        '{"grades":{"candidate_1":4.0,"candidate_2":4}}',
        # string
        '{"grades":{"candidate_1":"4","candidate_2":4}}',
        # out of range
        '{"grades":{"candidate_1":12,"candidate_2":4}}',
        # wrong envelope key
        '{"scores":{"candidate_1":4,"candidate_2":4}}',
    ],
)
def test_parse_grades_rejects_anything_off_the_rubric(content):
    with pytest.raises(QwenProtocolError):
        parse_grades(content, {"candidate_1", "candidate_2"})


def test_parse_grades_repairs_one_deleted_digit_in_candidate_key():
    repairs: list[tuple[str, str]] = []
    grades = parse_grades(
        '{"grades":{"candidate_2018":6,"candidate_1017":10}}',
        {"candidate_2018", "candidate_10197"},
        repairs,
    )
    assert grades == {"candidate_2018": 6, "candidate_10197": 10}
    assert repairs == [("candidate_1017", "candidate_10197")]


def test_parse_scores_still_accepts_the_binary_scale():
    """The legacy path must keep its own contract while grade mode exists beside it."""
    assert parse_scores(
        '{"scores":{"candidate_1":0.0,"candidate_2":1.0}}',
        {"candidate_1", "candidate_2"},
    ) == {"candidate_1": 0.0, "candidate_2": 1.0}


def test_state_never_adds_gold_fields(tmp_path):
    scorer = QwenEvidenceScorer(
        None,
        "http://offline/v1",
        "qwen-test",
        "unused",
        journal_dir=tmp_path,
        response_format=False,
    )
    state = scorer._build_state(
        "question",
        [
            {
                "id": 1,
                "title": "Evidence",
                "content": "passage",
                "previous_jev_score": 0.4,
            }
        ],
        None,
    )
    encoded = json.dumps(state)
    assert "is_supporting" not in encoded
    assert "gold_docs" not in encoded
    assert "answer" not in encoded
    assert state["previously_retrieved_evidence"][0]["previous_jev_score"] == 0.4


class FakeQwenJournal:
    def __init__(self):
        self.calls = 0

    async def post(self, service, endpoint, api_key, payload):
        self.calls += 1
        request = json.loads(payload["messages"][1]["content"])
        scores = {key: 0.75 for key in request["candidates"]}
        return {
            "model": "qwen-fake",
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"scores": scores}),
                    }
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5},
        }


def test_scorer_returns_search_compatible_scores(tmp_path):
    scorer = QwenEvidenceScorer(
        None,
        "http://offline/v1",
        "qwen-test",
        "unused",
        journal_dir=tmp_path,
        batch_size=10,
        response_format=False,
    )
    scorer.journal = FakeQwenJournal()
    candidates = [
        {"id": 1, "title": "One", "content": "first"},
        {"id": 2, "title": "Two", "content": "second"},
    ]
    scored, usage = asyncio.run(scorer.score("question", candidates, []))
    assert [row["id"] for row in scored] == [1, 2]
    assert all(row["jev_score"] == 0.75 for row in scored)
    assert all(row["jev_model"] == "qwen-fake" for row in scored)
    assert all("jev_grade" not in row for row in scored)
    assert usage["judgments"] == 2
    assert usage["requests"] == 1
    assert usage["input_tokens"] == 20
    assert usage["scoring_scale"] == "binary"


class FakeGradeJournal:
    """Echoes back rubric anchors so the grade-mode plumbing can be exercised offline."""

    def __init__(self, grades: dict[str, int]):
        self.grades = grades
        self.payloads: list[dict] = []

    async def post(self, service, endpoint, api_key, payload):
        request = json.loads(payload["messages"][1]["content"])
        self.payloads.append(payload)
        self.payloads[-1]["criterion"] = request["criterion"]
        grades = {
            key: self.grades[key.removeprefix("candidate_")]
            for key in request["candidates"]
        }
        return {
            "model": "qwen-fake",
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"grades": grades}),
                    }
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5},
        }


def test_grade_mode_normalizes_to_unit_interval_and_keeps_the_raw_grade(tmp_path):
    scorer = QwenEvidenceScorer(
        None,
        "http://offline/v1",
        "qwen-test",
        "unused",
        journal_dir=tmp_path,
        batch_size=10,
        response_format=False,
        scoring_scale="grade",
    )
    journal = FakeGradeJournal({"1": 10, "2": 4, "3": 0})
    scorer.journal = journal
    candidates = [
        {"id": 1, "title": "One", "content": "first"},
        {"id": 2, "title": "Two", "content": "second"},
        {"id": 3, "title": "Three", "content": "third"},
    ]
    scored, usage = asyncio.run(scorer.score("question", candidates, []))
    assert [row["jev_grade"] for row in scored] == [10, 4, 0]
    assert [row["jev_score"] for row in scored] == [1.0, 0.4, 0.0]
    assert usage["scoring_scale"] == "grade"
    # The rubric must reach the model, and the binary criterion must not.
    assert "Rate how much this candidate passage contributes" in journal.payloads[0]["criterion"]
    assert "necessary evidence" not in journal.payloads[0]["criterion"]


def test_grade_mode_rejects_an_unknown_scoring_scale(tmp_path):
    with pytest.raises(ValueError):
        QwenEvidenceScorer(
            None,
            "http://offline/v1",
            "qwen-test",
            "unused",
            journal_dir=tmp_path,
            scoring_scale="zero_to_ten",
        )
