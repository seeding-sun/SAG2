import asyncio
import json

import pytest

from qwen_evidence_scorer import (
    QwenEvidenceScorer,
    QwenProtocolError,
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
    assert usage["judgments"] == 2
    assert usage["requests"] == 1
    assert usage["input_tokens"] == 20
