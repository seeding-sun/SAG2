import asyncio

import numpy as np
import pytest

from embedding_client import EmbeddingClient, embeddings_endpoint


class FakeJournal:
    def __init__(self, response):
        self.response = response
        self.records = [{"reused": False, "actual_model": "fake-embed"}]

    async def post(self, service, endpoint, api_key, payload):
        return self.response


def test_embeddings_endpoint_is_idempotent():
    assert (
        embeddings_endpoint("http://localhost:8013/v1")
        == "http://localhost:8013/v1/embeddings"
    )
    assert (
        embeddings_endpoint("http://localhost:8013/v1/embeddings")
        == "http://localhost:8013/v1/embeddings"
    )


def test_embedding_rows_are_reordered_and_normalized(tmp_path):
    response = {
        "model": "fake-embed",
        "data": [
            {"index": 1, "embedding": [0.0, 2.0, 0.0]},
            {"index": 0, "embedding": [3.0, 0.0, 0.0]},
        ],
        "usage": {"prompt_tokens": 4},
    }
    client = EmbeddingClient(
        None,
        "http://offline/v1",
        "fake-embed",
        "",
        journal_dir=tmp_path,
        batch_size=2,
        concurrency=1,
        expected_dimension=3,
    )
    client.journal = FakeJournal(response)
    vectors, usage = asyncio.run(client.embed(["first", "second"]))
    assert vectors.shape == (2, 3)
    assert np.allclose(vectors[0], [1.0, 0.0, 0.0])
    assert np.allclose(vectors[1], [0.0, 1.0, 0.0])
    assert usage["requests"] == 1


def test_embedding_count_mismatch_fails(tmp_path):
    response = {
        "model": "fake-embed",
        "data": [{"index": 0, "embedding": [1.0, 0.0]}],
    }
    client = EmbeddingClient(
        None,
        "http://offline/v1",
        "fake-embed",
        "",
        journal_dir=tmp_path,
        batch_size=2,
        concurrency=1,
        expected_dimension=2,
    )
    client.journal = FakeJournal(response)
    with pytest.raises(ValueError, match="count mismatch"):
        asyncio.run(client.embed(["first", "second"]))
