"""Qwen Chat adapter implementing the SAG2 evidence-scorer interface."""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from pathlib import Path

import aiohttp
import tiktoken

from model_api import RequestJournal, token_usage
from sag2 import CRITERION, EVIDENCE_SCORE_NOTE, compact_json


class QwenProtocolError(ValueError):
    pass


def chat_completions_endpoint(base_url: str) -> str:
    endpoint = base_url.rstrip("/")
    return endpoint if endpoint.endswith("/chat/completions") else endpoint + "/chat/completions"


def parse_scores(content: str, expected_keys: set[str]) -> dict[str, float]:
    if not isinstance(content, str) or not content.strip():
        raise QwenProtocolError("Qwen returned empty content")
    text = content.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise QwenProtocolError(f"Qwen returned invalid JSON: {error}") from error
    scores = payload.get("scores") if isinstance(payload, dict) else None
    if not isinstance(scores, dict):
        raise QwenProtocolError("Qwen response must contain a scores object")
    if set(scores) != expected_keys:
        missing = sorted(expected_keys - set(scores))
        extra = sorted(set(scores) - expected_keys)
        raise QwenProtocolError(
            f"Qwen score keys do not match candidates; missing={missing[:5]}, extra={extra[:5]}"
        )
    normalized = {}
    for key, value in scores.items():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise QwenProtocolError(f"Invalid evidence score for {key}: {value!r}")
        normalized[key] = float(value)
    return normalized


class QwenEvidenceScorer:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        model: str,
        api_key: str,
        *,
        journal_dir: Path,
        batch_size: int = 10,
        concurrency: int = 4,
        response_format: bool = True,
        max_candidate_tokens: int = 32_000,
        max_request_tokens: int = 64_000,
    ):
        if batch_size < 1 or concurrency < 1:
            raise ValueError("Qwen batch size and concurrency must be positive")
        self.session = session
        self.endpoint = chat_completions_endpoint(base_url)
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.response_format = response_format
        self.max_candidate_tokens = max_candidate_tokens
        self.max_request_tokens = max_request_tokens
        self.encoding = tiktoken.get_encoding("o200k_base")
        self.journal = RequestJournal(session, journal_dir)

    @staticmethod
    def _build_state(
        query: str,
        evidence: list[dict],
        context: dict | None,
    ) -> dict:
        state = {
            "original_question": context["original_question"] if context else query,
            "previously_retrieved_evidence": [
                {
                    "passage_id": candidate["id"],
                    "title": candidate["title"],
                    "passage": candidate["content"],
                    **(
                        {"heading": candidate["heading"]}
                        if candidate.get("heading")
                        else {}
                    ),
                    **(
                        {"previous_jev_score": candidate["previous_jev_score"]}
                        if "previous_jev_score" in candidate
                        else {}
                    ),
                }
                for candidate in evidence
            ],
        }
        if context and context.get("search_history"):
            combined = {c["id"]: c for c in [*context["evidence"], *evidence]}
            state.update(
                {
                    "current_search_query": query,
                    "retrieval_history": context["search_history"],
                    "previous_decisions": context["previous_decisions"],
                    "history_note": (
                        "Earlier decisions are tentative search reasoning. Ground facts in "
                        "the source passages; evaluate each candidate against the complete "
                        "original question."
                    ),
                    "previously_retrieved_evidence": [
                        {
                            "passage_id": candidate["id"],
                            "title": candidate["title"],
                            "passage": candidate["content"],
                            **(
                                {"heading": candidate["heading"]}
                                if candidate.get("heading")
                                else {}
                            ),
                        }
                        for candidate in combined.values()
                    ],
                }
            )
        if any("previous_jev_score" in candidate for candidate in evidence):
            state["evidence_score_note"] = EVIDENCE_SCORE_NOTE
        return state

    def _pack_candidates(
        self,
        state: dict,
        candidates: list[dict],
    ) -> list[dict[str, dict]]:
        state_tokens = len(self.encoding.encode(compact_json(state), disallowed_special=())) + 1024
        batches: list[dict[str, dict]] = []
        current: dict[str, dict] = {}
        current_tokens = state_tokens
        for candidate in candidates:
            key = f"candidate_{candidate['id']}"
            item = {
                "title": candidate["title"],
                "passage": candidate["content"],
                **({"heading": candidate["heading"]} if candidate.get("heading") else {}),
            }
            tokens = len(self.encoding.encode(compact_json(item), disallowed_special=())) + 32
            if tokens > self.max_candidate_tokens:
                raise ValueError(
                    f"Candidate {candidate['id']} exceeds the per-candidate Qwen budget"
                )
            if state_tokens + tokens > self.max_request_tokens:
                raise ValueError(
                    f"Candidate {candidate['id']} exceeds the Qwen context budget; "
                    "source text was not truncated"
                )
            if current and (
                len(current) >= self.batch_size
                or current_tokens + tokens > self.max_request_tokens
            ):
                batches.append(current)
                current, current_tokens = {}, state_tokens
            current[key] = item
            current_tokens += tokens
        if current:
            batches.append(current)
        return batches

    async def _request_batch(
        self,
        state: dict,
        candidates: dict[str, dict],
        attempt: int,
    ) -> tuple[dict[str, float], dict, int]:
        expected = set(candidates)
        system = (
            "You are a strict evidence judge. Return only one JSON object with a "
            "'scores' object. Each candidate id must appear exactly once and map to a "
            "number from 0 to 1. Do not add explanations or keys."
        )
        if attempt > 1:
            system += " The previous response was invalid; correct the JSON format."
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": compact_json(
                        {
                            "criterion": CRITERION,
                            "shared_context": state,
                            "candidates": candidates,
                        }
                    ),
                },
            ],
            "temperature": 0,
        }
        if self.response_format:
            payload["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        response = await self.journal.post(
            "qwen_evidence", self.endpoint, self.api_key, payload
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise QwenProtocolError("Qwen response has no chat completion content") from error
        scores = parse_scores(content, expected)
        usage = token_usage(response)
        return scores, {
            "candidate_count": len(candidates),
            "attempts": attempt,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "model": response.get("model") or self.model,
            "latency_ms": elapsed_ms,
        }, attempt

    async def _score_batch(
        self,
        state: dict,
        candidates: dict[str, dict],
        semaphore: asyncio.Semaphore,
    ) -> tuple[dict[str, float], dict]:
        async with semaphore:
            last_error = None
            for attempt in (1, 2):
                try:
                    scores, detail, _ = await self._request_batch(
                        state, candidates, attempt
                    )
                    return scores, detail
                except QwenProtocolError as error:
                    last_error = error
            raise QwenProtocolError(f"Qwen score protocol failed after retry: {last_error}")

    async def score(
        self,
        query: str,
        candidates: list[dict],
        evidence: list[dict],
        *,
        context: dict | None = None,
    ) -> tuple[list[dict], dict]:
        started = time.perf_counter()
        state = self._build_state(query, evidence, context)
        batches = self._pack_candidates(state, candidates)
        semaphore = asyncio.Semaphore(self.concurrency)
        results = await asyncio.gather(
            *(self._score_batch(state, batch, semaphore) for batch in batches)
        )
        scores = {
            key: value
            for batch_scores, _ in results
            for key, value in batch_scores.items()
        }
        details = [detail for _, detail in results]
        return [
            {
                **candidate,
                "jev_score": scores[f"candidate_{candidate['id']}"],
                "jev_model": next(
                    detail["model"]
                    for batch, (_, detail) in zip(batches, results, strict=True)
                    if f"candidate_{candidate['id']}" in batch
                ),
            }
            for candidate in candidates
        ], {
            "requests": len(details),
            "judgments": len(candidates),
            "batches": details,
            "input_tokens": sum(detail["input_tokens"] for detail in details),
            "output_tokens": sum(detail["output_tokens"] for detail in details),
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }
