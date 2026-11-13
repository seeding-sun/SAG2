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
from sag2 import CRITERION, EVIDENCE_SCORE_NOTE, GRADE_CRITERION, GRADE_RUBRIC, compact_json


class QwenProtocolError(ValueError):
    pass


class QwenModelMismatch(QwenProtocolError):
    """The gateway answered with a different model than the run is configured for.

    Raised in strict mode so a silent model substitution can never be reported as a
    valid run.  ``qwen3.5-122b-a10b`` being remapped to ``qwen3.8-27b`` is exactly
    this failure: HTTP 200, plausible latencies, and a completely different grader.
    """


def chat_completions_endpoint(base_url: str) -> str:
    endpoint = base_url.rstrip("/")
    return endpoint if endpoint.endswith("/chat/completions") else endpoint + "/chat/completions"


def _is_one_char_deletion(shorter: str, longer: str) -> bool:
    """Return True when `shorter` equals `longer` with exactly one character removed."""
    if len(longer) != len(shorter) + 1:
        return False
    return any(longer[:i] + longer[i + 1 :] == shorter for i in range(len(longer)))


def _is_one_char_substitution(left: str, right: str) -> bool:
    """Return True when two equal-length digit runs differ in exactly one position."""
    if len(left) != len(right) or left == right:
        return False
    return sum(a != b for a, b in zip(left, right, strict=True)) == 1


def _digit_distance_1(emitted: str, wanted: str) -> bool:
    """True when `emitted` looks like `wanted` with one digit dropped or mistyped."""
    return _is_one_char_deletion(emitted, wanted) or _is_one_char_substitution(
        emitted, wanted
    )


def repair_deleted_digit_key(
    scores: dict, expected_keys: set[str]
) -> tuple[dict, list[tuple[str, str]]]:
    """Repair one unambiguous candidate-key typo.

    Qwen occasionally corrupts a long candidate id, most often by dropping a single
    digit (``candidate_1017`` for ``candidate_10197``) or by mistyping the last digit
    (``candidate_11483`` for ``candidate_11486``). The mapping is only accepted when it
    is the sole possible reading:

    * exactly one expected key is missing and exactly one unexpected key is present;
    * the unexpected key is one digit-drop or one digit-substitution away from the
      missing key;
    * no *other* expected key is equally close to the unexpected key, so the intended
      target cannot be ambiguous.

    Anything else is left untouched so the caller still raises.
    """
    missing = sorted(expected_keys - set(scores))
    extra = sorted(set(scores) - expected_keys)
    if len(missing) != 1 or len(extra) != 1:
        return scores, []
    wanted, emitted = missing[0], extra[0]
    prefix = "candidate_"
    if not (wanted.startswith(prefix) and emitted.startswith(prefix)):
        return scores, []
    emitted_digits, wanted_digits = emitted[len(prefix) :], wanted[len(prefix) :]
    if not _digit_distance_1(emitted_digits, wanted_digits):
        return scores, []
    # Guard against a typo that sits one edit away from several real candidates: in
    # that case the model's intended id cannot be recovered, so refuse to guess.
    rivals = [
        key
        for key in expected_keys
        if key != wanted
        and key.startswith(prefix)
        and _digit_distance_1(emitted_digits, key[len(prefix) :])
    ]
    if rivals:
        return scores, []
    repaired = {(wanted if key == emitted else key): value for key, value in scores.items()}
    return repaired, [(emitted, wanted)]


def _unwrap_keyed_object(
    content: str,
    expected_keys: set[str],
    field: str,
    repairs: list[tuple[str, str]] | None = None,
    missing: list[str] | None = None,
    fill_missing: float | None = None,
    dropped: list[str] | None = None,
) -> dict:
    """Decode the model's JSON envelope and align its keys with the candidate ids.

    Qwen occasionally corrupts the key set: it may omit a candidate outright, or emit
    an id that belongs to no candidate in the batch. `fill_missing` decides what
    happens then. `None` keeps the strict historical behaviour and raises, while a
    numeric value scores every omitted candidate at that value and discards invented
    keys. Filling and dropping are both recorded so the run states exactly what it
    imputed and what it ignored.
    """
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
    scores = payload.get(field) if isinstance(payload, dict) else None
    if not isinstance(scores, dict):
        raise QwenProtocolError(f"Qwen response must contain a {field} object")
    if set(scores) != expected_keys:
        scores, applied = repair_deleted_digit_key(scores, expected_keys)
        if applied and repairs is not None:
            repairs.extend(applied)
        omitted = sorted(expected_keys - set(scores))
        unexpected = sorted(set(scores) - expected_keys)
        # A pure omission is imputable.  An unexpected key can never be trusted as a
        # grade, but aborting a long run over a mistyped id costs an entire arm, so in
        # imputation mode the invented key is dropped and the omitted candidates are
        # imputed.  Both decisions are recorded, so the run stays auditable.
        if fill_missing is not None and omitted:
            # A single stray id is a typo; a large share of stray ids means the model
            # answered a different batch entirely.  Refuse that instead of quietly
            # imputing most of the round at the floor value.
            if len(unexpected) > max(1, len(expected_keys) // 5):
                raise QwenProtocolError(
                    f"Qwen returned {len(unexpected)} keys outside this batch "
                    f"({unexpected[:5]}); refusing to impute wholesale"
                )
            if missing is not None:
                missing.extend(omitted)
            if dropped is not None:
                dropped.extend(unexpected)
            scores = {k: v for k, v in scores.items() if k in expected_keys}
            scores.update({key: fill_missing for key in omitted})
        if set(scores) != expected_keys:
            raise QwenProtocolError(
                f"Qwen score keys do not match candidates; missing={omitted[:5]}, "
                f"extra={unexpected[:5]}"
            )
    return scores


def parse_scores(
    content: str,
    expected_keys: set[str],
    repairs: list[tuple[str, str]] | None = None,
    missing: list[str] | None = None,
    fill_missing: float | None = None,
    dropped: list[str] | None = None,
) -> dict[str, float]:
    scores = _unwrap_keyed_object(
        content, expected_keys, "scores", repairs, missing, fill_missing, dropped
    )
    normalized = {}
    for key, value in scores.items():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise QwenProtocolError(f"Invalid evidence score for {key}: {value!r}")
        normalized[key] = float(value)
    return normalized


def parse_grades(
    content: str,
    expected_keys: set[str],
    repairs: list[tuple[str, str]] | None = None,
    missing: list[str] | None = None,
    fill_missing: int | None = None,
    dropped: list[str] | None = None,
) -> dict[str, int]:
    """Decode grade-mode output, which must use one of the rubric's discrete anchors."""
    allowed = set(GRADE_RUBRIC)
    grades = _unwrap_keyed_object(
        content, expected_keys, "grades", repairs, missing, fill_missing, dropped
    )
    normalized = {}
    for key, value in grades.items():
        if type(value) is not int or value not in allowed:
            raise QwenProtocolError(
                f"Invalid evidence grade for {key}: {value!r}; expected one of {sorted(allowed)}"
            )
        normalized[key] = value
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
        scoring_scale: str = "binary",
        fill_missing_score: float | None = None,
        max_output_tokens_per_candidate: int = 0,
        max_candidate_tokens: int = 32_000,
        max_request_tokens: int = 64_000,
        strict_model: bool = True,
    ):
        if batch_size < 1 or concurrency < 1:
            raise ValueError("Qwen batch size and concurrency must be positive")
        if scoring_scale not in ("binary", "grade"):
            raise ValueError("Qwen scoring scale must be 'binary' or 'grade'")
        if max_output_tokens_per_candidate < 0:
            raise ValueError("Qwen output token budget must not be negative")
        if fill_missing_score is not None:
            if scoring_scale == "grade":
                if type(fill_missing_score) is not int or fill_missing_score not in GRADE_RUBRIC:
                    raise ValueError(
                        "fill_missing_score must be a rubric anchor in grade mode, "
                        f"one of {sorted(GRADE_RUBRIC)}"
                    )
            elif not 0 <= fill_missing_score <= 1:
                raise ValueError("fill_missing_score must lie in [0, 1] in binary mode")
        self.session = session
        self.endpoint = chat_completions_endpoint(base_url)
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.response_format = response_format
        self.scoring_scale = scoring_scale
        self.fill_missing_score = fill_missing_score
        self.max_output_tokens_per_candidate = max_output_tokens_per_candidate
        self.strict_model = strict_model
        self.model_mismatches: list[dict[str, str]] = []
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
        graded = self.scoring_scale == "grade"
        if graded:
            system = (
                "You are a strict evidence judge. Return only one JSON object with a "
                "'grades' object. Each candidate id must appear exactly once and map to an "
                f"integer grade from {sorted(GRADE_RUBRIC)}. Do not add explanations or keys."
            )
            criterion = GRADE_CRITERION
        else:
            system = (
                "You are a strict evidence judge. Return only one JSON object with a "
                "'scores' object. Each candidate id must appear exactly once and map to a "
                "number from 0 to 1. Do not add explanations or keys."
            )
            criterion = CRITERION
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
                            "criterion": criterion,
                            "shared_context": state,
                            "candidates": candidates,
                        }
                    ),
                },
            ],
            "temperature": 0,
        }
        # Without an output cap a single degenerate generation can run to the server's
        # max-model-len (262144 here), which blocks a slot for hours and looks like a
        # wedged server. Each candidate needs one id plus one small integer, so a
        # generous per-candidate budget still leaves ample headroom.
        output_budget = None
        if self.max_output_tokens_per_candidate:
            output_budget = 256 + self.max_output_tokens_per_candidate * len(candidates)
            payload["max_tokens"] = output_budget
        if self.response_format:
            payload["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        response = await self.journal.post(
            "qwen_evidence", self.endpoint, self.api_key, payload
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        actual_model = response.get("model") or self.model
        # A gateway that cannot reach the requested model may silently substitute a
        # different one and still return HTTP 200.  That poisons the science without
        # any error surfacing, so a mismatch is treated as a hard protocol failure in
        # strict mode.  Non-strict mode records the mismatch and lets callers decide.
        if actual_model != self.model:
            self.model_mismatches.append(
                {"requested": self.model, "actual": actual_model}
            )
            if self.strict_model:
                raise QwenModelMismatch(
                    f"Qwen model identity changed: requested {self.model!r}, "
                    f"received {actual_model!r}. The gateway is routing this request "
                    "to a different deployment; results would not be comparable. "
                    "Start a run whose configured model is actually served, or pass a "
                    "model explicitly with --evidence-model."
                )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise QwenProtocolError("Qwen response has no chat completion content") from error
        repairs: list[tuple[str, str]] = []
        missing: list[str] = []
        dropped: list[str] = []
        scores = (
            parse_grades(
                content, expected, repairs, missing, self.fill_missing_score, dropped
            )
            if graded
            else parse_scores(
                content, expected, repairs, missing, self.fill_missing_score, dropped
            )
        )
        usage = token_usage(response)
        return scores, {
            "candidate_count": len(candidates),
            "attempts": attempt,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "model": actual_model,
            "latency_ms": elapsed_ms,
            "repaired_keys": [f"{emitted}->{wanted}" for emitted, wanted in repairs],
            "imputed_keys": missing,
            "dropped_keys": dropped,
            "max_tokens": output_budget,
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
                except QwenModelMismatch:
                    # Retrying cannot help: the gateway will keep routing to the same
                    # substitute deployment.  Fail immediately with the specific error.
                    raise
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
        graded = self.scoring_scale == "grade"
        return [
            {
                **candidate,
                "jev_score": (
                    scores[f"candidate_{candidate['id']}"] / 10
                    if graded
                    else scores[f"candidate_{candidate['id']}"]
                ),
                **(
                    {"jev_grade": scores[f"candidate_{candidate['id']}"]}
                    if graded
                    else {}
                ),
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
            "scoring_scale": self.scoring_scale,
            "fill_missing_score": self.fill_missing_score,
            "imputed_judgments": sum(len(detail["imputed_keys"]) for detail in details),
            "dropped_judgments": sum(len(detail["dropped_keys"]) for detail in details),
            "batches": details,
            "input_tokens": sum(detail["input_tokens"] for detail in details),
            "output_tokens": sum(detail["output_tokens"] for detail in details),
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }
