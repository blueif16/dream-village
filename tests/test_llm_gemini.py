"""Gemini path: an Anthropic-shaped Req for a gemini-* model is sent to google-genai generate_content with the
same prompt text, and the response comes back as the same Result (text, stop_reason, usage, cost, cache)."""

from __future__ import annotations

import copy
import threading

import pytest
from google.genai import errors, types

from tests.helpers import make_req, run_batch, run_sync

GEMINI = "gemini-4-argon"
SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["CORRECT", "WRONG"]}},
    "required": ["label"],
    "additionalProperties": False,
}


def response_payload(
    text: str = "answer",
    *,
    thought: str | None = "let me think",
    finish_reason: str = "STOP",
    prompt_tokens: int = 1000,
    cached_tokens: int = 400,
    output_tokens: int = 50,
    thought_tokens: int = 150,
) -> dict:
    parts = ([{"text": thought, "thought": True}] if thought else []) + [{"text": text}]
    return {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finish_reason": finish_reason, "index": 0}],
        "usage_metadata": {
            "prompt_token_count": prompt_tokens,
            "cached_content_token_count": cached_tokens,
            "candidates_token_count": output_tokens,
            "thoughts_token_count": thought_tokens,
            "total_token_count": prompt_tokens + output_tokens + thought_tokens,
        },
        "model_version": GEMINI,
        "response_id": "resp_fake_1",
    }


def gemini_error(code: int) -> errors.APIError:
    status = {400: "INVALID_ARGUMENT", 429: "RESOURCE_EXHAUSTED", 503: "UNAVAILABLE"}[code]
    body = {"error": {"code": code, "message": f"fake {code}", "status": status}}
    return (errors.ClientError if code < 500 else errors.ServerError)(code, body)


class FakeGemini:
    """The google-genai client surface memobench uses: `client.models.generate_content(model, contents, config)`.

    Replies are real `types.GenerateContentResponse` objects; `script` is a queue of payload dicts or exceptions
    consumed one per call (empty -> the default payload)."""

    def __init__(self):
        self.calls: list[dict] = []
        self.script: list = []
        self._lock = threading.Lock()
        self.models = self

    def generate_content(self, *, model, contents, config=None):
        with self._lock:
            self.calls.append({"model": model, "contents": copy.deepcopy(contents), "config": config})
            step = self.script.pop(0) if self.script else response_payload()
        if isinstance(step, BaseException):
            raise step
        return types.GenerateContentResponse.model_validate(step)


def sent_config(call: dict) -> types.GenerateContentConfig:
    cfg = call["config"]
    return cfg if isinstance(cfg, types.GenerateContentConfig) else types.GenerateContentConfig.model_validate(cfg)


@pytest.fixture
def gemini() -> FakeGemini:
    return FakeGemini()


@pytest.fixture
def gllm(make_llm, gemini):
    return make_llm(gemini_client=gemini, allowed_models=("claude-haiku-5-5", GEMINI))


def greq(prompt="hello", **params):
    return make_req(prompt, model=GEMINI, **params)


# ---------------------------------------------------------------- request translation


def test_prompt_blocks_are_sent_as_one_text_part_without_cache_control(gllm, gemini):
    blocks = [
        {"type": "text", "text": "Static head. ", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        {"type": "text", "text": "Memories here.\n"},
        {"type": "text", "text": "Question: when?"},
    ]
    gllm.call(greq(messages=[{"role": "user", "content": blocks}]))
    [call] = gemini.calls
    assert call["model"] == GEMINI
    assert call["contents"] == [{"role": "user", "parts": [{"text": "Static head. Memories here.\nQuestion: when?"}]}]


def test_assistant_turns_become_model_turns(gllm, gemini):
    msgs = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": [{"type": "text", "text": "A1"}]},
        {"role": "user", "content": "Q2"},
    ]
    gllm.call(greq(messages=msgs))
    assert [c["role"] for c in gemini.calls[0]["contents"]] == ["user", "model", "user"]
    assert gemini.calls[0]["contents"][1]["parts"] == [{"text": "A1"}]


def test_system_effort_schema_and_max_tokens_map_to_generation_config(gllm, gemini):
    system = [{"type": "text", "text": "You are a judge. ", "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "Be strict."}]
    gllm.call(greq(system=system, max_tokens=2048, output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}))
    cfg = sent_config(gemini.calls[0])
    assert cfg.system_instruction == "You are a judge. Be strict."
    assert cfg.max_output_tokens == 2048
    assert cfg.thinking_config.thinking_level == types.ThinkingLevel.LOW
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_json_schema == SCHEMA


def test_plain_text_request_asks_for_no_json(gllm, gemini):
    gllm.call(greq(output_config={"effort": "low"}))
    cfg = sent_config(gemini.calls[0])
    assert cfg.response_mime_type is None and cfg.response_json_schema is None
    assert cfg.system_instruction is None


def test_unsupported_param_is_rejected_before_any_call(gllm, gemini):
    with pytest.raises(ValueError, match="stop_sequences"):
        gllm.call(greq(stop_sequences=["\n"]))
    assert gemini.calls == []


# ---------------------------------------------------------------- response mapping


def test_result_text_is_the_answer_without_thoughts(gllm, gemini):
    res = gllm.call(greq())
    assert res.text == "answer"
    assert res.stop_reason == "end_turn"
    assert res.error is None and res.cached is False


def test_usage_splits_cached_prompt_tokens_and_counts_thoughts_as_output(gllm, gemini):
    res = gllm.call(greq())
    assert res.usage["input_tokens"] == 600
    assert res.usage["cache_read_input_tokens"] == 400
    assert res.usage["output_tokens"] == 200


def test_cost_uses_argon_rates(gllm, gemini):
    res = gllm.call(greq())
    # gemini-4-argon intro pricing, $/MTok: input 2.00, cached input 0.10, output (incl. thinking) 10.00
    assert res.cost_usd == pytest.approx((600 * 2.00 + 400 * 0.10 + 200 * 10.00) / 1e6, rel=1e-9)


@pytest.mark.parametrize(
    ("finish", "stop"),
    [("MAX_TOKENS", "max_tokens"), ("SAFETY", "refusal"), ("PROHIBITED_CONTENT", "refusal"), ("RECITATION", "refusal")],
)
def test_finish_reasons_map_to_anthropic_stop_reasons(gllm, gemini, finish, stop):
    gemini.script = [response_payload(finish_reason=finish)]
    assert gllm.call(greq()).stop_reason == stop


def test_blocked_prompt_with_no_candidates_is_an_empty_refusal(gllm, gemini):
    gemini.script = [{"prompt_feedback": {"block_reason": "SAFETY"}, "usage_metadata": {"prompt_token_count": 50}}]
    res = gllm.call(greq())
    assert (res.text, res.stop_reason, res.error) == ("", "refusal", None)


def test_response_is_cached(gllm, gemini):
    first = gllm.call(greq())
    second = gllm.call(greq())
    assert len(gemini.calls) == 1
    assert second.cached is True and second.text == first.text and second.stop_reason == "end_turn"


# ---------------------------------------------------------------- execution


def test_batch_mode_runs_real_time_without_submitting_a_batch(gllm, gemini, fake):
    reqs = [greq(f"q{i}") for i in range(5)]
    out = run_batch(gllm, reqs)
    assert len(gemini.calls) == 5
    assert fake.calls == []  # nothing reached the Anthropic client (no batches.create)
    assert all(out[r.key].text == "answer" and out[r.key].mode == "sync" for r in reqs)


def test_429_is_retried_then_succeeds(gllm, gemini, sleeps):
    gemini.script = [gemini_error(429), gemini_error(503), response_payload()]
    res = gllm.call(greq())
    assert res.text == "answer"
    assert len(gemini.calls) == 3 and len(sleeps) == 2


def test_400_in_run_becomes_a_result_error_not_a_crash(gllm, gemini):
    gemini.script = [gemini_error(400)]
    [res] = run_sync(gllm, [greq()]).values()
    assert res.error is not None and "400" in res.error
    assert len(gemini.calls) == 1  # not retried
