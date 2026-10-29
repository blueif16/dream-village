"""Memora's ChatCompletionModel on the Anthropic path (port-plan Part 1a).

`memobench.runtime.get_llm` is replaced by a recording fake LLM that returns spec-shaped `Result`s
(docs/llm-layer-spec.md), so no network call is made; every assertion is about the `Req` the
adapter built or the value `invoke` returned.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import anthropic
import pytest
from omegaconf import OmegaConf
from pydantic import BaseModel

import memobench.runtime
from memobench.context import clear_meta, set_meta
from memobench.llm import Req, Result

REPO = Path(__file__).resolve().parent.parent
MEMORA_SRC = REPO / "vendor" / "Memora" / "src"
if str(MEMORA_SRC) not in sys.path:
    sys.path.insert(0, str(MEMORA_SRC))

MODEL = "claude-haiku-5-5"


# --------------------------------------------------------------------------- fakes


def spec_result(text: str, *, stop_reason: str = "end_turn", error: str | None = None) -> Result:
    """A complete Result as memobench.llm returns it for a real-time call."""
    usage = {"input_tokens": 1000, "output_tokens": 200, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    message = {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }
    return Result(
        key="0" * 64,
        text=text,
        message=message,
        stop_reason=stop_reason,
        usage=usage,
        cost_usd=0.0002,
        latency_ms=12.0,
        cached=False,
        mode="sync",
        batch_id=None,
        error=error,
    )


class RecordingLLM:
    """Stands in for memobench.llm.LLM: records every Req passed to call(), replies via `reply(req)`."""

    def __init__(self, reply=None):
        self.reqs: list[Req] = []
        self._lock = threading.Lock()
        self.reply = reply or (lambda req: spec_result("ok"))

    def call(self, req: Req) -> Result:
        with self._lock:
            self.reqs.append(req)
        return self.reply(req)


@pytest.fixture
def fake_llm(monkeypatch, tmp_path):
    fake = RecordingLLM()
    monkeypatch.setattr(memobench.runtime, "get_llm", lambda: fake)
    monkeypatch.setenv("MEMOBENCH_ROOT", str(tmp_path))
    clear_meta()
    yield fake
    clear_meta()


@pytest.fixture
def model():
    from memora.utils.llm import ChatCompletionModel

    cfg = OmegaConf.create({"llm": {"model": MODEL, "seed": 42}, "openai": {"api_type": "azure"}})
    return ChatCompletionModel(cfg)


def user_text(params: dict) -> str:
    """Concatenated text of the single user message (string content or text blocks)."""
    (msg,) = params["messages"]
    assert msg["role"] == "user"
    content = msg["content"]
    if isinstance(content, str):
        return content
    assert all(b["type"] == "text" for b in content)
    return "".join(b["text"] for b in content)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------- imports


HEAVY = ("torch", "transformers", "peft", "mem0", "bert_score", "sentence_transformers")


def test_memora_core_modules_import_without_heavy_deps():
    """The anthropic path must not need torch/transformers/mem0 (they are not installed by design).
    A meta-path blocker makes this hold even on a machine where they happen to be installed."""
    code = f"""
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {HEAVY!r}:
            raise ImportError('blocked heavy dep: ' + name)
sys.meta_path.insert(0, Block())
sys.path.insert(0, {str(MEMORA_SRC)!r})
import memora, memora.utils.llm, memora.memora_client
import memora.builder.chat_memory_builder, memora.builder.memory_builder
import memora.retriever.prompted_policy_retriever, memora.retriever.semantic_retriever
print('ok')
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr[-2000:]


# --------------------------------------------------------------------------- (i) request shape


def test_claude_model_name_selects_anthropic_path_without_local_model(fake_llm, model):
    assert model.model_type == "anthropic"
    assert model.hf_model is None


def test_params_have_frozen_shape_and_no_sampling_params_even_when_caller_passes_them(fake_llm, model):
    model.invoke("Hello {name}", {"name": "Alice"}, temperature=0.0, top_p=0.9, top_k=5, seed=7, n=1)

    (req,) = fake_llm.reqs
    p = req.params
    assert p["model"] == MODEL
    assert p["output_config"]["effort"] == "low"
    assert isinstance(p["max_tokens"], int) and p["max_tokens"] > 0
    for banned in ("temperature", "top_p", "top_k", "seed", "n"):
        assert banned not in p
        assert banned not in p["output_config"]


# --------------------------------------------------------------------------- (ii) prompt bytes


def test_str_prompt_with_args_is_one_user_message_identical_to_format(fake_llm, model):
    template = "Timestamp: {timestamp}\nJSON: {{'a': 1}}\nInput:\n{content}\nOutput:\n"
    args = {"timestamp": "8 May, 2023", "content": "Caroline: I went to a LGBTQ support group {yesterday}."}

    out = model.invoke(template, args)

    (req,) = fake_llm.reqs
    assert user_text(req.params) == template.format(**args)
    assert "system" not in req.params
    assert out == "ok"


def test_list_input_moves_leading_system_messages_to_system_param(fake_llm, model):
    model.invoke([{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Hi"}])

    (req,) = fake_llm.reqs
    assert req.params["system"] == "Be terse."
    assert req.params["messages"] == [{"role": "user", "content": "Hi"}]


# --------------------------------------------------------------------------- (iii) policy cache split


def _policy_prompt() -> str:
    from memora.retriever.prompted_policy_retriever import LLM_POLICY_PROMPT

    return LLM_POLICY_PROMPT.format(
        user_question="When did Caroline go to the LGBTQ support group?",
        current_query="Caroline LGBTQ support group date",
        step=1,
        max_steps=5,
        W_summary="[1] Caroline attended an LGBTQ support group on 7 May 2023.",
        F_summary="(empty)",
        trace="No history yet",
    )


def test_policy_prompt_is_split_with_cache_control_on_static_prefix(fake_llm, model):
    from memora.retriever.prompted_policy_retriever import LLM_POLICY_PROMPT

    prompt = _policy_prompt()
    # exactly how prompted_policy_retriever.py calls it
    model.invoke(input=prompt, temperature=0.0, source="PromptedPolicyRetriever")

    (req,) = fake_llm.reqs
    blocks = req.params["messages"][0]["content"]
    expected_prefix = LLM_POLICY_PROMPT.split("{user_question}")[0].replace("{{", "{").replace("}}", "}")
    assert len(expected_prefix) > 2000  # worth caching (>= ~512 Haiku tokens)
    assert isinstance(blocks, list) and len(blocks) == 2
    assert blocks[0]["text"] == expected_prefix
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in blocks[1]
    assert user_text(req.params) == prompt


def test_prompts_matching_no_registered_template_carry_no_cache_control(fake_llm, model):
    model.invoke("Summarize this: {x}", prompt_args={"x": "a" * 5000}, source="SomethingElse")

    (req,) = fake_llm.reqs
    assert "cache_control" not in json.dumps(req.params)


def _assert_static_head_cached(req, template, prompt_args):
    blocks = req.params["messages"][0]["content"]
    head = template.split("{", 1)[0]
    assert isinstance(blocks, list) and len(blocks) == 2
    assert blocks[0]["text"] == head and blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in blocks[1]
    assert user_text(req.params) == template.format(**prompt_args)


def test_extraction_prompt_static_head_is_cached(fake_llm, model):
    from memora.builder.memory_builder import MemoryOutputs
    from memora.builder.chat_memory_builder import PROMPT_BUILD_MEMORY

    args = {"content": "user: I moved to Boston last week.", "timestamp": "21 October, 2026"}
    # exactly how chat_memory_builder.py:103 calls it
    model.invoke(input=PROMPT_BUILD_MEMORY, prompt_args=args, response_format=MemoryOutputs)

    _assert_static_head_cached(fake_llm.reqs[0], PROMPT_BUILD_MEMORY, args)


def test_cue_prompt_static_head_is_cached(fake_llm, model):
    from memora.core import cue_index_generator as cig

    gen = cig.CueIndexGenerator(model.cfg, model_client=model)
    gen.generate_cue_indices_batch([{"index": "User's move to Boston", "value": "The user moved to Boston in October 2026."}])

    req = fake_llm.reqs[0]
    blocks = req.params["messages"][0]["content"]
    assert isinstance(blocks, list) and blocks[0].get("cache_control") == {"type": "ephemeral"}
    assert len(blocks[0]["text"]) > 2000
    assert "".join(b["text"] for b in blocks) == user_text(req.params)


# --------------------------------------------------------------------------- (iv) structured outputs


def test_response_format_sends_json_schema_and_returns_parsed_object(fake_llm, model):
    from memora.utils.memory import MemoryUpdateDecision

    payload = {"should_update": True, "best_candidate_index": 0, "updated_value": "v2", "updated_index": "i2", "updated_cues": ["c"]}
    fake_llm.reply = lambda req: spec_result(json.dumps(payload))

    out = model.invoke("New: {new_index}", {"new_index": "x"}, response_format=MemoryUpdateDecision)

    (req,) = fake_llm.reqs
    assert req.params["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": anthropic.transform_schema(MemoryUpdateDecision)},
    }
    assert isinstance(out, MemoryUpdateDecision)
    assert out == MemoryUpdateDecision(**payload)


def test_no_response_format_sends_no_format_and_returns_text(fake_llm, model):
    fake_llm.reply = lambda req: spec_result('{"action": "STOP"}')

    out = model.invoke("plain")

    (req,) = fake_llm.reqs
    assert "format" not in req.params["output_config"]
    assert out == '{"action": "STOP"}'


# --------------------------------------------------------------------------- (v) parse failures


def test_invalid_json_is_logged_and_returns_upstream_empty_fallback(fake_llm, model, tmp_path):
    from memora.builder.memory_builder import MemoryOutputs

    fake_llm.reply = lambda req: spec_result('{"entries": [ {"index": ')

    out = model.invoke("Extract: {content}", {"content": "c"}, response_format=MemoryOutputs)

    assert out == MemoryOutputs(entries=[])  # what upstream returns on a content-filter error
    (line,) = read_jsonl(tmp_path / "logs" / "parse_failures.jsonl")
    assert line["stage"] == "extraction"
    assert line["raw_text"] == '{"entries": [ {"index": '
    assert line["error"]
    assert "source" in line


def test_invalid_json_without_empty_constructor_is_logged_then_raised(fake_llm, model, tmp_path):
    from memora.utils.memory import MemoryUpdateDecision

    fake_llm.reply = lambda req: spec_result("not json")

    with pytest.raises(Exception):  # upstream's last resort re-raises; the update call site catches it
        model.invoke("x", response_format=MemoryUpdateDecision)

    (line,) = read_jsonl(tmp_path / "logs" / "parse_failures.jsonl")
    assert line["stage"] == "update" and line["raw_text"] == "not json"


def test_max_tokens_stop_is_logged_and_falls_back_even_if_text_parses(fake_llm, model, tmp_path):
    from memora.builder.memory_builder import MemoryOutputs

    fake_llm.reply = lambda req: spec_result('{"entries": []}', stop_reason="max_tokens")

    out = model.invoke("x", response_format=MemoryOutputs)

    assert out == MemoryOutputs(entries=[])
    (line,) = read_jsonl(tmp_path / "logs" / "parse_failures.jsonl")
    assert "max_tokens" in line["error"]


def test_result_error_raises(fake_llm, model):
    fake_llm.reply = lambda req: spec_result("", error="invalid_request_error: bad")

    with pytest.raises(Exception, match="invalid_request_error"):
        model.invoke("x")


# --------------------------------------------------------------------------- (vi) meta


class SegmentationOutput(BaseModel):  # shape-only stand-in; the app's class is matched by source
    episodes: list[str]


def _stage_cases():
    from memora.builder.chat_memory_builder import EpisodicMemoryOutput
    from memora.builder.memory_builder import MemoryOutputs
    from memora.core.cue_index_generator import BatchCueIndices
    from memora.utils.memory import MemoryUpdateDecision

    return [
        ("segmentation", dict(response_format=SegmentationOutput, source="ConversationSegmenter", temperature=0.0), '{"episodes": []}'),
        ("extraction", dict(response_format=MemoryOutputs), '{"entries": []}'),
        ("episodic", dict(response_format=EpisodicMemoryOutput), '{"episodic_index": "i", "episodic_value": "v"}'),
        ("cue", dict(response_format=BatchCueIndices), '{"results": []}'),
        ("update", dict(response_format=MemoryUpdateDecision), '{"should_update": false}'),
        ("policy", dict(source="PromptedPolicyRetriever", temperature=0.0), '{"action": "STOP"}'),
    ]


def test_each_call_site_gets_its_stage(fake_llm, model):
    for stage, kwargs, reply in _stage_cases():
        fake_llm.reply = lambda req, reply=reply: spec_result(reply)
        model.invoke("prompt", **kwargs)
        assert fake_llm.reqs[-1].meta["stage"] == stage, kwargs


def test_meta_carries_stage_and_the_calling_threads_meta(fake_llm, model):
    from memora.utils.memory import MemoryUpdateDecision

    fake_llm.reply = lambda req: spec_result('{"should_update": false}')

    def worker(qid: str):
        set_meta(system="memora-p", conv_id="conv-26", question_id=qid)
        model.invoke("x", response_format=MemoryUpdateDecision)

    for qid in ("q-1", "q-2"):
        t = threading.Thread(target=worker, args=(qid,))
        t.start()
        t.join()

    metas = {r.meta["question_id"]: r.meta for r in fake_llm.reqs}
    for qid in ("q-1", "q-2"):
        m = metas[qid]
        assert m["system"] == "memora-p" and m["conv_id"] == "conv-26" and m["stage"] == "update"
        assert "source" in m
