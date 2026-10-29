"""One fixed answering agent (port-plan Part 2b): every system answers through ANSWER_PROMPT_COMBINED,
rendered exactly as app/locomo/providers/memora/search.py renders it, with frozen params.

No network: memobench.runtime.get_llm is a RecordingLLM, memobench.embed is a fake, the memory store is a stub.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from collections import defaultdict

import pytest
from jinja2 import Template

from tests.apphelpers import (
    MODEL, RecordingLLM, install_llm, load_cfg, tiny_locomo, use_app, user_text, write_json,
)

FORBIDDEN = {"temperature", "top_p", "top_k", "seed"}

MEMORIES = json.dumps(
    ["\n=== Factual Details ===\n", '8 May 2023: Ann said "{{ not jinja }}" about her café {cat}'], indent=4
)
QUESTION = 'What did Ann say about "{x}" on 8 May?'


@pytest.fixture
def locomo(monkeypatch):
    return use_app(monkeypatch, "locomo")


def upstream_render(memories: str, question: str) -> str:
    """search.py:413-418 (combined-user mode), copied: Template(ANSWER_PROMPT_COMBINED).render(...)."""
    prompts = importlib.import_module("prompts")
    return Template(prompts.ANSWER_PROMPT_COMBINED).render(memories=memories, question=question)


# --------------------------------------------------------------------------- the answer request


@pytest.mark.parametrize("cache_memories", [False, True])
def test_answer_blocks_concatenate_to_upstream_rendered_prompt(locomo, cache_memories):
    answerer = importlib.import_module("answerer")
    req = answerer.build_answer_req(MEMORIES, QUESTION, cache_memories=cache_memories)
    assert user_text(req) == upstream_render(MEMORIES, QUESTION)


@pytest.mark.parametrize("cache_memories", [False, True])
def test_cache_control_on_prefix_and_on_memories_only_when_asked(locomo, cache_memories):
    answerer = importlib.import_module("answerer")
    prefix, memories, tail = answerer.build_answer_req(MEMORIES, QUESTION, cache_memories=cache_memories).params[
        "messages"][0]["content"]
    assert memories["text"] == MEMORIES
    assert prefix["cache_control"]["type"] == "ephemeral"
    assert ("cache_control" in memories) is cache_memories
    assert "cache_control" not in tail
    if cache_memories:
        # 1h on the conversation; the earlier prefix breakpoint must not be shorter-lived (API ordering rule)
        assert memories["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert prefix["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_answer_request_params_are_frozen(locomo):
    answerer = importlib.import_module("answerer")
    req = answerer.build_answer_req(MEMORIES, QUESTION, cache_memories=False)
    p = req.params
    assert p["model"] == MODEL
    assert p["output_config"] == {"effort": "low"}
    assert type(p["max_tokens"]) is int and p["max_tokens"] == 4096
    assert not FORBIDDEN & set(p)
    assert "system" not in p and len(p["messages"]) == 1
    assert req.meta["stage"] == "answer"


# --------------------------------------------------------------------------- Memora S / P / noabs (real-time)

MEMS = [{"index": "m1", "value": "Ann adopted a cat", "timestamp": "8 May 2023", "score": 0.9, "memory_type": "factual"}]
# what _format_speaker_memories yields for MEMS with config use_segments_as_episodic=true (no episodes linked)
MEMS_FORMATTED = ["\n=== Factual Details ===\n", "8 May 2023: Ann adopted a cat"]


def test_memora_answer_is_one_realtime_call_with_upstream_prompt(locomo, tmp_path, monkeypatch):
    llm = install_llm(monkeypatch, RecordingLLM(reply=lambda r: "reasoning...\nFINAL ANSWER: a cat"))
    search = importlib.import_module("providers.memora.search")
    s = search.MemoraSearch(load_cfg(locomo), str(tmp_path / "out.json"), 5, retrieval_strategy="semantic")
    monkeypatch.setattr(s, "search_memory", lambda *a, **k: (MEMS, 0.0))

    out = s.answer_question("Ann_Bob_0", None, "What pet did Ann adopt?")

    assert llm.runs == []
    (req,) = llm.calls
    assert user_text(req) == upstream_render(json.dumps(MEMS_FORMATTED, indent=4), "What pet did Ann adopt?")
    assert req.params == importlib.import_module("answerer").build_answer_req(
        json.dumps(MEMS_FORMATTED, indent=4), "What pet did Ann adopt?", cache_memories=False).params
    assert out[0] == "a cat"


def test_memora_run_skips_category_5_and_tags_each_question(locomo, tmp_path, monkeypatch):
    llm = install_llm(monkeypatch, RecordingLLM())
    search = importlib.import_module("providers.memora.search")
    out_path = tmp_path / "out.json"
    s = search.MemoraSearch(load_cfg(locomo), str(out_path), 5, retrieval_strategy="prompt")
    monkeypatch.setattr(s, "search_memory", lambda *a, **k: (MEMS, 0.0))

    s.process_data_file(str(write_json(tmp_path / "locomo.json", tiny_locomo())))

    tagged = {(r.meta["system"], r.meta["conv_id"], r.meta["question_id"], r.meta["stage"]) for r in llm.calls}
    assert tagged == {
        ("memora-p", "conv-a", "conv-a/q0", "answer"), ("memora-p", "conv-a", "conv-a/q2", "answer"),
        ("memora-p", "conv-b", "conv-b/q1", "answer"), ("memora-p", "conv-b", "conv-b/q2", "answer"),
    }
    assert len(llm.calls) == 4
    results = json.loads(out_path.read_text())
    assert sorted(str(r["category"]) for v in results.values() for r in v) == ["1", "2", "4", "4"]


# --------------------------------------------------------------------------- full context (batch)

FULL_CONTEXT = {
    "conv-a": "1:00 pm on 8 May, 2023 | Ann: I adopted a cat.\n1:00 pm on 8 May, 2023 | Bob: Nice!\n"
              "2:00 pm on 9 May, 2023 | Bob: I moved to Oslo.\n",
    "conv-b": "1:00 pm on 8 May, 2023 | Cy: I run marathons.\n1:00 pm on 8 May, 2023 | Di: Cool.\n"
              "2:00 pm on 9 May, 2023 | Di: I bought a red bike.\n",
}
ANSWERABLE = {
    "conv-a": {"What pet did Ann adopt?", "When did Bob move?"},
    "conv-b": {"What sport does Cy do?", "What did Di buy?"},
}


def test_full_context_one_batch_req_per_answerable_question_sharing_the_conversation(locomo, tmp_path, monkeypatch):
    llm = install_llm(monkeypatch, RecordingLLM())
    fc = importlib.import_module("providers.full_context")
    fc.FullContextManager(cfg=load_cfg(locomo), data_path=str(write_json(tmp_path / "d.json", tiny_locomo()))) \
        .process_all_conversations(str(tmp_path / "out.json"))

    assert llm.calls == []
    ((reqs, mode, _),) = llm.runs
    assert mode == "batch"
    groups = defaultdict(list)
    for r in reqs:
        groups[r.meta["prefix_group"]].append(r)
    assert set(groups) == set(FULL_CONTEXT)
    for conv_id, rs in groups.items():
        assert {r.params["messages"][0]["content"][1]["text"] for r in rs} == {FULL_CONTEXT[conv_id]}
        rendered = {user_text(r) for r in rs}
        assert rendered == {upstream_render(FULL_CONTEXT[conv_id], q) for q in ANSWERABLE[conv_id]}
        assert all(r.meta["system"] == "full-context" and r.meta["conv_id"] == conv_id for r in rs)
        assert all(r.params["messages"][0]["content"][1]["cache_control"]["ttl"] == "1h" for r in rs)


def test_full_context_results_file_has_parsed_answers_and_no_category_5(locomo, tmp_path, monkeypatch):
    install_llm(monkeypatch, RecordingLLM(reply=lambda r: "FINAL ANSWER: stub"))
    fc = importlib.import_module("providers.full_context")
    out = tmp_path / "out.json"
    fc.FullContextManager(cfg=load_cfg(locomo), data_path=str(write_json(tmp_path / "d.json", tiny_locomo()))) \
        .process_all_conversations(str(out))

    results = json.loads(out.read_text())
    assert {k: {r["question"] for r in v} for k, v in results.items()} == ANSWERABLE
    assert {r["response"] for v in results.values() for r in v} == {"stub"}


# --------------------------------------------------------------------------- RAG (batch, chunk 500, k=3)

MARKERS = ["alpha", "bravo", "charlie", "delta"]
QUERY_VEC = [0.9, 0.8, 0.7, 0.1, 0.0]  # alpha > bravo > charlie > delta; filler chunks score 0


def rag_data():
    filler = " ".join(["the weather was calm and the river was slow"] * 70)  # ~700 tokens: one marker per chunk
    conv = {"speaker_a": "Ann", "speaker_b": "Bob"}
    for i, m in enumerate(MARKERS, start=1):
        conv[f"session_{i}_date_time"] = f"{i}:00 pm on 8 May, 2023"
        conv[f"session_{i}"] = [{"speaker": "Ann", "dia_id": f"D{i}:1", "text": f"{filler} {m} {filler}"}]
    qa = [
        {"question": "Which marker came first?", "answer": "alpha", "category": 1},
        {"question": "Adversarial?", "adversarial_answer": "x", "category": 5},
        {"question": "Which marker came last?", "answer": "delta", "category": 2},
    ]
    return [{"sample_id": "conv-r", "conversation": conv, "qa": qa}]


def test_rag_answers_from_top3_chunks_embedded_as_document_and_query_in_one_batch(locomo, tmp_path, monkeypatch):
    embed_calls = []

    def fake_embed(texts, kind):
        texts = list(texts)
        embed_calls.append((kind, texts))
        if kind == "query":
            return [list(QUERY_VEC) for _ in texts]
        return [[float(m in t) for m in MARKERS] + [0.01] for t in texts]

    fake = types.ModuleType("memobench.embed")
    fake.embed = fake_embed
    monkeypatch.setitem(sys.modules, "memobench.embed", fake)
    monkeypatch.setattr(sys.modules["memobench"], "embed", fake, raising=False)
    llm = install_llm(monkeypatch, RecordingLLM())
    rag = importlib.import_module("providers.rag")
    cfg = load_cfg(locomo)

    rag.RAGManager(cfg=cfg, data_path=str(write_json(tmp_path / "d.json", rag_data())),
                   chunk_size=cfg.memory.chunk_size, k=cfg.memory.num_chunks) \
        .process_all_conversations(str(tmp_path / "out.json"))

    kinds = defaultdict(list)
    for kind, texts in embed_calls:
        kinds[kind] += texts
    assert set(kinds) == {"document", "query"}
    assert sorted(kinds["query"]) == ["Which marker came first?", "Which marker came last?"]
    assert len(kinds["document"]) >= 5  # ~2,900 tokens at 500 per chunk
    assert llm.calls == []
    ((reqs, mode, _),) = llm.runs
    assert mode == "batch" and len(reqs) == 2
    for r in reqs:
        parts = r.params["messages"][0]["content"][1]["text"].split("\n<->\n")
        assert len(parts) == 3
        assert [m for p in parts for m in MARKERS if m in p] == ["alpha", "bravo", "charlie"]
        assert all(p in kinds["document"] for p in parts)
        assert r.meta["system"] == "rag" and r.meta["stage"] == "answer"


# --------------------------------------------------------------------------- LongMemEval Memora (real-time)


def test_lme_memora_answer_is_realtime_call_with_default_template(tmp_path, monkeypatch):
    app = use_app(monkeypatch, "longmemeval")
    llm = install_llm(monkeypatch, RecordingLLM(reply=lambda r: "Step 1...\nFINAL ANSWER: a red bike"))
    search = importlib.import_module("providers.memora.search")
    prompts = importlib.import_module("prompts")
    s = search.MemoraSearch(load_cfg(app), str(tmp_path / "out.json"), 5, retrieval_strategy="semantic")
    mems = [{"index": "m1", "value": "User owns a red bike", "timestamp": "2023/05/09", "score": 0.9,
             "memory_type": "factual"}]
    monkeypatch.setattr(s, "search_memory", lambda *a, **k: (mems, 0.0))
    item = {"question_id": "q1_abs", "question": "What bike do I own?", "answer": "red bike",
            "question_type": "single-session-user", "question_date": "2023/05/10 (Wed) 10:00"}

    out = s.process_question(item, 0)

    assert llm.runs == []
    (req,) = llm.calls
    assert user_text(req) == Template(prompts.ANSWER_PROMPT_LONGMEMEVAL).render(
        memories="\n".join(["\n=== Factual Details ===\n", "2023/05/09: User owns a red bike"]),
        question="What bike do I own?", question_date="2023/05/10 (Wed) 10:00")
    assert req.params["model"] == MODEL and req.params["output_config"] == {"effort": "low"}
    assert type(req.params["max_tokens"]) is int and not FORBIDDEN & set(req.params)
    assert (req.meta["system"], req.meta["question_id"], req.meta["stage"]) == ("memora-s", "q1_abs", "answer")
    assert out["generated_answer"] == "a red bike"
