"""LongMemEval baselines (full context, RAG) answer through the same fixed answerer as Memora: the repo-default
`longmemeval` template (prompts.ANSWER_PROMPT_LONGMEMEVAL), rendered exactly as providers/memora/search.py renders it.

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

from tests.apphelpers import MODEL, RecordingLLM, install_llm, load_cfg, use_app, user_text, write_json

FORBIDDEN = {"temperature", "top_p", "top_k", "seed"}

# Memories as Memora formats them ("\n".join(formatted_memories)), with text a template engine must not touch.
MEMORIES = "\n".join(["=== Episodes ===\n", 'Conversation on 9 May, 2023:\nuser: I said "{{ question }}" {x}\n',
                      "\n=== Factual Details ===\n", "2023/05/09: User owns a red bike"])
QUESTION = 'What did I say about "{x}"?'
QUESTION_DATE = "2023/05/10 (Wed) 10:00"


@pytest.fixture
def lme(monkeypatch):
    return use_app(monkeypatch, "longmemeval")


def memora_render(memories: str, question: str, question_date: str) -> str:
    """providers/memora/search.py (process_question), copied: Template(ANSWER_PROMPT).render(...) with the
    repo-default template."""
    prompts = importlib.import_module("prompts")
    return Template(prompts.ANSWER_PROMPT_LONGMEMEVAL).render(
        memories=memories, question=question, question_date=question_date)


def blocks(req):
    (msg,) = req.params["messages"]
    return msg["content"]


# --------------------------------------------------------------------------- the shared answer request


@pytest.mark.parametrize("memories", [MEMORIES, ""])
@pytest.mark.parametrize("cache_memories,batch", [(False, False), (False, True), (True, True)])
def test_shared_answer_req_text_is_byte_identical_to_memora_render(lme, memories, cache_memories, batch):
    answerer = importlib.import_module("answerer")
    req = answerer.build_answer_req(memories, QUESTION, QUESTION_DATE, cache_memories=cache_memories, batch=batch)
    assert user_text(req) == memora_render(memories, QUESTION, QUESTION_DATE)


@pytest.mark.parametrize("cache_memories,batch,ttl", [(False, False, None), (False, True, "1h"), (True, True, "1h")])
def test_shared_answer_req_caches_prefix_and_history_block_only_when_asked(lme, cache_memories, batch, ttl):
    answerer = importlib.import_module("answerer")
    prefix, memories, tail = blocks(
        answerer.build_answer_req(MEMORIES, QUESTION, QUESTION_DATE, cache_memories=cache_memories, batch=batch))
    want = {"type": "ephemeral", **({"ttl": ttl} if ttl else {})}
    assert memories["text"] == MEMORIES
    assert prefix["cache_control"] == want
    assert memories.get("cache_control") == (want if cache_memories else None)
    assert "cache_control" not in tail


def test_memora_answer_goes_through_the_shared_request(lme, tmp_path, monkeypatch):
    llm = install_llm(monkeypatch, RecordingLLM(reply=lambda r: "Step 1...\nFINAL ANSWER: a red bike"))
    search = importlib.import_module("providers.memora.search")
    answerer = importlib.import_module("answerer")
    s = search.MemoraSearch(load_cfg(lme), str(tmp_path / "out.json"), 5, retrieval_strategy="semantic")
    mems = [{"index": "m1", "value": "User owns a red bike", "timestamp": "2023/05/09", "score": 0.9,
             "memory_type": "factual"}]
    monkeypatch.setattr(s, "search_memory", lambda *a, **k: (mems, 0.0))
    item = {"question_id": "q1", "question": QUESTION, "answer": "red bike",
            "question_type": "single-session-user", "question_date": QUESTION_DATE}

    out = s.process_question(item, 0)

    (req,) = llm.calls
    formatted = "\n".join(["\n=== Factual Details ===\n", "2023/05/09: User owns a red bike"])
    assert user_text(req) == memora_render(formatted, QUESTION, QUESTION_DATE)
    assert req.params == answerer.build_answer_req(formatted, QUESTION, QUESTION_DATE).params
    p = req.params
    assert (p["model"], p["output_config"], p["max_tokens"]) == (MODEL, {"effort": "low"}, 8192)
    assert not FORBIDDEN & set(p)
    assert out["generated_answer"] == "a red bike"


# --------------------------------------------------------------------------- tiny LongMemEval-shaped data


def lme_item(qid, qtype, question, answer, sessions, question_date="2023/06/01 (Thu) 09:00"):
    return {
        "question_id": qid, "question_type": qtype, "question": question, "answer": answer,
        "question_date": question_date,
        "haystack_session_ids": [f"{qid}_s{i}" for i in range(len(sessions))],
        "haystack_dates": [date for date, _ in sessions],
        "haystack_sessions": [turns for _, turns in sessions],
        "answer_session_ids": [f"{qid}_s0"],
    }


def u(text, **kw):
    return {"role": "user", "content": text, **kw}


def a(text):
    return {"role": "assistant", "content": text}


def tiny_lme() -> list[dict]:
    """Three questions with their own haystacks; dataset order is chronological by day only (as in
    longmemeval_s_cleaned.json), so the haystack order is kept, not re-sorted."""
    return [
        lme_item("q-a", "single-session-user", "What pet did I adopt?", "a cat", [
            ("2023/05/20 (Sat) 02:21", [u("I adopted a cat named Luna.", has_answer=True), a("Congrats on Luna!")]),
            ("2023/05/20 (Sat) 01:57", [u("Plan a trip to\nOslo for me."), a("Day 1: the fjord {walk}.")]),
        ]),
        lme_item("q-b", "multi-session", "How many bikes do I own?", 2, [
            ("2023/04/01 (Sat) 10:00", [u("I bought a red bike.", has_answer=True), a("Nice bike.")]),
        ]),
        lme_item("q-c_abs", "temporal-reasoning", "When did I buy my hamster?", "You did not mention a hamster.", [
            ("2023/03/01 (Wed) 08:00", [u("My cat Luna is sick."), a("Sorry to hear that.")]),
            ("2023/03/02 (Thu) 09:30", [u("Luna is better today."), a("Glad she recovered.")]),
            ("2023/03/05 (Sun) 18:45", [u("I start a new job Monday."), a("Good luck!")]),
        ]),
    ]


ITEMS = {it["question_id"]: it for it in tiny_lme()}
SELECTED = ["q-a", "q-c_abs"]  # eval.subset_idx = "1,3" (1-based, as run_memora.py / search.py read it)


def assert_history_has_every_session_in_order(text: str, item: dict):
    pos = 0
    for date, session in zip(item["haystack_dates"], item["haystack_sessions"]):
        at = text.find(date, pos)
        assert at >= 0, f"session date {date!r} missing or out of order"
        pos = at
        for turn in session:
            at = text.find(turn["content"], pos)
            assert at >= 0, f"turn {turn['content']!r} missing or out of order"
            pos = at + len(turn["content"])
    assert "has_answer" not in text  # the evidence label never reaches the answerer


def lme_cfg(app, tmp_path, **memory):
    cfg = load_cfg(app)
    cfg.eval.subset_idx = "1,3"
    cfg.general.project_path = str(tmp_path)
    for k, v in memory.items():
        cfg.memory[k] = v
    return cfg


# --------------------------------------------------------------------------- full context (batch)


def test_full_context_one_batch_req_per_selected_question_with_its_whole_history(lme, tmp_path, monkeypatch):
    llm = install_llm(monkeypatch, RecordingLLM())
    fc = importlib.import_module("providers.full_context")

    fc.FullContextManager(cfg=lme_cfg(lme, tmp_path), data_path=str(write_json(tmp_path / "d.json", tiny_lme()))) \
        .process_all_questions(str(tmp_path / "out.json"))

    assert llm.calls == []
    ((reqs, mode, _),) = llm.runs
    assert mode == "batch"
    assert sorted(r.meta["question_id"] for r in reqs) == SELECTED
    for r in reqs:
        item = ITEMS[r.meta["question_id"]]
        meta = (r.meta["system"], r.meta["conv_id"], r.meta["stage"])
        assert meta == ("full-context", item["question_id"], "answer")
        # every question has its own haystack: no reuse, so no warm-up group (it would run real-time) and no 1h write
        assert "prefix_group" not in r.meta
        history = blocks(r)[1]
        assert "cache_control" not in history
        assert user_text(r) == memora_render(history["text"], item["question"], item["question_date"])
        assert_history_has_every_session_in_order(history["text"], item)


# --------------------------------------------------------------------------- RAG (batch, chunk 500, k=3)

MARKERS = ["alpha", "bravo", "charlie", "delta"]
QUERY_VEC = [0.9, 0.8, 0.7, 0.1, 0.0]  # alpha > bravo > charlie > delta; filler chunks score 0


def rag_data() -> list[dict]:
    filler = " ".join(["the weather was calm and the river was slow"] * 70)  # ~700 tokens: one marker per chunk
    sessions = [(f"2023/05/{10 + i} (Wed) 10:00", [u(f"{filler} {m} {filler}"), a("ok")]) for i, m in enumerate(MARKERS)]
    return [
        lme_item("r-1", "multi-session", "Which marker came first?", "alpha", sessions),
        lme_item("r-2", "multi-session", "Not selected?", "x", sessions[:1]),
        lme_item("r-3_abs", "single-session-user", "Which marker came last?", "delta", sessions),
    ]


def install_fake_embed(monkeypatch) -> list[tuple[str, list[str]]]:
    calls = []

    def fake_embed(texts, kind):
        texts = list(texts)
        calls.append((kind, texts))
        if kind == "query":
            return [list(QUERY_VEC) for _ in texts]
        return [[float(m in t) for m in MARKERS] + [0.01] for t in texts]

    fake = types.ModuleType("memobench.embed")
    fake.embed = fake_embed
    monkeypatch.setitem(sys.modules, "memobench.embed", fake)
    monkeypatch.setattr(sys.modules["memobench"], "embed", fake, raising=False)
    return calls


def test_rag_answers_from_top3_chunks_of_at_most_500_tokens_in_one_batch(lme, tmp_path, monkeypatch):
    import tiktoken

    embed_calls = install_fake_embed(monkeypatch)
    llm = install_llm(monkeypatch, RecordingLLM())
    rag = importlib.import_module("providers.rag")
    cfg = lme_cfg(lme, tmp_path)
    data = {it["question_id"]: it for it in rag_data()}

    rag.RAGManager(cfg=cfg, data_path=str(write_json(tmp_path / "d.json", rag_data())),
                   chunk_size=cfg.memory.chunk_size, k=cfg.memory.num_chunks) \
        .process_all_questions(str(tmp_path / "out.json"))

    kinds = defaultdict(list)
    for kind, texts in embed_calls:
        kinds[kind] += texts
    assert set(kinds) == {"document", "query"}
    assert sorted(kinds["query"]) == ["Which marker came first?", "Which marker came last?"]
    enc = tiktoken.get_encoding("cl100k_base")
    assert len(kinds["document"]) >= 6 and all(len(enc.encode(c)) <= 500 for c in kinds["document"])
    assert llm.calls == []
    ((reqs, mode, _),) = llm.runs
    assert mode == "batch"
    assert sorted(r.meta["question_id"] for r in reqs) == ["r-1", "r-3_abs"]
    for r in reqs:
        item = data[r.meta["question_id"]]
        context = blocks(r)[1]
        parts = context["text"].split("\n<->\n")
        assert len(parts) == 3
        assert [m for p in parts for m in MARKERS if m in p] == ["alpha", "bravo", "charlie"]
        assert all(p in kinds["document"] for p in parts)
        assert "cache_control" not in context  # retrieved chunks differ per question
        assert user_text(r) == memora_render(context["text"], item["question"], item["question_date"])
        assert (r.meta["system"], r.meta["conv_id"], r.meta["stage"]) == ("rag", item["question_id"], "answer")


# --------------------------------------------------------------------------- runner: answer batch -> judge batch -> scores


@pytest.mark.parametrize("memory_type,system", [("full_context", "full-context"), ("rag", "rag")])
def test_runner_answers_then_judges_and_scores_per_type_overall_and_abstention(lme, tmp_path, monkeypatch,
                                                                               memory_type, system):
    install_fake_embed(monkeypatch)

    def reply(req):
        if req.meta["stage"] == "judge":  # only the abstention question is judged correct
            return json.dumps({"label": "yes" if req.meta["question_id"].endswith("_abs") else "no"})
        return "Step 1: ...\nFINAL ANSWER: stub answer"

    llm = install_llm(monkeypatch, RecordingLLM(reply=reply))
    (tmp_path / "data").mkdir()
    write_json(tmp_path / "data" / "longmemeval_s_cleaned.json", tiny_lme())
    runner = importlib.import_module("run_experiments")
    evals = importlib.import_module("evals")

    output_dir = runner.run_baseline(lme_cfg(lme, tmp_path, type=memory_type))

    assert llm.calls == []
    assert [(mode, sorted({r.meta["stage"] for r in reqs}), sorted(r.meta["question_id"] for r in reqs))
            for reqs, mode, _ in llm.runs] == [("batch", ["answer"], SELECTED), ("batch", ["judge"], SELECTED)]
    assert {r.meta["system"] for reqs, _, _ in llm.runs for r in reqs} == {system}
    answers = evals._load_input_data(str(tmp_path / output_dir / f"{memory_type}_output.json"))
    assert {e["question_id"]: e["generated_answer"] for e in answers} == {q: "stub answer" for q in SELECTED}
    scores = json.loads((tmp_path / output_dir / f"{memory_type}_scores.json").read_text())
    assert scores["mean_scores_per_question_type"]["llm_score"] == {"single-session-user": 0.0,
                                                                    "temporal-reasoning": 1.0}
    assert scores["overall_mean_scores"]["llm_score"] == 0.5
    assert scores["abstention"] == {"count": 1, "llm_score": 1.0}


def test_rag_chunks_special_token_text_as_plain_text(lme, tmp_path, monkeypatch):
    # longmemeval_s_cleaned.json has a turn containing the literal text "<|endoftext|>" (question gpt4_78cf46a3);
    # tiktoken refuses to encode it by default, which would abort the whole RAG run.
    embed_calls = install_fake_embed(monkeypatch)
    install_llm(monkeypatch, RecordingLLM())
    rag = importlib.import_module("providers.rag")
    cfg = lme_cfg(lme, tmp_path)
    cfg.eval.subset_idx = -1
    item = lme_item("s-1", "multi-session", "What ends the text?", "x",
                    [("2023/05/10 (Wed) 10:00", [u("A model emits <|endoftext|> at the end."), a("Right.")])])

    rag.RAGManager(cfg=cfg, data_path=str(write_json(tmp_path / "d.json", [item])), chunk_size=500, k=3) \
        .process_all_questions(str(tmp_path / "out.json"))

    docs = [t for kind, texts in embed_calls if kind == "document" for t in texts]
    assert "".join(docs) == importlib.import_module("utils").render_history(item)
