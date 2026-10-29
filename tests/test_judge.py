"""Batch judges (port-plan Part 2c): LoCoMo = Mem0 ACCURACY_PROMPT verbatim, LongMemEval = upstream
evaluate_qa.py get_anscheck_prompt verbatim; labels via structured outputs; all reqs in ONE batch run.

Expected prompts come from the vendored upstream sources (vendor/upstream/), not from the code under test.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import re
import sys
import types

import pytest

from tests.apphelpers import MODEL, UPSTREAM, RecordingLLM, install_llm, load_cfg, use_app, user_text, write_json

FORBIDDEN = {"temperature", "top_p", "top_k", "seed"}


def upstream_accuracy_prompt() -> str:
    """Mem0's ACCURACY_PROMPT (vendor/upstream/mem0_llm_judge.py; importing it needs mem0 + an OpenAI key)."""
    for node in ast.parse((UPSTREAM / "mem0_llm_judge.py").read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "ACCURACY_PROMPT":
            return ast.literal_eval(node.value)
    raise AssertionError("ACCURACY_PROMPT not found upstream")


def upstream_anscheck(monkeypatch):
    """get_anscheck_prompt from vendor/upstream/lme_evaluate_qa.py (its `backoff` import is stubbed: not installed)."""
    backoff = types.ModuleType("backoff")
    backoff.expo = object()
    backoff.on_exception = lambda *a, **k: (lambda f: f)
    monkeypatch.setitem(sys.modules, "backoff", backoff)
    spec = importlib.util.spec_from_file_location("upstream_lme_evaluate_qa", UPSTREAM / "lme_evaluate_qa.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_anscheck_prompt


def assert_frozen_judge_params(req, labels):
    p = req.params
    assert p["model"] == MODEL and p["max_tokens"] == 2048 and not FORBIDDEN & set(p)
    assert p["output_config"]["effort"] == "low"
    fmt = p["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["properties"]["label"]["enum"] == labels
    assert fmt["schema"]["required"] == ["label"]
    assert req.meta["stage"] == "judge"


# --------------------------------------------------------------------------- LoCoMo

LOCOMO_RESULTS = {
    "0": [
        {"question": "What pet did Ann adopt?", "answer": "a cat", "response": "a cat", "category": "4"},
        {"question": "Where did Ann move to?", "answer": "", "adversarial_answer": "Oslo", "response": "Oslo",
         "category": "5"},
        {"question": "When did Bob move?", "answer": "9 May 2023", "response": "June 2023", "category": "2"},
    ],
    "1": [{"question": "What sport does Cy do?", "answer": "running", "response": "running", "category": 1}],
}
ANSWERABLE = [it for v in LOCOMO_RESULTS.values() for it in v if str(it["category"]) != "5"]


def label_by_exact_match(req):
    text = user_text(req)  # the prompt's own example also has a "Gold answer:" line: take the real (last) one
    gold = re.findall(r"\nGold answer: (.*)\n", text)[-1]
    gen = re.findall(r"\nGenerated answer: (.*)\n", text)[-1]
    return json.dumps({"label": "CORRECT" if gold == gen else "WRONG"})


@pytest.fixture
def locomo(monkeypatch):
    return use_app(monkeypatch, "locomo")


def test_locomo_judge_sends_mem0_accuracy_prompt_in_one_batch(locomo, tmp_path, monkeypatch):
    llm = install_llm(monkeypatch, RecordingLLM(reply=label_by_exact_match))
    evals = importlib.import_module("evals")

    evals.evaluate(load_cfg(locomo), str(write_json(tmp_path / "out.json", LOCOMO_RESULTS)), str(tmp_path / "eval.json"))

    assert llm.calls == []
    ((reqs, mode, _),) = llm.runs
    assert mode == "batch"
    prompt = upstream_accuracy_prompt()
    assert sorted(user_text(r) for r in reqs) == sorted(
        prompt.format(question=it["question"], gold_answer=str(it["answer"]), generated_answer=str(it["response"]))
        for it in ANSWERABLE
    )
    for r in reqs:
        assert_frozen_judge_params(r, ["CORRECT", "WRONG"])


def test_locomo_judge_label_becomes_the_llm_score_generate_scores_reads(locomo, tmp_path, monkeypatch):
    install_llm(monkeypatch, RecordingLLM(reply=label_by_exact_match))
    evals = importlib.import_module("evals")
    eval_file, score_file = tmp_path / "eval.json", tmp_path / "scores.json"

    evals.evaluate(load_cfg(locomo), str(write_json(tmp_path / "out.json", LOCOMO_RESULTS)), str(eval_file))
    scores = evals.generate_scores(str(eval_file), str(score_file))

    judged = {it["question"]: it["llm_score"] for v in json.loads(eval_file.read_text()).values() for it in v}
    assert judged == {"What pet did Ann adopt?": 1, "When did Bob move?": 0, "What sport does Cy do?": 1}
    assert scores["overall_mean_scores"]["llm_score"] == round(2 / 3, 4)
    assert json.loads(score_file.read_text())["summary"]["total_questions"] == 3


# --------------------------------------------------------------------------- LongMemEval

# one real item (vendor dataset ids) per question_type, plus one abstention item
LME_ENTRIES = [
    {"question_id": "e47becba", "question_type": "single-session-user",
     "question": "What degree did I graduate with?", "answer": "Business Administration"},
    {"question_id": "7161e7e2", "question_type": "single-session-assistant",
     "question": "I'm checking our previous chat about the shift rotation sheet. Who was on Sundays?",
     "answer": "Admon was assigned to the 8 am - 4 pm (Day Shift) on Sundays."},
    {"question_id": "0a995998", "question_type": "multi-session",
     "question": "How many items of clothing do I need to pick up or return from a store?", "answer": 3},
    {"question_id": "gpt4_59149c77", "question_type": "temporal-reasoning",
     "question": "How many days passed between my visit to MoMA and the 'Ancient Civilizations' exhibit?",
     "answer": "7 days. 8 days (including the last day) is also acceptable."},
    {"question_id": "6a1eabeb", "question_type": "knowledge-update",
     "question": "What was my personal best time in the charity 5K run?", "answer": "25 minutes and 50 seconds (or 25:50)"},
    {"question_id": "8a2466db", "question_type": "single-session-preference",
     "question": "Can you recommend some resources where I can learn more about video editing?",
     "answer": "The user would prefer responses that suggest resources specifically tailored to Adobe Premiere Pro."},
    {"question_id": "0862e8bf_abs", "question_type": "single-session-user",
     "question": "What is the name of my hamster?",
     "answer": "You did not mention this information. You mentioned your cat Luna but not your hamster."},
]
for i, e in enumerate(LME_ENTRIES):
    e["generated_answer"] = f"OFF-TOPIC-{i}" if i % 2 else e["answer"]


@pytest.fixture
def lme(monkeypatch):
    return use_app(monkeypatch, "longmemeval")


def test_lme_judge_prompt_is_upstream_anscheck_per_question_type_and_abstention(lme, tmp_path, monkeypatch):
    get_anscheck_prompt = upstream_anscheck(monkeypatch)
    llm = install_llm(monkeypatch, RecordingLLM(reply=lambda r: '{"label": "yes"}'))
    evals = importlib.import_module("evals")

    evals.evaluate(load_cfg(lme), str(write_json(tmp_path / "out.json", LME_ENTRIES)), str(tmp_path / "x_eval.json"),
                   use_question_type_eval=True)

    assert llm.calls == []
    ((reqs, mode, _),) = llm.runs
    assert mode == "batch"
    sent = {r.meta["question_id"]: user_text(r) for r in reqs}
    assert sent == {
        e["question_id"]: get_anscheck_prompt(e["question_type"], e["question"], e["answer"], e["generated_answer"],
                                              abstention="_abs" in e["question_id"])
        for e in LME_ENTRIES
    }
    for r in reqs:
        assert_frozen_judge_params(r, ["yes", "no"])


def test_lme_judge_yes_no_label_becomes_llm_score(lme, tmp_path, monkeypatch):
    install_llm(monkeypatch, RecordingLLM(
        reply=lambda r: json.dumps({"label": "no" if "OFF-TOPIC-" in user_text(r) else "yes"})))
    evals = importlib.import_module("evals")
    out = tmp_path / "x_eval.json"

    scores = evals.evaluate(load_cfg(lme), str(write_json(tmp_path / "out.json", LME_ENTRIES)), str(out),
                            use_question_type_eval=True)

    judged = {it["question_id"]: it["llm_score"] for it in json.loads(out.read_text())}
    assert judged == {e["question_id"]: (0.0 if "OFF-TOPIC-" in str(e["generated_answer"]) else 1.0)
                      for e in LME_ENTRIES}
    assert scores["overall_mean_scores"]["llm_score"] == round(4 / 7, 4)
