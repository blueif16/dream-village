"""scorer.answers: every (probe, system) answered by the same Sonnet 5.5 answerer, Batch-driven, one round per hop."""

from __future__ import annotations

import pytest

from scorer import answers as AN
from scorer.gold import Gold
from scorer.probes import Probe
from tests import scorer_village as V
from tests.memsys_fakes import tool_use
from tests.scorer_fakes import make_llm, user_texts
from tests.scorer_replay import make_replay

T = "2025-09-08T12:00:00"  # after s01-s04, before s05
FOOD = "---\nname: food\ndescription: Food tastes\nsources: [chat]\naliases: []\n---\n\n- [stated] vegetarian (kai.s03)\n"


@pytest.fixture
def gold(tmp_path):
    return Gold(V.build(tmp_path / "village"))


@pytest.fixture
def replay(tmp_path):
    return make_replay(tmp_path / "replay", [
        ("2025-09-05T19:00:00", "write", "kai.s03", {"/profile.md": "- [stated] student\n", "/topics/food.md": FOOD}),
    ])  # fmt: skip


def probe(pid, kind="fact", t=T, request="plan my dinners this week?"):
    return Probe(pid, V.P, "cp", t, kind, request, ["kai.f02"], [], [], [], [], {}, None, {})


def context(params) -> str:
    return "".join(user_texts(params))


def v0_then_text(params):
    """v0 reads its food file on the first hop, then answers; every other request answers at once."""
    is_v0 = any(t.get("name") == "memory_read" for t in params.get("tools", []))
    tool_results = [b for b in params["messages"][-1]["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]
    if is_v0 and not tool_results:
        return {"content": [tool_use("memory_read", paths=["/topics/food.md"])], "stop_reason": "tool_use",
                "usage": {"input_tokens": 900, "output_tokens": 50, "cache_creation_input_tokens": 0,
                          "cache_read_input_tokens": 0}}  # fmt: skip
    return {"text": "Lentil tacos, veggie chili.", "usage": {"input_tokens": 1200, "output_tokens": 80,
            "cache_creation_input_tokens": 300, "cache_read_input_tokens": 500}}  # fmt: skip


def test_all_pairs_run_together_one_batch_round_per_tool_hop(gold, replay, tmp_path):
    llm, fake = make_llm(tmp_path / "runs", v0_then_text)
    probes = [probe("p1"), probe("p2", request="what should I cook tomorrow?")]
    out = AN.run_answers(llm, gold, probes, ["v0", "no_memory", "oracle"], replay=replay)
    rounds = fake.batch_creates
    assert [len(r) for r in rounds] == [6, 2]  # round 1: every pair's first request; round 2: v0's second hop
    v0 = [a for a in out if a["system"] == "v0"]
    assert {a["hops"] for a in v0} == {2} and all("Lentil" in a["text"] for a in v0)
    assert {a["hops"] for a in out if a["system"] != "v0"} == {1}


def test_v0_answers_from_its_memory_at_t_and_records_context_tokens(gold, replay, tmp_path):
    llm, fake = make_llm(tmp_path / "runs", v0_then_text)
    [a] = AN.run_answers(llm, gold, [probe("p1")], ["v0"], replay=replay)
    first = fake.seen[0]
    assert "- [stated] student" in context(first)  # the profile at t is injected
    second = fake.seen[1]
    assert "vegetarian" in str(second["messages"][-1]["content"])  # memory_read returned the file at t
    assert a["context_tokens"] == 1200 + 300 + 500  # the last hop's prompt
    assert a["prompt_tokens_total"] == 900 + 2000
    assert first["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_every_system_uses_the_same_answerer_model(gold, replay, tmp_path):
    llm, fake = make_llm(tmp_path / "runs", v0_then_text)
    AN.run_answers(llm, gold, [probe("p1")], list(AN.SYSTEMS), replay=replay, full_context_ids={"p1"})
    assert {p["model"] for p in fake.seen} == {"claude-sonnet-5-5"}
    assert {p["output_config"]["effort"] for p in fake.seen} == {"medium"}


def test_no_memory_sees_only_the_date_and_the_request(gold, tmp_path):
    req = AN.baseline_req(gold, "no_memory", probe("p1"))
    text = context(req.params)
    assert "plan my dinners this week?" in text and "2025-09-08" in text
    assert "carnitas" not in text and "<recent_sessions>" not in text and "<memory>" not in text


def test_recent_window_holds_the_last_sessions_before_t(gold, monkeypatch):
    monkeypatch.setattr(AN, "WINDOW", 2)
    text = context(AN.baseline_req(gold, "recent_window", probe("p1")).params)
    assert "kai.s03" in text and "kai.s04" in text
    assert "kai.s02" not in text and "kai.s05" not in text


def test_full_context_holds_every_session_before_t(gold, monkeypatch):
    monkeypatch.setattr(AN, "WINDOW", 2)
    text = context(AN.baseline_req(gold, "full_context", probe("p1")).params)
    assert all(f"kai.s0{i}" in text for i in (1, 2, 3, 4)) and "kai.s05" not in text


def test_oracle_holds_the_oracle_memory_and_the_recent_window(gold):
    text = context(AN.baseline_req(gold, "oracle", probe("p1")).params)
    assert gold.oracle_memory(V.P, T).strip() in text
    assert "kai.s04" in text


def test_answers_at_one_checkpoint_share_the_context_prefix(gold):
    a = AN.baseline_req(gold, "recent_window", probe("p1", request="one"))
    b = AN.baseline_req(gold, "recent_window", probe("p2", request="two"))
    assert user_texts(a.params)[0] == user_texts(b.params)[0]
    assert a.params["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_full_context_runs_only_on_the_chosen_subset(gold, replay, tmp_path):
    llm, fake = make_llm(tmp_path / "runs", v0_then_text)
    out = AN.run_answers(llm, gold, [probe("p1"), probe("p2")], ["full_context"], full_context_ids={"p2"})
    assert [a["probe_id"] for a in out] == ["p2"]


def test_stratified_subset_takes_up_to_n_per_principal_and_kind():
    probes = [probe(f"f{i}") for i in range(4)] + [probe(f"r{i}", kind="restraint") for i in range(3)]
    chosen = AN.stratified_subset(probes, 2)
    assert len(chosen) == 4
    assert sum(c.startswith("f") for c in chosen) == 2 and sum(c.startswith("r") for c in chosen) == 2
    assert chosen == AN.stratified_subset(list(reversed(probes)), 2)  # deterministic


def test_a_refused_answer_is_recorded_as_refused(gold, tmp_path):
    llm, _ = make_llm(tmp_path / "runs", lambda p: {"content": [], "stop_reason": "refusal"})
    [a] = AN.run_answers(llm, gold, [probe("p1")], ["no_memory"])
    assert a["refused"] is True and a["text"] == ""
