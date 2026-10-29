"""scorer.run: the CLI chains every stage, refuses to exceed --budget, and resumes from the response cache."""

from __future__ import annotations

import json
import re

import pytest

from scorer import run as R
from tests import scorer_village as V
from tests.scorer_fakes import json_reply, make_llm, user_texts
from tests.scorer_replay import make_replay

FOOD = "---\nname: food\ndescription: Food tastes\nsources: [chat]\naliases: []\n---\n\n- [stated] vegetarian (kai.s03)\n"
TARGETS = {"fact": {"must_use": ["kai.f02"]}, "impression": {"must_use": ["kai.i01"], "criteria": ["names the award"]},
           "reference": {"must_use": ["kai.r01"]}, "restraint": {"must_not": ["cast.f01"]}}  # fmt: skip


def respond(params):
    system = params["system"][0]["text"]
    text = "".join(user_texts(params))
    if system.startswith("You are a senior evaluation designer"):  # writes each slot around its required target
        slots = re.findall(r"slot (\d+): (\w+); required target: (\S+)", text)
        return json_reply({"probes": [{"slot": int(i), "kind": k, "plan": "p", "request": f"help with {k} {i}",
                                       "must_use": [], "may_use": [], "must_not": [], "stale": [], "criteria": [],
                                       "forbidden": [], **TARGETS[k],
                                       **({"must_not": [r]} if k == "restraint" else {"must_use": [r]})}
                                      for i, k, r in slots]})  # fmt: skip
    if system.startswith("You are a strict, careful grader"):
        ids = re.search(r"<judge_these>\n(.*?)\n</judge_these>", text, re.S).group(1).split("\n")
        return json_reply({"verdicts": [{"check": c, "evidence": "none", "pass": True} for c in ids]})
    if system.startswith("You are a precise matcher"):
        ns = re.findall(r'<line n="(\d+)"', text)
        return json_reply({"lines": [{"n": int(n), "kind": "fact", "entries": [
            {"id": "kai.f02", "relation": "consistent", "current": True}]} for n in ns]})  # fmt: skip
    return {"text": "Here is a plan."}


@pytest.fixture
def setup(tmp_path):
    village = V.build(tmp_path / "village")
    replay = make_replay(tmp_path / "replay", [
        ("2025-09-05T19:00:00", "write", "kai.s03", {"/profile.md": "- [stated] student\n", "/topics/food.md": FOOD}),
    ])  # fmt: skip
    llm, fake = make_llm(tmp_path / "runs", respond)
    args = ["--village", str(village), "--replay", str(replay), "--out", str(tmp_path / "out"), "--budget", "5",
            "--probes-per-checkpoint", "4", "--max-checkpoints", "2"]  # fmt: skip
    return args, llm, fake, tmp_path / "out"


def test_every_stage_runs_and_writes_the_scorecard(setup):
    args, llm, fake, out = setup
    assert R.main(args, llm=llm) == 0
    for name in ("probes.jsonl", "answers.jsonl", "verdicts.jsonl", "audit.json", "scorecard.json", "scorecard.md",
                 "cards/cards.jsonl", "cards/cards.csv", "costs.json"):
        assert (out / name).exists(), name
    card = json.loads((out / "scorecard.json").read_text())
    assert set(card["systems"]) == {"v0", "no_memory", "recent_window", "oracle", "full_context"}
    assert card["systems"]["v0"]["remembering"]["provisional"] is True
    assert card["systems"]["v0"]["integrity"]["score"] is not None
    assert "provisional (Claude-only judges, D28)" in (out / "scorecard.md").read_text()


def test_the_markdown_reports_judge_disagreement_per_system(setup):
    args, llm, _, out = setup
    R.main(args, llm=llm)
    md = (out / "scorecard.md").read_text()
    assert "Judge disagreement" in md
    assert re.search(r"\| v0 \| \d+/\d+ \(", md)  # disagree/both (share) on must_not, per system


def test_probes_log_reports_planned_and_made_targets_per_lag_bucket_and_the_eval_version(setup):
    args, llm, _, out = setup
    R.main([*args, "--stages", "probes"], llm=llm)
    log = json.loads((out / "probes_log.json").read_text())
    assert log["eval_version"] == "dv-eval-v2"
    kai = log["buckets"]["kai"]
    assert sum(kai["planned"].values()) == 8 and set(kai["planned"]) <= {"old", "after_consolidation", "recent"}
    assert kai["made"] == kai["planned"]  # the fake generator keeps every required target
    assert json.loads((out / "probes.jsonl").read_text().splitlines()[0])["required"]


def test_cost_and_gates_come_from_the_replay_timeline_and_its_memory(setup):
    args, llm, _, out = setup
    R.main(args, llm=llm)
    v0 = json.loads((out / "scorecard.json").read_text())["systems"]["v0"]
    assert v0["cost"]["write_usd_per_sim_day"] == pytest.approx(0.01 / 21)  # one $0.01 commit, 21 simulated days
    assert v0["gates"]["size_cap"]["profile_words_max"] == 2  # "- [stated] student"
    assert v0["gates"]["output_contracts"]["memory_frontmatter_violations"] == 0


def test_the_scorecard_records_whether_observed_disclosures_were_counted(setup):
    args, llm, _, out = setup
    R.main([*args, "--observed-disclosures"], llm=llm)
    assert json.loads((out / "scorecard.json").read_text())["eval"]["observed_disclosures"] is True


def test_a_second_run_is_served_from_the_response_cache(setup):
    args, llm, fake, _ = setup
    R.main(args, llm=llm)
    sent = len(fake.seen)
    assert R.main(args, llm=llm) == 0
    assert len(fake.seen) == sent


def test_an_aa_run_of_v0_alone_keeps_run_1s_shortcut_flags_and_compares_against_it(setup, tmp_path):
    args, llm, _, out = setup
    R.main(args, llm=llm)
    first = json.loads((out / "scorecard.json").read_text())["eval"]["shortcuts_flagged"]
    assert first  # the fake no_memory passes every check, so memory-needing probes are flagged in run 1
    b = tmp_path / "out_b"
    run2 = [*args[:4], "--out", str(b), "--probes-from", str(out), "--systems", "v0", "--compare", str(out),
            "--budget", "5", "--salt", "B"]  # fmt: skip
    assert R.main(run2, llm=llm) == 0
    card = json.loads((b / "scorecard.json").read_text())
    assert card["eval"]["shortcuts_flagged"] == first  # no no_memory in run 2: run 1's filter still applies
    assert card["comparison"]["flip_checks"] > 0


def test_other_runs_spending_on_the_shared_ledger_do_not_use_up_this_runs_budget(tmp_path):
    village = V.build(tmp_path / "village")
    replay = make_replay(tmp_path / "replay", [
        ("2025-09-05T19:00:00", "write", "kai.s03", {"/profile.md": "- [stated] student\n", "/topics/food.md": FOOD}),
    ])  # fmt: skip
    ledger = tmp_path / "runs" / "ledger.jsonl"

    def busy_neighbour(params):  # another process (e.g. village generation) spends $4.9 while we run
        if not ledger.exists() or "generator" not in ledger.read_text():
            with open(ledger, "a") as f:
                f.write(json.dumps({"run_id": "generator", "stage": "x", "key": "k", "cost_usd": 4.9, "mode": "batch",
                                    "ts": "t"}) + "\n")  # fmt: skip
        return respond(params)

    llm, _ = make_llm(tmp_path / "runs", busy_neighbour)
    args = ["--village", str(village), "--replay", str(replay), "--out", str(tmp_path / "out"), "--budget", "5",
            "--probes-per-checkpoint", "4", "--max-checkpoints", "2"]  # fmt: skip
    assert R.main(args, llm=llm) == 0
    assert (tmp_path / "out" / "scorecard.json").exists()


def test_a_stage_whose_estimate_exceeds_the_budget_is_refused_before_anything_is_sent(setup):
    args, llm, fake, out = setup
    assert R.main([*args[:-6], "--budget", "0.0001", *args[-4:], "--stages", "probes"], llm=llm) == 2
    assert fake.seen == [] and not (out / "probes.jsonl").exists()


def test_config_flag_answers_v0_with_that_candidates_prompts(setup):
    # optimization round 1: C1/C5 change only the answer prompt, so the scorer must answer with the candidate's config
    args, llm, fake, _ = setup
    assert R.main([*args, "--stages", "probes,answers", "--systems", "v0", "--config", "memsys/config/c1.json"],
                  llm=llm) == 0  # fmt: skip
    systems = ["".join(b["text"] for b in p["system"]) for p in fake.seen if isinstance(p.get("system"), list)]
    assert any("<read_memory_first>" in s for s in systems)
