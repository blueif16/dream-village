"""village.generate: the code-side logic of the generator (schedule, task mix, id scoping, required disclosures, plan
repair, verification flags) and its Batch round runner, driven through the real LLM layer on the fake client."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime

import pytest

from tests.fakes import FakeAnthropic, prompt_of
from village import generate as g
from village.schema import (
    Arc, Beat, Bible, Claim, DisclosureItem, Episode, Fact, Impression, Reference, ScheduledSession, SessionPlan,
    Usage, Voice, World,
)  # fmt: skip
from village.validate import validate

DAYS = g.horizon_days()


def F(id, attribute, value, valid_from=None, valid_to=None, superseded_by=None, class_="useful", subject="kai") -> Fact:
    return Fact(id, subject, attribute, value, valid_from, valid_to, superseded_by, class_)


def sessions(*starts: str, principal="kai") -> list[ScheduledSession]:
    week = lambda s: (date.fromisoformat(s[:10]) - DAYS[0]).days // 7 + 1  # noqa: E731
    return [ScheduledSession(f"{principal}.s{n:02d}", principal, s, week(s), "practical_guidance")
            for n, s in enumerate(starts, 1)]  # fmt: skip


# --------------------------------------------------------------------------- horizon + schedule


def test_horizon_is_four_weeks_from_monday_2025_08_25():
    assert DAYS[0] == date(2025, 8, 25) and DAYS[0].weekday() == 0
    assert len(DAYS) == 28 and DAYS[-1] == date(2025, 9, 21)


@pytest.mark.parametrize("frequency,low,high", [("daily", 5, 8), ("several_per_week", 3, 4)])
def test_schedule_sessions_per_week_follow_usage_frequency(frequency, low, high):
    sched = g.make_schedule("kai", Usage(frequency, [7, 13, 22]), DAYS)
    per_week = Counter(s.week for s in sched)
    assert sorted(per_week) == [1, 2, 3, 4]
    assert all(low <= n <= high for n in per_week.values()), per_week


def test_schedule_is_deterministic():
    usage = Usage("daily", [7, 22])
    assert g.make_schedule("kai", usage, DAYS) == g.make_schedule("kai", usage, DAYS)


def test_schedule_differs_between_principals():
    usage = Usage("daily", [7, 22])
    starts = lambda p: [s.start for s in g.make_schedule(p, usage, DAYS)]  # noqa: E731
    assert starts("kai") != starts("nadia")


def test_schedule_is_chronological_inside_horizon_and_in_active_hours():
    sched = g.make_schedule("kai", Usage("several_per_week", [6, 23]), DAYS)
    starts = [datetime.fromisoformat(s.start) for s in sched]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)
    assert all(DAYS[0] <= t.date() <= DAYS[-1] and t.hour in (6, 23) for t in starts)
    assert [s.id for s in sched] == [f"kai.s{n:02d}" for n in range(1, len(sched) + 1)]


def test_schedule_task_types_follow_the_usage_mix():
    sched = g.make_schedule("kai", Usage("daily", [7, 22]), DAYS)
    assert Counter(s.task_type for s in sched) == Counter(
        {k: v for k, v in g.task_counts(len(sched)).items() if v}
    )


def test_task_counts_largest_remainder_for_24_sessions():
    """Weights 29/24/16/8/10/5/7/2 (writing 24% split two-thirds editing); quotas for 24 hand-computed."""
    assert g.task_counts(24) == {
        "practical_guidance": 7, "seeking_information": 6, "writing_edit": 4, "writing_new": 2, "tutoring": 2,
        "technical_help": 1, "multimedia": 2, "personal_reflection": 0,
    }  # fmt: skip


@pytest.mark.parametrize("n", [1, 7, 14, 25, 101])
def test_task_counts_sum_to_n(n):
    assert sum(g.task_counts(n).values()) == n


# --------------------------------------------------------------------------- id scoping


def local_bible() -> Bible:
    facts = [
        F("f01", "job", "barista", valid_to="2025-09-08", superseded_by="f02"),
        F("f02", "job", "shift lead", valid_from="2025-09-08"),
    ]
    imp = Impression("i01", "Downplays wins", "kai", [Episode("e1", "2025-08-27", "x")], None, None, "", False, False)
    arcs = [Arc("a01", "promotion", "", [Beat("2025-09-08", "promoted", ["f02", "cast.f04"])])]
    return Bible("kai", "", facts, [imp], [Reference("r01", "the usual", "latte", "2025-08-26")], arcs,
                 Voice("", "", [], []), Usage("daily", [7]))  # fmt: skip


def test_scope_bible_prefixes_local_ids_and_rewrites_references():
    b = g.scope_bible(local_bible(), "kai")
    assert [f.id for f in b.facts] == ["kai.f01", "kai.f02"]
    assert b.facts[0].superseded_by == "kai.f02"
    assert b.impressions[0].id == "kai.i01" and b.impressions[0].evidence[0].id == "kai.i01.e1"
    assert b.references[0].id == "kai.r01" and b.arcs[0].id == "kai.a01"


def test_scope_bible_leaves_global_ids_alone():
    b = g.scope_bible(local_bible(), "kai")
    assert b.arcs[0].beats[0].facts == ["kai.f02", "cast.f04"]


# --------------------------------------------------------------------------- required disclosures


def truth_bible() -> Bible:
    facts = [
        F("kai.f01", "job", "barista", valid_to="2025-09-08", superseded_by="kai.f02"),
        F("kai.f02", "job", "shift lead", valid_from="2025-09-08"),
        F("kai.f03", "snack", "takis", class_="trivia"),
        F("kai.f05", "failed_driving_test", "twice", class_="forgotten"),
    ]
    imp = Impression(
        "kai.i01", "Downplays wins", "kai",
        [Episode("kai.i01.e1", "2025-08-26", ""), Episode("kai.i01.e2", "2025-08-26", ""),
         Episode("kai.i01.e3", "2025-09-10", "")],
        None, None, "", False, False,
    )  # fmt: skip
    return Bible("kai", "", facts, [imp], [Reference("kai.r01", "the usual", "latte", "2025-08-28")], [],
                 Voice("", "", [], []), Usage("daily", [8, 20]))  # fmt: skip


SCHED = sessions(
    "2025-08-25T08:00:00", "2025-08-26T08:00:00", "2025-08-26T20:00:00", "2025-08-28T08:00:00",
    "2025-09-01T08:00:00", "2025-09-03T08:00:00", "2025-09-08T20:00:00", "2025-09-11T08:00:00",
)  # fmt: skip


def required_by_item() -> dict[str, list[DisclosureItem]]:
    out: dict[str, list[DisclosureItem]] = {}
    for d in g.required_items(truth_bible(), SCHED):
        out.setdefault(d.item_id, []).append(d)
    return out


def test_required_change_goes_to_first_session_on_or_after_it():
    (d,) = required_by_item()["kai.f02"]
    assert d.session_id == "kai.s07" and d.mode == "stated" and not d.forget_request


def test_required_forget_request_comes_strictly_after_the_disclosure():
    told, forget = sorted(required_by_item()["kai.f05"], key=lambda d: d.forget_request)
    start = {s.id: s.start for s in SCHED}
    assert not told.forget_request and forget.forget_request
    assert start[forget.session_id] > start[told.session_id]


def test_required_impression_episodes_get_distinct_later_sessions():
    """e1 and e2 both happen 08-26: the 08-26 evening session (after 18:00) shows one, the next session the other."""
    items = required_by_item()["kai.i01"]
    assert [d.session_id for d in items] == ["kai.s03", "kai.s04", "kai.s08"]
    assert all(d.mode == "hinted" for d in items)


def test_required_reference_first_use_on_or_after_its_date():
    (d,) = required_by_item()["kai.r01"]
    assert d.session_id == "kai.s04"


def test_required_items_satisfy_the_validator(tmp_path):
    from village.schema import Cast, Person

    cast = Cast([Person("kai", "Kai", "principal", 19, "", "", [])], [], [])
    plans = {}
    for d in g.required_items(truth_bible(), SCHED):
        plans.setdefault(d.session_id, SessionPlan(d.session_id, "practical_guidance", "", [])).items.append(d)
    out = validate(horizon=(DAYS[0].isoformat(), DAYS[-1].isoformat()), cast=cast, bibles=[truth_bible()],
                   life_logs=[], schedule=SCHED, plans=list(plans.values()))  # fmt: skip
    assert out == []


# --------------------------------------------------------------------------- plan repair


def repair(items: list[DisclosureItem], required: list[DisclosureItem] = ()):
    b = truth_bible()
    facts = {f.id: f for f in b.facts} | {"kai.f04": F("kai.f04", "bank_acct", "1", class_="never_store")}
    plans = [SessionPlan("kai.s05", "practical_guidance", "budget", items)]
    return g.repair_plans(plans, list(required), facts=facts, impressions={"kai.i01"},
                          references={r.id: r for r in b.references}, schedule=SCHED)  # fmt: skip


def it(item_id, mode="stated", session="kai.s05", forget=False) -> DisclosureItem:
    return DisclosureItem(session, item_id, mode, "practical_guidance", "", forget)


def test_repair_drops_unknown_ids_and_logs_it():
    (plan,), log = repair([it("kai.f03"), it("kai.f99")])
    assert [i.item_id for i in plan.items] == ["kai.f03"]
    assert any("kai.f99" in line for line in log)


def test_repair_drops_never_store_items():
    (plan,), log = repair([it("kai.f04", "hinted")])
    assert plan.items == [] and any("kai.f04" in line for line in log)


def test_repair_turns_stated_impressions_into_hinted():
    (plan,), log = repair([it("kai.i01", "stated")])
    assert [(i.item_id, i.mode) for i in plan.items] == [("kai.i01", "hinted")] and log


def test_repair_drops_a_fact_told_before_it_is_true():
    (plan,), log = repair([it("kai.f02")])  # s05 is 09-01; the new job starts 09-08
    assert plan.items == [] and any("kai.f02" in line for line in log)


def test_repair_drops_unrequired_mentions_of_forgotten_facts():
    (plan,), log = repair([it("kai.f05"), it("kai.f03", forget=True)])
    assert plan.items == [] and len(log) == 2


def test_repair_inserts_missing_required_items_once():
    req = [it("kai.f03", "hinted")]
    (plan,), log = repair([it("kai.r01")], req)
    assert sorted(i.item_id for i in plan.items) == ["kai.f03", "kai.r01"]
    (plan,), _ = repair([it("kai.f03", "hinted")], req)
    assert [i.item_id for i in plan.items] == ["kai.f03"]


# --------------------------------------------------------------------------- verification flags


PLAN = SessionPlan("kai.s05", "practical_guidance", "", [
    it("kai.f03"), it("kai.i01", "hinted"), it("kai.f05", forget=True),
])  # fmt: skip


def C(item_id, kind="fact", relation="consistent", text="x") -> Claim:
    return Claim(text, "kai", kind, item_id, relation)


def kinds(flags) -> list[tuple[str, str | None]]:
    return sorted((f.kind, f.item_id) for f in flags)


def test_flags_none_when_every_planned_item_is_present_and_consistent():
    claims = [C("kai.f03"), C("kai.i01", "trait_shown"), C("kai.f05", "forget_request"), C(None, relation="new")]
    assert g.verification_flags(PLAN, claims, {"kai.i01"}) == []


def test_flags_missing_planned_items():
    flags = g.verification_flags(PLAN, [C("kai.f05", "forget_request")], {"kai.i01"})
    assert kinds(flags) == [("missing", "kai.f03"), ("missing", "kai.i01")]


def test_flags_forget_request_needs_a_forget_claim():
    claims = [C("kai.f03"), C("kai.i01", "trait_shown"), C("kai.f05")]  # mentioned, never asked to forget
    assert kinds(g.verification_flags(PLAN, claims, {"kai.i01"})) == [("missing", "kai.f05")]


def test_flags_conflicting_claims_planned_or_not():
    claims = [C("kai.f03"), C("kai.i01", "trait_shown"), C("kai.f05", "forget_request"),
              C("kai.f01", relation="contradicts")]  # fmt: skip
    assert kinds(g.verification_flags(PLAN, claims, {"kai.i01"})) == [("conflict", "kai.f01")]


def test_flags_a_trait_stated_outright():
    claims = [C("kai.f03"), C("kai.i01", "trait_stated"), C("kai.f05", "forget_request")]
    assert kinds(g.verification_flags(PLAN, claims, {"kai.i01"})) == [("trait_stated", "kai.i01")]


# --------------------------------------------------------------------------- requests


def test_build_req_uses_low_effort_structured_output_and_caches_the_shared_prefix():
    req = g.build_req(g.STAGES["r1_world"], ["instructions", "shared context"], "go", World, meta={"principal": "-"})
    p = req.params
    assert p["model"] == "claude-opus-5-5"
    assert p["output_config"]["effort"] == "low"
    assert p["output_config"]["format"]["type"] == "json_schema"
    assert p["output_config"]["format"]["schema"]["properties"].keys() >= {"town", "places", "events"}
    assert p["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert not {"fallbacks", "tool_choice", "temperature", "top_p", "top_k", "thinking"} & set(p)
    assert req.meta["stage"] == "r1_world"


def test_stage_models_match_d25():
    """D25, as amended by D32(f): the recoverability check runs on Sonnet, not on the truth's own model (Opus)."""
    models = {name: s.model for name, s in g.STAGES.items()}
    assert {models[s] for s in ("r1_world", "r2_cast", "r3_bible")} == {"claude-opus-5-5"}
    assert {models[s] for s in ("r4_life_log", "r5_plan", "r6_render", "r8_recover", "r8_judge")} == {"claude-sonnet-5-5"}
    assert models["r7_verify"] == "claude-haiku-5-5"


# --------------------------------------------------------------------------- batch rounds


@pytest.fixture
def batch_llm(tmp_path, fake):
    from memobench.llm import LLM

    return LLM(root=str(tmp_path), run_id="t", client=fake, sleep=lambda s: None,
               allowed_models=tuple(g.MODELS.values()), budget_usd=15, approval_usd=15)  # fmt: skip


def job(name: str, user: str, stage="r7_verify") -> g.Job:
    req = g.build_req(g.STAGES[stage], ["sys"], user, g.JudgeOut, meta={"principal": "kai", "item": name})
    return g.Job(name, req, lambda text: json.loads(text)["recovered"])


def test_round_runs_through_batch_only(batch_llm, fake: FakeAnthropic):
    fake.reply("q1", text='{"recovered": true, "matched": "", "rationale": ""}')
    out = g.run_round(batch_llm, [job("a", "q1")], budget_usd=15, stats=g.Stats())
    assert out == {"a": True}
    assert fake.create_calls == [] and len(fake.batch_creates) == 1


def test_round_regenerates_a_refusal_once_and_counts_it(batch_llm, fake: FakeAnthropic):
    fake.reply("q1", text="", stop_reason="refusal")
    fake.reply(g.retry_user("q1", "refusal"), text='{"recovered": false, "matched": "", "rationale": ""}')
    stats = g.Stats()
    out = g.run_round(batch_llm, [job("a", "q1")], budget_usd=15, stats=stats)
    assert out == {"a": False}
    assert stats.refusals == 1 and stats.retries == 1
    assert [prompt_of(r["params"]) for b in fake.batch_creates for r in b] == ["q1", g.retry_user("q1", "refusal")]


def test_round_gives_up_after_one_retry(batch_llm, fake: FakeAnthropic):
    fake.reply("q1", text="not json")  # the retry gets the fake's default reply, also not JSON
    stats = g.Stats()
    out = g.run_round(batch_llm, [job("a", "q1")], budget_usd=15, stats=stats)
    assert out == {"a": None}
    assert stats.retries == 1 and stats.failures == ["a"] and len(fake.batch_creates) == 2


def test_round_halts_before_submitting_when_the_budget_cannot_cover_it(batch_llm, fake: FakeAnthropic):
    with pytest.raises(g.BudgetHalt):
        g.run_round(batch_llm, [job("a", "q1", stage="r3_bible")], budget_usd=0.01, stats=g.Stats())
    assert fake.batch_creates == []


def test_round_replays_from_cache_without_new_batches(batch_llm, fake: FakeAnthropic):
    fake.reply("q1", text='{"recovered": true, "matched": "", "rationale": ""}')
    g.run_round(batch_llm, [job("a", "q1")], budget_usd=15, stats=g.Stats())
    stats = g.Stats()
    assert g.run_round(batch_llm, [job("a", "q1")], budget_usd=15, stats=stats) == {"a": True}
    assert len(fake.batch_creates) == 1 and stats.rounds == []


def test_upper_bound_counts_the_output_schema_as_input():
    """Structured outputs bill the JSON schema as prompt tokens (smoke: a 3 KB schema cost ~1.9k input tokens)."""
    from village.schema import Bible as B

    small = g.build_req(g.STAGES["r6_render"], ["sys"], "go", g.JudgeOut, meta={})
    big = g.build_req(g.STAGES["r6_render"], ["sys"], "go", B, meta={})
    schema_chars = len(json.dumps(big.params["output_config"]))
    assert g.upper_bound_usd([big]) - g.upper_bound_usd([small]) >= 0.9 * (schema_chars / 1.6) * 2.50 / 1e6 * 0.5


def test_verify_request_carries_the_session_date():
    """Validity windows only decide 'contradicts' if the verifier knows when the conversation happened."""
    from village.schema import Cast, Person, Session, Turn

    truth = g.Truth("kai", {"kai": Person("kai", "Kai", "principal", 19, "", "", [])}, {}, {}, {})
    cast = Cast(list(truth.people.values()), [], [])
    sess = Session("kai.s07", "kai", "2025-09-01T05:20:00", [Turn("user", "hi"), Turn("assistant", "hello")])
    user = g.verify_job(truth, cast, sess, "kai.s07").req.params["messages"][-1]["content"]
    assert "2025-09-01" in user and "Monday" in user


# The pilot's conflict flags, audited against the bible: claims contradicting a decoy (false by design, so its
# counter-evidence must contradict it) and claims "contradicting" a fact that did not hold on the session's date.
FACTS = {
    "kai.f01": F("kai.f01", "job", "barista", valid_to="2025-09-08", superseded_by="kai.f02"),
    "kai.f02": F("kai.f02", "job", "shift lead", valid_from="2025-09-08"),
}
PLAN2 = SessionPlan("kai.s05", "practical_guidance", "", [it("kai.f03")])
PRESENT = [C("kai.f03")]


def flags2(claims, day="2025-09-10", decoys=()):
    return kinds(g.verification_flags(PLAN2, PRESENT + claims, {"kai.i01", "kai.i09"}, facts=FACTS,
                                      decoys=set(decoys), day=date.fromisoformat(day)))  # fmt: skip


def test_contradicting_a_decoy_is_not_a_conflict():
    assert flags2([C("kai.i09", "trait_shown", "contradicts")], decoys={"kai.i09"}) == []


def test_contradicting_an_ordinary_impression_is_a_conflict():
    assert flags2([C("kai.i01", "trait_shown", "contradicts")], decoys={"kai.i09"}) == [("conflict", "kai.i01")]


def test_contradicting_a_fact_that_no_longer_holds_is_not_a_conflict():
    assert flags2([C("kai.f01", relation="contradicts")], day="2025-09-10") == []


def test_contradicting_a_fact_not_yet_true_is_not_a_conflict():
    assert flags2([C("kai.f02", relation="contradicts")], day="2025-09-01") == []


def test_contradicting_a_fact_that_holds_that_day_is_a_conflict():
    assert flags2([C("kai.f02", relation="contradicts")], day="2025-09-08") == [("conflict", "kai.f02")]


def test_denying_a_trait_is_not_stating_it():
    assert flags2([C("kai.i09", "trait_stated", "contradicts")], decoys={"kai.i09"}) == []


# --------------------------------------------------------------------------- D32(b): overlap repair across weeks


def week(n, facts, refs=()):
    from village.schema import LifeLogEntry, LifeLogWeek

    day = (DAYS[0] + __import__("datetime").timedelta(days=7 * (n - 1))).isoformat()
    return LifeLogWeek("kai", n, [LifeLogEntry(f"kai.w{n}.l01", day, "x", [], "", "", list(refs), [])], facts)


def test_overlap_same_value_merges_into_the_earlier_fact_and_rewrites_entries():
    a = F("kai.w2.n04", "essay_status", "drafting", valid_from="2025-09-02", subject="priya")
    b = F("kai.w4.n03", "essay_status", " Drafting", valid_from="2025-09-16", subject="priya")
    logs, log = g.repair_fact_overlaps([week(2, [a]), week(4, [b], refs=["kai.w4.n03"])])
    assert [f.id for w in logs for f in w.new_facts] == ["kai.w2.n04"]
    assert logs[1].entries[0].facts == ["kai.w2.n04"]
    assert any("kai.w2.n04" in line and "kai.w4.n03" in line for line in log)


def test_overlap_merge_keeps_the_union_of_the_windows():
    a = F("kai.w2.n04", "essay_status", "drafting", valid_from="2025-09-02", valid_to="2025-09-10", subject="priya")
    b = F("kai.w3.n01", "essay_status", "drafting", valid_from="2025-09-05", subject="priya")
    logs, _ = g.repair_fact_overlaps([week(2, [a]), week(3, [b])])
    (kept,) = [f for w in logs for f in w.new_facts]
    assert (kept.valid_from, kept.valid_to) == ("2025-09-02", None)


def test_overlap_different_value_chains_the_earlier_into_the_later():
    a = F("kai.w2.n04", "essay_status", "drafting", valid_from="2025-09-02", subject="priya")
    b = F("kai.w4.n03", "essay_status", "submitted", valid_from="2025-09-16", subject="priya")
    logs, log = g.repair_fact_overlaps([week(2, [a]), week(4, [b])])
    first, second = [f for w in logs for f in w.new_facts]
    assert (first.valid_to, first.superseded_by) == ("2025-09-16", "kai.w4.n03")
    assert (second.valid_from, second.valid_to) == ("2025-09-16", None) and log


def test_overlap_chain_without_a_start_date_starts_the_later_fact_on_its_week():
    a = F("kai.w1.n01", "phone_case", "pink", subject="kai")
    b = F("kai.w3.n02", "phone_case", "clear", subject="kai")
    logs, _ = g.repair_fact_overlaps([week(1, [a]), week(3, [b])])
    first, second = [f for w in logs for f in w.new_facts]
    assert second.valid_from == "2025-09-08" and first.valid_to == "2025-09-08"


def test_overlap_repair_leaves_the_validator_clean():
    from village.schema import Cast, Person

    a = F("kai.w2.n04", "essay_status", "drafting", valid_from="2025-09-02", subject="priya")
    b = F("kai.w4.n03", "essay_status", "submitted", valid_from="2025-09-16", subject="priya")
    c = F("kai.w3.n01", "essay_status", "outlined", valid_from="2025-09-09", subject="priya")
    logs, _ = g.repair_fact_overlaps([week(2, [a]), week(3, [c]), week(4, [b])])
    cast = Cast([Person("kai", "Kai", "principal", 19, "", "", []), Person("priya", "Priya", "periphery", 19, "", "", [])], [], [])
    out = validate(horizon=(DAYS[0].isoformat(), DAYS[-1].isoformat()), cast=cast, bibles=[], life_logs=logs,
                   schedule=[], plans=[])  # fmt: skip
    assert "overlap" not in {v.rule for v in out}


def test_required_items_pin_useful_changes_from_life_logs():
    """A chained life-log fact is a change like any other: told in the first session on or after it happens."""
    a = F("kai.w2.n04", "essay_status", "drafting", valid_to="2025-09-08", superseded_by="kai.w3.n01", subject="kai")
    b = F("kai.w3.n01", "essay_status", "submitted", valid_from="2025-09-08", subject="kai")
    items = g.required_items(truth_bible(), SCHED, extra_facts=[a, b])
    assert [(d.item_id, d.session_id) for d in items if d.item_id == "kai.w3.n01"] == [("kai.w3.n01", "kai.s07")]


def test_overlap_repair_ignores_facts_that_do_not_overlap():
    a = F("kai.w1.n01", "essay_status", "drafting", valid_from="2025-08-26", valid_to="2025-09-01", subject="priya")
    b = F("kai.w3.n01", "essay_status", "submitted", valid_from="2025-09-09", subject="priya")
    c = F("kai.w3.n02", "phone_case", "clear", valid_from="2025-09-09", subject="kai")
    weeks = [week(1, [a]), week(3, [b, c])]
    logs, log = g.repair_fact_overlaps(weeks)
    assert logs == weeks and log == []


# --------------------------------------------------------------------------- D32(a): quote-confirmed presence


from village.schema import PlannedCheck, Session, Turn  # noqa: E402

TURNS = [
    Turn("user", "ok so  Kapoor said Thursday pop quizzes have NO makeups 😭"),
    Turn("assistant", "Got it. Since the quizzes have no makeups, let's plan ahead."),
    Turn("user", "also pls forget that i told u about my crush on ethan"),
    Turn("assistant", "Done."),
]
PLAN3 = SessionPlan("kai.s05", "practical_guidance", "", [it("cast.f33"), it("kai.i01", "hinted"),
                                                           it("kai.f05", forget=True)])  # fmt: skip


def presence(checks):
    return kinds(g.presence_flags(PLAN3, checks, TURNS))


def test_normalize_ignores_case_whitespace_and_typographic_quotes():
    assert g.normalize("Don’t  STOP\n“now”") == g.normalize("don't stop \"now\"")


def test_presence_confirmed_by_a_verbatim_user_quote():
    checks = [PlannedCheck("cast.f33", True, "kapoor said thursday pop quizzes have no makeups"),
              PlannedCheck("kai.i01", True, "ok so kapoor said"),
              PlannedCheck("kai.f05", True, "pls forget that i told u about my crush")]  # fmt: skip
    assert presence(checks) == []


def test_presence_with_a_quote_not_in_the_transcript_counts_as_absent():
    checks = [PlannedCheck("cast.f33", True, "Kapoor never allows makeup quizzes"),
              PlannedCheck("kai.i01", True, "ok so kapoor said"),
              PlannedCheck("kai.f05", True, "pls forget that i told u about my crush")]  # fmt: skip
    assert presence(checks) == [("missing", "cast.f33")]


def test_presence_quote_from_an_assistant_turn_does_not_count():
    checks = [PlannedCheck("cast.f33", True, "since the quizzes have no makeups"),
              PlannedCheck("kai.i01", True, "ok so kapoor said"),
              PlannedCheck("kai.f05", True, "pls forget that i told u about my crush")]  # fmt: skip
    assert presence(checks) == [("missing", "cast.f33")]


def test_presence_absent_or_unchecked_items_are_missing():
    checks = [PlannedCheck("cast.f33", False, ""), PlannedCheck("kai.f05", True, "pls forget that i told u")]
    assert presence(checks) == [("missing", "cast.f33"), ("missing", "kai.i01")]


def test_presence_empty_quote_never_confirms():
    checks = [PlannedCheck("cast.f33", True, ""), PlannedCheck("kai.i01", True, "  "),
              PlannedCheck("kai.f05", True, "pls forget that i told u about my crush")]  # fmt: skip
    assert presence(checks) == [("missing", "cast.f33"), ("missing", "kai.i01")]


def test_claims_count_only_when_their_quote_is_in_the_transcript():
    plan = SessionPlan("kai.s05", "practical_guidance", "", [])
    real = Claim("x", "kai", "fact", "kai.f02", "contradicts", quote="kapoor said thursday")
    invented = Claim("y", "kai", "fact", "kai.f02", "contradicts", quote="kapoor said friday")
    flags = g.verification_flags(plan, [real, invented], set(), facts=FACTS, day=date(2025, 9, 9), turns=TURNS)
    assert [(f.kind, f.detail) for f in flags] == [("conflict", "unplanned: x")]


def test_verifier_sees_only_facts_valid_on_the_session_date():
    from village.schema import Cast, Person

    truth = g.Truth("kai", {"kai": Person("kai", "Kai", "principal", 19, "", "", [])}, FACTS, {}, {})
    sess = Session("kai.s03", "kai", "2025-09-01T09:00:00", TURNS)
    req = g.verify_job(truth, Cast(list(truth.people.values()), [], []), sess, "kai.s03", plan=PLAN3).req
    text = json.dumps(req.params)
    assert "barista" in text and "shift lead" not in text  # kai.f02 starts 2025-09-08


# --------------------------------------------------------------------------- D32(d): grounded assistant turns


CONVO = [
    Turn("user", "can u help me plan sunday, my mom works 10 hour shifts at kestrel"),
    Turn("assistant", "Since your mom works 10-hour shifts, plan dinner for 6. Gloria's shifts end at 4:30, right?"),
    Turn("user", "yeah she gets off at 4:30"),
    Turn("assistant", "Then dinner at 6 works."),
]


def A(quote, support="", item_id=None) -> Claim:
    return Claim("x", "kai", "fact", item_id, "consistent", speaker="assistant", quote=quote, support_quote=support)


def ungrounded(claims):
    return [f.detail for f in g.ungrounded_flags(claims, CONVO) if f.kind == "ungrounded"]


def test_assistant_claim_grounded_in_an_earlier_user_turn():
    assert ungrounded([A("your mom works 10-hour shifts", support="my mom works 10 hour shifts at kestrel")]) == []


def test_assistant_claim_the_user_confirms_only_later_is_ungrounded():
    """The Gloria case (pilot-v1 ines.s07): the assistant knew her shift end before anyone said it."""
    flags = ungrounded([A("Gloria's shifts end at 4:30", support="she gets off at 4:30")])
    assert len(flags) == 1 and "4:30" in flags[0]


def test_assistant_claim_with_no_support_is_ungrounded():
    assert len(ungrounded([A("Then dinner at 6 works.", support="")])) == 1


def test_support_quote_must_really_be_in_a_user_turn():
    assert len(ungrounded([A("your mom works 10-hour shifts", support="my mom is a nurse")])) == 1


def test_assistant_claim_not_in_any_assistant_turn_is_ignored():
    assert ungrounded([A("your dad drives a forklift")]) == []


def test_user_claims_are_never_ungrounded():
    user = Claim("x", "kai", "fact", None, "new", speaker="user", quote="my mom works 10 hour shifts")
    assert ungrounded([user]) == []


# --------------------------------------------------------------------------- D32(c): never-store disclosures


from dataclasses import replace  # noqa: E402

from village.schema import SensitiveCheck  # noqa: E402


def ns_bible(n_facts=1):
    b = truth_bible()
    extra = [F(f"kai.f8{n}", f"id_number_{n}", f"00488219{n}7", class_="never_store") for n in range(n_facts)]
    return replace(b, facts=b.facts + extra)


def test_never_store_slots_use_bible_facts_first_then_mint():
    assert g.never_store_slots(ns_bible(1), weeks=4, per_month=2) == [(2, "kai.f80"), (4, None)]


def test_never_store_slots_cover_every_month_of_a_long_horizon():
    slots = g.never_store_slots(ns_bible(3), weeks=52, per_month=2)
    assert len(slots) == 26 and [fid for _, fid in slots[:4]] == ["kai.f80", "kai.f81", "kai.f82", None]
    assert {w for w, _ in slots} == {4 * m + k for m in range(13) for k in (2, 4)}


WEEK2 = [s for s in SCHED if s.week == 2]  # kai.s05 (09-01), kai.s06 (09-03)


def plan_json(items_by_session, minted=()):
    return json.dumps({"sessions": [{"session_id": sid, "task": "t", "items": items} for sid, items in items_by_session],
                       "never_store_facts": list(minted)})  # fmt: skip


def I(item_id, mode="stated"):
    return {"item_id": item_id, "mode": mode, "note": "", "forget_request": False}


def test_plan_must_carry_its_never_store_slot_exactly_once():
    text = plan_json([("kai.s05", [I("kai.f80")]), ("kai.s06", [I("kai.f80")])])
    with pytest.raises(ValueError, match="kai.f80"):
        g.parse_plan(text, WEEK2, "kai", 2, [(2, "kai.f80")])


def test_plan_places_a_never_store_slot_fact_the_planner_left_out():
    # vicente.w2 in heldout-v1: a weekly principal's one session; twice the planner wrote the slot's account number into
    # another item's note instead of planning the fact. Place it in the week's lightest session, as for minted facts.
    text = plan_json([("kai.s05", [I("kai.f03"), I("kai.f01")]), ("kai.s06", [I("kai.f01")])])
    plans, _ = g.parse_plan(text, WEEK2, "kai", 2, [(2, "kai.f80")])
    placed = [(p.session_id, i.mode, i.never_store) for p in plans for i in p.items if i.item_id == "kai.f80"]
    assert placed == [("kai.s06", "stated", True)]


def test_plan_mints_a_never_store_fact_for_a_mint_slot():
    minted = {"id": "ns01", "subject": "kai", "attribute": "bank_account", "value": "4471-0938-22", "valid_from": None,
              "valid_to": None, "superseded_by": None, "class": "trivia"}  # fmt: skip
    text = plan_json([("kai.s05", [I("ns01")]), ("kai.s06", [I("kai.f01")])], [minted])
    plans, facts = g.parse_plan(text, WEEK2, "kai", 2, [(2, None)])
    assert [(f.id, f.class_) for f in facts] == [("kai.w2.ns01", "never_store")]
    assert plans[0].items[0].item_id == "kai.w2.ns01"


def test_plan_without_a_mint_slot_may_not_mint():
    minted = {"id": "ns01", "subject": "kai", "attribute": "x", "value": "y", "valid_from": None, "valid_to": None,
              "superseded_by": None, "class": "never_store"}  # fmt: skip
    with pytest.raises(ValueError, match="mint"):
        g.parse_plan(plan_json([("kai.s05", [I("kai.f03")]), ("kai.s06", [])], [minted]), WEEK2, "kai", 2, [])


def test_plan_places_a_minted_never_store_fact_the_planner_left_out():
    # grace.w4 in full13: the planner minted the fact on its retry but planned it in no session; place it, don't halt
    minted = {"id": "ns01", "subject": "kai", "attribute": "bank_account", "value": "4471-0938-22", "valid_from": None,
              "valid_to": None, "superseded_by": None, "class": "never_store"}  # fmt: skip
    text = plan_json([("kai.s05", [I("kai.f03"), I("kai.f01")]), ("kai.s06", [I("kai.f01")])], [minted])
    plans, facts = g.parse_plan(text, WEEK2, "kai", 2, [(2, None)])
    placed = [(p.session_id, i.mode, i.never_store) for p in plans for i in p.items if i.item_id == "kai.w2.ns01"]
    assert [f.id for f in facts] == ["kai.w2.ns01"] and placed == [("kai.s06", "stated", True)]


def test_plan_still_rejects_a_minted_never_store_fact_planned_twice():
    minted = {"id": "ns01", "subject": "kai", "attribute": "bank_account", "value": "4471-0938-22", "valid_from": None,
              "valid_to": None, "superseded_by": None, "class": "never_store"}  # fmt: skip
    text = plan_json([("kai.s05", [I("ns01")]), ("kai.s06", [I("ns01")])], [minted])
    with pytest.raises(ValueError, match="exactly one session"):
        g.parse_plan(text, WEEK2, "kai", 2, [(2, None)])


def repair_ns(items, weeks):
    b = ns_bible(1)
    return g.repair_plans([SessionPlan("kai.s05", "practical_guidance", "", items)], [], facts={f.id: f for f in b.facts},
                          impressions={"kai.i01"}, references={}, schedule=SCHED, never_store_weeks=weeks)  # fmt: skip


def test_repair_keeps_and_marks_the_never_store_slot_as_stated():
    (plan,), log = repair_ns([it("kai.f80", "hinted")], {"kai.f80": 2})
    assert [(i.item_id, i.mode, i.never_store) for i in plan.items] == [("kai.f80", "stated", True)] and log


def test_repair_drops_never_store_outside_its_slot_week():
    (plan,), _ = repair_ns([it("kai.f80")], {"kai.f80": 3})
    assert plan.items == []


def test_repair_inserts_a_missing_never_store_slot_into_its_week():
    plans, log = g.repair_plans([SessionPlan("kai.s05", "practical_guidance", "", []),
                                 SessionPlan("kai.s06", "practical_guidance", "", [])], [],
                                facts={f.id: f for f in ns_bible(1).facts}, impressions=set(), references={},
                                schedule=SCHED, never_store_weeks={"kai.f80": 2})  # fmt: skip
    assert [(p.session_id, [i.item_id for i in p.items]) for p in plans] == [("kai.s05", ["kai.f80"]), ("kai.s06", [])]
    assert any("inserted never-store" in line for line in log)


LEAK_TURNS = [
    Turn("user", "fill this in: account 4471-0938-22, routing 1210-0035 pls"),
    Turn("assistant", "Here's the form with your account number field left for you to paste. Due 2025-09-15."),
    Turn("user", "also my student id is 0048821937"),
    Turn("assistant", "I've put student ID 0048821937 on the form."),
]
NS = [F("kai.f80", "bank_account", "credit union checking 4471-0938-22", class_="never_store"),
      F("kai.f81", "student_id", "CVC student ID 0048821937", class_="never_store")]


def test_leak_when_the_assistant_repeats_a_never_store_item():
    flags = g.leak_flags([SensitiveCheck("kai.f81", "my student id is 0048821937", "student ID 0048821937")], NS, LEAK_TURNS)
    assert [(f.kind, f.item_id) for f in flags] == [("leak", "kai.f81")]


def test_leak_found_by_its_digits_even_when_the_verifier_misses_it():
    flags = g.leak_flags([], NS, LEAK_TURNS)
    assert [(f.kind, f.item_id) for f in flags] == [("leak", "kai.f81")]


def test_no_leak_when_only_the_user_states_it_and_the_assistant_refers_generically():
    turns = LEAK_TURNS[:2]
    assert g.leak_flags([SensitiveCheck("kai.f80", "account 4471-0938-22", "")], NS, turns) == []


def test_an_unconfirmed_assistant_quote_is_not_a_leak():
    assert g.leak_flags([SensitiveCheck("kai.f80", "", "account 4471-0938-22")], NS, LEAK_TURNS[:2]) == []


# --------------------------------------------------------------------------- D32(e): unplanned claims become trivia


from village.schema import Verification  # noqa: E402

TRIVIA_TURNS = [
    Turn("user", "ok so i got a new phone case, its pink. also my student id is 0048821937 for the form"),
    Turn("assistant", "Pink is a fun choice. I'll leave the ID field for you."),
]
SESSION = ScheduledSession("kai.s05", "kai", "2025-09-01T09:00:00", 2, "practical_guidance")


def U(text, quote, relation="new", kind="fact", about="kai", speaker="user", item_id=None) -> Claim:
    return Claim(text, about, kind, item_id, relation, speaker=speaker, quote=quote)


def trivia(claims, sensitive=()):
    v = Verification("kai.s05", "m", 1, claims, [], True, sensitive=list(sensitive))
    return g.session_facts(SESSION, v, TRIVIA_TURNS, people={"kai", "priya"})


def test_unplanned_user_claim_becomes_a_dated_trivia_fact_with_its_source():
    (sf,) = trivia([U("got a new pink phone case", "i got a new phone case, its pink")])
    f = sf.fact
    assert (f.id, f.subject, f.class_, f.valid_from, f.value) == ("kai.s05.t01", "kai", "trivia", "2025-09-01",
                                                                  "got a new pink phone case")  # fmt: skip
    assert sf.source == "kai.s05" and sf.quote == "i got a new phone case, its pink"


def test_claims_that_match_or_contradict_the_record_are_not_trivia():
    assert trivia([U("a", "i got a new phone case", "consistent", item_id="kai.f03"),
                   U("b", "its pink", "contradicts", item_id="kai.f03")]) == []  # fmt: skip


def test_assistant_claims_unconfirmed_quotes_and_strangers_are_not_trivia():
    assert trivia([U("a", "Pink is a fun choice", speaker="assistant"), U("b", "i got a blue case"),
                   U("c", "i got a new phone case", about="stranger")]) == []  # fmt: skip


def test_trait_claims_are_not_trivia():
    assert trivia([U("a", "i got a new phone case", kind="trait_shown"), U("b", "its pink", kind="trait_stated")]) == []


def test_a_never_store_disclosure_never_becomes_trivia():
    sens = [SensitiveCheck("kai.f81", "my student id is 0048821937", "")]
    assert trivia([U("student id 0048821937", "my student id is 0048821937")], sens) == []


def test_trivia_facts_keep_the_validator_clean():
    from village.schema import Cast, Person

    facts = [sf.fact for sf in trivia([U("got a pink case", "i got a new phone case, its pink"),
                                       U("case is pink", "its pink")])]  # fmt: skip
    cast = Cast([Person("kai", "Kai", "principal", 19, "", "", [])], [], [])
    out = validate(horizon=(DAYS[0].isoformat(), DAYS[-1].isoformat()), cast=cast, bibles=[], life_logs=[],
                   schedule=[], plans=[], extra_facts=facts)  # fmt: skip
    assert len(facts) == 2 and out == []


# --------------------------------------------------------------------------- pilot-v2 findings: cast changes, minted duplicates


def test_cast_changes_are_pinned_to_one_linked_principal():
    from village.schema import Cast, Link, Person

    people = [Person(i, i, t, 30, "", "", []) for i, t in [("kai", "principal"), ("nadia", "principal"),
                                                            ("mom", "inner"), ("boss", "inner"), ("hermit", "periphery")]]
    old = F("cast.f01", "job", "nurse", valid_to="2025-09-08", superseded_by="cast.f02", subject="boss")
    new = F("cast.f02", "job", "manager", valid_from="2025-09-08", subject="boss")
    mom_old = F("cast.f03", "hours", "40", valid_to="2025-09-02", superseded_by="cast.f04", subject="mom")
    mom_new = F("cast.f04", "hours", "30", valid_from="2025-09-02", subject="mom")
    lone = F("cast.f05", "pet", "cat", valid_from="2025-09-03", subject="hermit")
    lone_old = F("cast.f06", "pet", "none", valid_to="2025-09-03", superseded_by="cast.f05", subject="hermit")
    links = [Link("mom", "kai", "mother"), Link("mom", "nadia", "mother"), Link("boss", "nadia", "manager")]
    cast = Cast(people, links, [old, new, mom_old, mom_new, lone, lone_old])
    owners = g.assign_cast_changes(cast, ["kai", "nadia"])
    assert {pid: sorted(f.id for f in fs) for pid, fs in owners.items()} == {
        "kai": ["cast.f03", "cast.f04", "cast.f05", "cast.f06"], "nadia": ["cast.f01", "cast.f02"]}  # fmt: skip


def test_a_minted_never_store_fact_that_repeats_an_existing_one_is_merged_into_it():
    existing = {"kai.f80": F("kai.f80", "student_id_number", "CVC student ID 0048821937", class_="never_store")}
    dup = F("kai.w4.ns01", "student_id_number", "CVC student ID 0048821937", class_="never_store")
    same_number = F("kai.w2.ns01", "cvc_id", "my id 0048821937", class_="never_store")
    fresh = F("kai.w3.ns01", "moms_status", "pending green card", class_="never_store", subject="kai")
    kept, rename, log = g.dedupe_minted([dup, same_number, fresh], existing)
    assert [f.id for f in kept] == ["kai.w3.ns01"]
    assert rename == {"kai.w4.ns01": "kai.f80", "kai.w2.ns01": "kai.f80"} and len(log) == 2


def test_repair_allows_a_never_store_fact_in_each_of_its_weeks():
    b = ns_bible(1)
    plans = [SessionPlan("kai.s05", "practical_guidance", "", [it("kai.f80")]),
             SessionPlan("kai.s07", "practical_guidance", "", [it("kai.f80", session="kai.s07")])]
    out, _ = g.repair_plans(plans, [], facts={f.id: f for f in b.facts}, impressions=set(), references={},
                            schedule=SCHED, never_store_weeks={"kai.f80": {2, 3}})  # fmt: skip
    assert [[(i.item_id, i.never_store) for i in p.items] for p in out] == [[("kai.f80", True)], [("kai.f80", True)]]


# --------------------------------------------------------------------------- scale: cache-friendly verify, linked planners


def test_undated_facts_sit_in_the_cached_prefix_and_dated_ones_are_filtered_per_session():
    from village.schema import Cast, Person

    facts = FACTS | {"kai.f03": F("kai.f03", "snack", "takis", class_="trivia")}
    truth = g.Truth("kai", {"kai": Person("kai", "Kai", "principal", 19, "", "", [])}, facts, {}, {})
    sess = Session("kai.s03", "kai", "2025-09-01T09:00:00", TURNS)
    p = g.verify_job(truth, Cast(list(truth.people.values()), [], []), sess, "kai.s03", plan=PLAN3).req.params
    system, user = p["system"][-1]["text"], p["messages"][-1]["content"]
    assert "takis" in system and "takis" not in user  # standing fact: cached once per principal
    assert "barista" in user and "barista" not in system  # dated fact, valid that day: filtered per session
    assert "shift lead" not in system + user  # dated fact not yet valid


def test_planner_context_lists_only_principals_linked_to_this_one():
    from village.schema import Cast, Link, Person

    people = [Person(i, i, "principal", 20, "", "", []) for i in ("a", "b", "c")]
    cast = Cast(people, [Link("a", "b", "siblings"), Link("c", "x", "friends")], [])
    assert g.linked_principals(cast, "a", ["a", "b", "c"]) == ["b"]
    assert g.linked_principals(cast, "c", ["a", "b", "c"]) == []


def test_calibrated_round_estimate_replaces_the_worst_case_bound(batch_llm, fake: FakeAnthropic):
    """T2 rounds are too big for the worst-case bound (every prompt token a cache write, every max_tokens used);
    a measured per-request cost by stage lets them run, and the layer's 1.5x overrun stop still guards it."""
    fake.reply("q1", text='{"recovered": true, "matched": "", "rationale": ""}')
    jobs = [job("a", "q1", stage="r3_bible")]
    with pytest.raises(g.BudgetHalt):
        g.run_round(batch_llm, jobs, budget_usd=0.05, stats=g.Stats())
    stats = g.Stats()
    out = g.run_round(batch_llm, jobs, budget_usd=0.05, stats=stats, calibration={"r3_bible": 0.01})
    assert out == {"a": True} and stats.rounds[0]["estimate_usd"] == 0.01


def test_calibration_falls_back_to_the_bound_for_unmeasured_stages(batch_llm, fake: FakeAnthropic):
    with pytest.raises(g.BudgetHalt):
        g.run_round(batch_llm, [job("a", "q1", stage="r3_bible")], budget_usd=0.05, stats=g.Stats(),
                    calibration={"r6_render": 0.001})  # fmt: skip


# --------------------------------------------------------------------------- deterministic name grounding (D32 d)


NAME_TURNS = [
    Turn("user", "my grandma takes her pills at 3. i work at marsh light fri"),
    Turn("assistant", "Set an alarm so Carmen takes her pills at 3, then head to Marsh Light Coffee. Miami trip later?"),
    Turn("user", "her name is carmen btw"),
    Turn("assistant", "Got it, Carmen at 3."),
]
NAMES = ["Carmen", "Mia", "Marsh Light Coffee", "Aguilar"]


def test_a_name_the_user_has_not_said_yet_is_ungrounded():
    flags = g.name_flags(NAME_TURNS, NAMES)
    assert [(f.kind, f.detail) for f in flags] == [("ungrounded", "assistant turn 2 uses the name 'Carmen' before the user said it")]


def test_a_place_counts_as_said_by_its_short_form_and_names_match_whole_words():
    flags = g.name_flags(NAME_TURNS[:2], NAMES)
    assert all("Marsh Light" not in f.detail and "Mia" not in f.detail for f in flags)


def test_names_the_user_said_first_are_grounded():
    turns = [Turn("user", "write it to Dr. Aguilar"), Turn("assistant", "Dear Dr. Aguilar,")]
    assert g.name_flags(turns, NAMES) == []


def test_village_names_are_cast_names_and_places():
    from village.schema import Cast, Person, Place

    cast = Cast([Person("ines", "Inés Aguilar", "principal", 27, "", "", []), Person("al", "Al Wu", "periphery", 30, "", "", [])], [], [])
    world = World("Alder Flats", "", "", [], [Place("pl-x", "Marsh Light Coffee", "cafe", "")], [])
    assert sorted(g.village_names(cast, world)) == ["Aguilar", "Inés", "Marsh Light Coffee"]


def test_judge_reasons_before_it_decides():
    """Pilot-v2: the judge wrote "the decoy was not generalized" and then recovered=false; the verdict must follow
    the rationale, so the schema asks for the rationale first."""
    order = g.json_schema(g.JudgeOut)["required"]
    assert order.index("rationale") < order.index("recovered")
