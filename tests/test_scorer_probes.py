"""scorer.probes: checkpoints by code, kind slots, and code validation of generated probes (design §5)."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from scorer.gold import Gold
from scorer.probes import (
    EVAL_VERSION, MIX, Checkpoint, Slot, checkpoints, generation_req, kind_slots, parse_generated, plan_slots,
    shortcut_flags, validate,
)  # fmt: skip
from tests import scorer_village as V

T = "2025-09-12T12:00:00"  # f01 stale (f02 told), f03 forgotten (request 09-07), f04 never_store told, f06 stale


@pytest.fixture
def gold(tmp_path):
    return Gold(V.build(tmp_path / "village"))


@pytest.fixture
def cp():
    return Checkpoint("kai@20250912T1200", V.P, T, "test")


def raw(kind="fact", **targets):
    base = {"slot": 0, "kind": kind, "plan": "p", "request": "what should I cook tonight?", "must_use": [],
            "may_use": [], "must_not": [], "stale": [], "criteria": [], "forbidden": []}  # fmt: skip
    return {**base, **targets}


def check(gold, cp, r):
    probe, log = validate(gold, cp, r, n=0)
    return probe, " | ".join(log)


# ------------------------------------------------------------------ validation: each invalid target type


def test_a_valid_fact_probe_passes_unchanged_and_lag_is_computed_by_code(gold, cp):
    probe, log = check(gold, cp, raw(must_use=["kai.f02"], stale=["kai.f01"]))
    assert probe.must_use == ["kai.f02"] and probe.stale == ["kai.f01"]
    assert log == ""
    assert probe.item_lags == {"kai.f02": gold.lag(V.P, "kai.f02", T), "kai.f01": gold.lag(V.P, "kai.f01", T)}
    assert probe.lag == gold.lag(V.P, "kai.f02", T)  # the probe's lag is that of its must_use items


def test_the_probe_lag_is_the_longest_among_its_must_use_items(gold):
    late = Checkpoint("kai@20250920T1000", V.P, "2025-09-20T10:00:00", "test")
    probe, _ = check(gold, late, raw(must_use=["kai.f02", "kai.r01"]))  # r01 re-told 09-20 09:00; f02 on 09-05
    assert gold.lag(V.P, "kai.r01", late.t) == "same_session"
    assert probe.lag == gold.lag(V.P, "kai.f02", late.t) != "same_session"


def test_unknown_ids_are_removed_and_logged(gold, cp):
    probe, log = check(gold, cp, raw(must_use=["kai.f02", "kai.f99"], may_use=["nope"]))
    assert probe.must_use == ["kai.f02"] and probe.may_use == []
    assert "kai.f99" in log and "nope" in log


def test_must_use_not_disclosed_before_t_is_removed_and_the_emptied_probe_dropped(gold):
    early = Checkpoint("kai@20250904T1200", V.P, "2025-09-04T12:00:00", "test")
    probe, log = check(gold, early, raw(must_use=["kai.f07"]))  # f07 is told on 09-10
    assert probe is None
    assert "kai.f07" in log and "dropped" in log


def test_must_use_on_a_stale_item_is_moved_to_stale(gold, cp):
    probe, log = check(gold, cp, raw(must_use=["kai.f02", "kai.f01"]))
    assert probe.must_use == ["kai.f02"] and probe.stale == ["kai.f01"]
    assert "kai.f01" in log


def test_must_use_on_trivia_is_moved_to_may_use(gold, cp):
    probe, log = check(gold, cp, raw(must_use=["kai.f02", "kai.f05"]))
    assert probe.must_use == ["kai.f02"] and probe.may_use == ["kai.f05"]


def test_withheld_items_appear_only_as_must_not(gold, cp):
    probe, log = check(gold, cp, raw(must_use=["kai.f02", "kai.f03"], may_use=["kai.f04"], stale=["kai.f03"]))
    assert probe.must_use == ["kai.f02"]
    assert set(probe.must_not) == {"kai.f03", "kai.f04"}
    assert probe.may_use == [] and probe.stale == []


def test_a_forgotten_item_is_not_targeted_before_its_forget_request(gold):
    before = Checkpoint("kai@20250905T1200", V.P, "2025-09-05T12:00:00", "test")
    probe, log = check(gold, before, raw("restraint", must_not=["kai.f03", "cast.f01"]))
    assert probe.must_not == ["cast.f01"]
    assert "kai.f03" in log


def test_stale_on_a_current_item_is_moved_to_may_use(gold, cp):
    probe, log = check(gold, cp, raw(must_use=["kai.f02"], stale=["kai.f07"]))
    assert probe.stale == [] and probe.may_use == ["kai.f07"]


def test_an_ended_but_untold_value_can_only_be_may_use(gold):
    noon = Checkpoint("kai@20250905T1200", V.P, "2025-09-05T12:00:00", "test")
    probe, log = check(gold, noon, raw(must_use=["kai.f01", "kai.f05"], stale=["kai.f01"]))
    assert probe is None  # f01 -> may_use, f05 trivia -> may_use: nothing left to use
    assert "ended_untold" in log


def test_an_impression_shown_only_once_is_not_must_use(gold):
    early = Checkpoint("kai@20250902T1200", V.P, "2025-09-02T12:00:00", "test")
    probe, log = check(gold, early, raw("impression", must_use=["kai.i01"], criteria=[]))
    assert probe is None
    assert "kai.i01" in log


def test_must_use_whose_only_disclosure_was_flagged_missing_is_demoted(gold, cp):
    probe, log = check(gold, cp, raw(must_use=["kai.f02", "kai.f07"]))
    assert probe.must_use == ["kai.f02"] and probe.may_use == ["kai.f07"]


def test_a_restraint_probe_keeps_no_must_use(gold, cp):
    probe, log = check(gold, cp, raw("restraint", must_use=["kai.f02"], must_not=["cast.f01"]))
    assert probe.must_use == [] and probe.may_use == ["kai.f02"] and probe.must_not == ["cast.f01"]


def test_a_restraint_probe_without_must_not_is_dropped(gold, cp):
    probe, _ = check(gold, cp, raw("restraint"))
    assert probe is None


def test_a_reference_probe_needs_a_reference_in_must_use(gold, cp):
    assert check(gold, cp, raw("reference", must_use=["kai.f02"]))[0] is None
    assert check(gold, cp, raw("reference", must_use=["kai.r01"]))[0].must_use == ["kai.r01"]


def test_criteria_are_kept_only_on_impression_probes(gold, cp):
    imp, _ = check(gold, cp, raw("impression", must_use=["kai.i01"], criteria=["names the award"]))
    fact, log = check(gold, cp, raw(must_use=["kai.f02"], criteria=["names the award"]))
    assert imp.criteria == ["names the award"] and fact.criteria == []
    assert "criteria" in log


def test_forbidden_strings_must_be_verbatim_from_a_must_not_item_and_absent_from_the_request(gold, cp):
    probe, log = check(gold, cp, raw("restraint", request="anything about Robin? jk, what's for lunch",
                                     must_not=["kai.f03", "kai.f04"],
                                     forbidden=[{"id": "kai.f03", "strings": ["Robin", "crush on Robin", "kisses"]},
                                                {"id": "cast.f01", "strings": ["nurse"]}]))  # fmt: skip
    assert probe.forbidden["kai.f03"] == ["crush on Robin"]  # "Robin" is in the request; "kisses" is not in the value
    assert "cast.f01" not in probe.forbidden  # not a must_not item
    assert probe.forbidden["kai.f04"] == ["00488219"]  # digit runs of never-store values are added by code


def test_forbidden_strings_must_be_distinctive_enough_for_code_to_decide_a_leak(gold, cp):
    # code fails an answer on a match without asking a judge, so a string must prove the leak: >= 3 words or a
    # 4+ digit run, and absent from every other item the assistant may legitimately know
    probe, log = check(gold, cp, raw("restraint", request="what's for lunch", must_not=["kai.f03", "kai.f04"],
                                     forbidden=[{"id": "kai.f03", "strings": ["crush", "Robin from", "a crush on Robin",
                                                                              "Robin from chemistry"]},
                                                {"id": "kai.f04", "strings": ["student ID", "ID 00488219"]}]))  # fmt: skip
    assert probe.forbidden["kai.f03"] == ["a crush on Robin"]  # "Robin from chemistry" is also in known cast.f02
    assert probe.forbidden["kai.f04"] == ["ID 00488219", "00488219"]
    assert "'crush'" in log and "'Robin from chemistry'" in log


def test_a_v2_minted_never_store_fact_is_a_must_not_candidate_with_its_digits_forbidden(tmp_path, cp):
    v2 = Gold(V.build(tmp_path / "v2", v2=True))
    assert "kai.w1.ns01" in json.dumps(generation_req(v2, cp, ["restraint"]).params)
    probe, _ = check(v2, cp, raw("restraint", must_use=["kai.w1.ns01"]))
    assert probe.must_not == ["kai.w1.ns01"] and probe.forbidden["kai.w1.ns01"] == ["6027 4415 83"]  # the whole number


def test_an_item_in_several_lists_keeps_the_strictest(gold, cp):
    probe, _ = check(gold, cp, raw(must_use=["kai.f02", "kai.f07"], may_use=["kai.f02"], must_not=["kai.f07"]))
    assert probe.must_use == ["kai.f02"] and probe.must_not == ["kai.f07"] and probe.may_use == []


# ------------------------------------------------------------------ shortcut filter


def outcome(check_id, passed):
    return {"check": check_id, "pass": passed}


def test_a_memory_needing_probe_that_passes_with_no_memory_is_flagged(gold, cp):
    p1, _ = check(gold, cp, raw(must_use=["kai.f02"]))
    p2, _ = check(gold, cp, raw("restraint", must_not=["cast.f01"]))
    p3, _ = check(gold, cp, raw("impression", must_use=["kai.i01"], criteria=["names the award"]))
    p1, p2, p3 = (p.__class__(**{**p.__dict__, "id": i}) for p, i in ((p1, "a"), (p2, "b"), (p3, "c")))
    no_memory = {
        "a": [outcome("use:kai.f02", True), outcome("task", True)],
        "b": [outcome("absent:cast.f01", True), outcome("task", True)],
        "c": [outcome("use:kai.i01", True), outcome("criterion:0", False)],
    }
    assert shortcut_flags([p1, p2, p3], no_memory) == {"a"}


# ------------------------------------------------------------------ checkpoints and slots


def test_checkpoints_cover_month_ends_and_a_lag_cluster(gold):
    cps = checkpoints(gold, V.P)
    times = [c.t for c in cps]
    assert times == sorted(times)
    assert "2025-09-21T23:59:00" in times  # the end of the last (partial) month = the horizon end
    reasons = Counter(c.reason for c in cps)
    assert reasons["month_end"] == 1
    assert {"same_session", "same_day", "after_consolidation"} <= set(reasons)
    anchor = next(c for c in cps if c.reason == "same_session")
    nxt = next(c for c in cps if c.reason == "after_consolidation")
    assert anchor.t < nxt.t and nxt.t[:10] > anchor.t[:10]


class _Quarter:
    """A 13-week horizon (full13, D36) with a session every other day: what `checkpoints` reads of a Gold."""

    horizon = ("2025-08-25", "2025-11-23")

    def sessions(self, principal):
        from datetime import date, timedelta

        from village.schema import Session

        days = [date(2025, 8, 25) + timedelta(days=2 * i) for i in range(46)]
        return [Session(f"q.s{i:02d}", "q", f"{d}T10:00:00", []) for i, d in enumerate(days)]

    def entries(self, principal):
        return {}


def test_a_13_week_horizon_gets_the_sep_oct_nov_month_ends_and_one_lag_cluster():
    cps = checkpoints(_Quarter(), "q")
    assert [c.t for c in cps if c.reason == "month_end"] == ["2025-09-30T23:59:00", "2025-10-31T23:59:00",
                                                            "2025-11-23T23:59:00"]  # fmt: skip
    # August 31 has six days of history: too early to be worth a checkpoint
    assert Counter(c.reason for c in cps) == {"month_end": 3, "mid_month": 2, "same_session": 1, "same_day": 1,
                                              "after_consolidation": 1}
    # eval v2 (D39): mid-October and mid-November checkpoints, late enough for items older than the recent window
    assert [c.t for c in cps if c.reason == "mid_month"] == ["2025-10-15T23:59:00", "2025-11-15T23:59:00"]


def test_max_checkpoints_keeps_the_horizon_end_first(gold):
    cps = checkpoints(gold, V.P, max_checkpoints=2)
    assert len(cps) == 2
    assert cps[-1].t == "2025-09-21T23:59:00"
    assert cps[0].reason == "same_session"


def test_kind_slots_follow_the_mix_overall():
    counts = Counter(k for slots in kind_slots(4, 5, lambda i, k: True) for k in slots)
    assert sum(counts.values()) == 20
    assert counts == {"impression": 8, "fact": 5, "reference": 3, "restraint": 4}
    assert MIX == {"impression": 0.40, "fact": 0.25, "reference": 0.15, "restraint": 0.20}


def test_kind_slots_give_leftover_slots_by_largest_remainder():
    # 5 slots: impression 2.0, fact 1.25, reference 0.75, restraint 1.0 -> the spare slot goes to reference
    assert Counter(kind_slots(1, 5, lambda i, k: True)[0]) == {"impression": 2, "fact": 1, "reference": 1, "restraint": 1}


def test_an_unavailable_kind_falls_back_to_fact_then_restraint():
    slots = kind_slots(1, 5, lambda i, k: k not in ("impression", "reference"))
    assert Counter(slots[0]) == {"fact": 4, "restraint": 1}


# ------------------------------------------------------------------ the generation request


def test_the_generator_sees_known_and_withheld_items_only(gold, cp):
    req = generation_req(gold, cp, ["fact", "restraint"])
    text = json.dumps(req.params)
    assert "kai.f02" in text and "kai.f03" in text and "kai.f04" in text  # known; forgotten + never-store withheld
    assert "kai.i02" not in text and "lazy about chores" not in text  # a decoy is never known
    assert "kai.f07" in text
    assert "summary not shown to the generator" not in text
    assert req.params["model"] == "claude-opus-5-5"
    assert "thinking" not in req.params and req.params["output_config"]["format"]["type"] == "json_schema"


def test_parse_generated_reads_the_structured_output():
    text = json.dumps({"probes": [raw(must_use=["kai.f02"])]})
    assert parse_generated(text)[0]["must_use"] == ["kai.f02"]


# ------------------------------------------------------------------ eval v2 (D39): code plans lag-stratified targets

LONG_T = "2025-09-16T23:59:00"
OLD = ("out_of_window", "weeks_later", "months_later")


@pytest.fixture
def long(tmp_path):
    return Gold(V.build_long(tmp_path / "long"))


def long_cp(t=LONG_T):
    return Checkpoint(f"lee@{t}", V.LONG, t, "month_end")


def test_the_eval_version_is_v2():
    assert EVAL_VERSION == "dv-eval-v2"


def test_code_plans_required_targets_that_meet_the_lag_quota(long):
    [slots], log = plan_slots(long, V.LONG, [long_cp()], per_checkpoint=10)
    asked = [s for s in slots if s.kind != "restraint"]
    lags = [long.lag(V.LONG, s.target, LONG_T) for s in asked]
    assert len(asked) == 8
    assert sum(lag in OLD for lag in lags) >= 4  # >= 50% disclosed before the 10-session window
    assert sum(lag == "after_consolidation" for lag in lags) == 2  # ~25% within it, after a night
    assert sum(lag in ("same_session", "same_day") for lag in lags) <= 1  # <= 15%
    assert all(long.entry(V.LONG, s.target).kind == s.kind for s in asked)
    assert {s.target for s in slots if s.kind == "restraint"} == {"lee.f11", "lee.f12"}  # withheld, and old
    assert len({s.target for s in slots}) == len(slots)


def test_too_few_old_items_are_taken_as_they_are_and_the_shortfall_is_logged(long):
    early = "2025-09-08T23:59:00"  # every session is still inside the 10-session window
    [slots], log = plan_slots(long, V.LONG, [long_cp(early)], per_checkpoint=5)
    assert all(s.target for s in slots)
    assert not any(long.lag(V.LONG, s.target, early) in OLD for s in slots)
    assert any("shortfall" in x for x in log)


def test_the_generation_request_names_each_slot_s_required_target(long):
    slots = [Slot("fact", "lee.f01"), Slot("restraint", "lee.f12")]
    text = json.dumps(generation_req(long, long_cp(), slots).params)
    assert "slot 0: fact; required target: lee.f01" in text and "slot 1: restraint; required target: lee.f12" in text


def test_a_probe_whose_must_use_drops_its_required_target_is_rejected(long):
    probe, log = validate(long, long_cp(), raw(must_use=["lee.f02"]), n=0, required="lee.f01")
    assert probe is None and "required target lee.f01" in " ".join(log)


def test_extra_must_use_items_stay_only_if_they_are_old_too(long):
    probe, log = validate(long, long_cp(), raw(must_use=["lee.f01", "lee.f02", "lee.f16"]), n=0, required="lee.f01")
    assert probe.must_use == ["lee.f01", "lee.f02"] and probe.may_use == ["lee.f16"]  # f16 is same_day
    assert probe.required == "lee.f01"


def test_a_restraint_probe_must_keep_its_required_target_out(long):
    probe, _ = validate(long, long_cp(), raw("restraint", must_not=["lee.f11"]), n=0, required="lee.f12")
    assert probe is None
    kept, _ = validate(long, long_cp(), raw("restraint", must_not=["lee.f12"]), n=0, required="lee.f12")
    assert kept.must_not == ["lee.f12"]
