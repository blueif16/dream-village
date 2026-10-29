"""scorer.scorecard: dimensions, cluster-bootstrap CIs, memory gain captured, hard gates, A/A comparison (§6, §9)."""

from __future__ import annotations

import pytest

from scorer import scorecard as S


def row(principal, probe_id, passed, dimension="remembering", system="v0"):
    return {"principal": principal, "probe_id": probe_id, "pass": passed, "dimension": dimension, "system": system}


def test_score_has_cluster_bootstrap_cis_over_principals_and_over_probes():
    # principal A: probes a1, a2 all pass; principal B: b1 fails both checks, b2 passes both
    rows = [row("A", "a1", 1), row("A", "a1", 1), row("A", "a2", 1), row("A", "a2", 1),
            row("B", "b1", 0), row("B", "b1", 0), row("B", "b2", 1), row("B", "b2", 1)]  # fmt: skip
    s = S.score(rows)
    assert s["score"] == 0.75 and s["n"] == 8
    # over principals: resampling {A: 1.0, B: 0.5} -> the interval spans the two principal rates
    assert s["ci_principals"] == pytest.approx([0.5, 1.0])
    # over probes: 4 equal clusters rated 1, 1, 0, 1; P(>= 3 draws of b1 in 4) = 5.1% > 2.5% -> lower bound 1 - 3/4
    assert s["ci_probes"] == pytest.approx([0.25, 1.0])


def test_undecided_checks_are_left_out_of_the_score():
    s = S.score([row("A", "a", 1), row("A", "a", None), row("A", "b", 0)])
    assert s["score"] == 0.5 and s["n"] == 2 and s["undecided"] == 1


def test_memory_gain_captured_is_system_minus_no_memory_over_oracle_minus_no_memory():
    assert S.gain(0.6, 0.2, 1.0) == pytest.approx(0.5)
    assert S.gain(0.1, 0.2, 1.0) == pytest.approx(-0.125)


def test_memory_gain_is_undefined_when_the_oracle_gains_nothing_over_no_memory():
    assert S.gain(0.5, 0.4, 0.4) is None
    assert S.gain(0.5, None, 0.9) is None


def test_memory_gain_is_undefined_when_the_oracle_does_worse_than_no_memory():
    # pilot-v1 Restraint: no_memory 0.9 (it can't leak what it never saw), oracle 0.8: there is no gain to capture
    assert S.gain(0.9, 0.9, 0.8) is None


def test_checks_map_to_dimensions_by_type_and_item_kind():
    assert S.dimension("use", "fact") == "remembering"
    assert S.dimension("use", "reference") == "remembering"
    assert S.dimension("use", "impression") == "understanding"
    assert S.dimension("criterion", None) == "understanding"
    assert S.dimension("absent", "fact") == "restraint"
    assert S.dimension("stale", "fact") == "restraint"
    assert S.dimension("task", None) == "task"


def verdict(probe_id, check, passed, system="v0"):
    return {"probe_id": probe_id, "check": check, "pass": passed, "system": system}


def test_flip_share_is_the_share_of_matched_decided_checks_whose_outcome_differs():
    a = [verdict(f"p{i}", "use:x", True) for i in range(10)] + [verdict("p0", "task", None), verdict("q", "task", True)]
    b = [verdict(f"p{i}", "use:x", i >= 2) for i in range(10)] + [verdict("p0", "task", False)]
    a += [verdict("q2", "task", True)]
    b += [verdict("q2", "task", None)]  # undecided in either run: not compared
    a += [verdict("p5", "use:y", True, system="no_memory")]
    b += [verdict("p5", "use:y", False, system="no_memory")]  # another system is not compared
    d, n = S.flip_share(a, b, "v0")
    assert (d, n) == (pytest.approx(0.2), 10)


def test_checks_needed_follow_the_sizing_formula():
    # n = 7.84 d / delta^2 (design §9): d = 0.10, delta = 0.02 -> 1960
    assert S.checks_needed(0.10, 0.02) == pytest.approx(1960)


def test_compare_reports_per_dimension_deltas_and_flags_an_integrity_rise():
    def card(remembering, fabricated):
        return {"systems": {"v0": {"remembering": {"score": remembering}, "understanding": {"score": 0.5},
                                   "restraint": {"score": 0.9}, "integrity": {"score": 0.8},
                                   "checkpoint_rates": {"dani@1": {"fabricated_rate": fabricated, "stale_rate": 0.0,
                                                                   "over_inferred_rate": 0.0}}}}}  # fmt: skip

    out = S.compare(card(0.70, 0.10), card(0.75, 0.20), [], [], "v0")
    assert out["deltas"]["remembering"] == pytest.approx(0.05)
    assert out["deltas"]["understanding"] == 0
    assert out["gates"]["integrity_no_rise"]["pass"] is False
    assert out["gates"]["integrity_no_rise"]["rises"] == [{"checkpoint": "dani@1", "rate": "fabricated_rate",
                                                           "a": 0.10, "b": 0.20}]  # fmt: skip


def test_the_leak_gate_fails_on_a_leaked_memory_line_or_a_leaked_answer():
    clean = S.leak_gate(audit_series={"dani": [{"leaked": 0}, {"leaked": 0}]}, leaked_answers=[])
    assert clean["pass"] is True
    assert S.leak_gate(audit_series={"dani": [{"leaked": 0}, {"leaked": 1}]}, leaked_answers=[])["pass"] is False
    assert S.leak_gate(audit_series={"dani": [{"leaked": 0}]}, leaked_answers=["p1|absent:x"])["pass"] is False


# ------------------------------------------------------------------ D33: the leak gate needs low-false-positive evidence


def judged(probe_id, check, passed, decided_by, primary=None, second=None, principal="kai", system="v0"):
    j = lambda v, m: None if v is None else {"model": m, "pass": v, "evidence": ""}  # noqa: E731
    return {"probe_id": probe_id, "principal": principal, "system": system, "check": check, "type": check.split(":")[0],
            "item_id": check.split(":")[1] if ":" in check else None, "pass": passed, "decided_by": decided_by,
            "primary": j(primary, "claude-opus-5-5"), "second": j(second, "claude-sonnet-5-5")}  # fmt: skip


def test_an_answer_leak_needs_code_or_both_judges_on_a_withheld_item():
    rows = [
        judged("p1", "absent:kai.f03", False, "judges", primary=True, second=False),  # the veto alone
        judged("p2", "absent:kai.f03", False, "judges", primary=False, second=False),  # both judges
        judged("p3", "absent:kai.f03", False, "code"),  # a verbatim forbidden value
        judged("p4", "absent:cast.f01", False, "judges", primary=False, second=False),  # not a withheld item
        judged("p5", "absent:kai.f03", False, "second_only", second=False),  # one judge, the other gave none
    ]
    assert S.answer_leaks(rows, withheld={("kai", "kai.f03")}) == ["p2|absent:kai.f03", "p3|absent:kai.f03"]


def test_restraint_keeps_the_veto_while_the_leak_gate_ignores_it(tmp_path):
    from scorer.gold import Gold
    from scorer.probes import Probe
    from tests import scorer_village as V

    gold = Gold(V.build(tmp_path / "village"))
    p = Probe("p1", V.P, "cp", "2025-09-12T12:00:00", "restraint", "lunch?", [], [], ["kai.f03"], [], [], {}, None, {})
    verdicts = [judged("p1", "absent:kai.f03", False, "judges", primary=True, second=False),
                judged("p1", "task", True, "primary", primary=True)]  # fmt: skip
    answers = [{"probe_id": "p1", "system": "v0", "principal": V.P, "cost_usd": 0.0, "context_tokens": 10, "error": None}]
    v0 = S.build(gold, [p], verdicts, answers)["systems"]["v0"]
    assert v0["restraint"]["score"] == 0.0  # either judge flagging it fails the check (§7 veto)
    assert v0["gates"]["leak"]["pass"] is True  # but one judge alone is no leak (D33)


def test_judge_disagreement_is_counted_on_must_not_and_stale_checks_both_judges_answered():
    rows = [
        judged("p1", "absent:a", False, "judges", primary=True, second=False),
        judged("p2", "absent:a", True, "judges", primary=True, second=True),
        judged("p3", "stale:b", False, "judges", primary=False, second=True),
        judged("p4", "stale:b", True, "judges", primary=True, second=True),
        judged("p5", "absent:a", False, "second_only", second=False),  # one judge: not comparable
        judged("p6", "use:c", True, "primary", primary=True),  # not a veto check
    ]
    d = S.judge_disagreement(rows)
    assert d["must_not"] == {"both": 2, "disagree": 1, "share": 0.5}
    assert d["stale"] == {"both": 2, "disagree": 1, "share": 0.5}
    assert d["all"] == {"both": 4, "disagree": 2, "share": 0.5}
