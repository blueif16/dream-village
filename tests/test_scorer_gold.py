"""scorer.gold: the disclosure record, what the assistant could know at t, and the oracle memory (design §3)."""

from __future__ import annotations

import json

import pytest

from scorer.gold import Gold
from tests import scorer_village as V


@pytest.fixture
def gold(tmp_path):
    return Gold(V.build(tmp_path / "village"))


def known_ids(gold, t):
    return {k.entry.id for k in gold.known_at(V.P, t)}


def known(gold, t, iid):
    return next(k for k in gold.known_at(V.P, t) if k.entry.id == iid)


def test_disclosure_record_lists_each_session_with_date_and_mode(gold):
    e = gold.entry(V.P, "kai.i01")
    assert [(d.session_id, d.at, d.mode) for d in e.disclosures] == [
        ("kai.s01", "2025-09-01T09:00:00", "hinted"),
        ("kai.s03", "2025-09-05T19:00:00", "hinted"),
        ("kai.s05", "2025-09-10T09:00:00", "hinted"),
    ]


def test_forget_request_is_recorded_on_the_entry_not_as_a_disclosure(gold):
    e = gold.entry(V.P, "kai.f03")
    assert e.forget_at == "2025-09-07T10:00:00"
    assert [d.session_id for d in e.disclosures] == ["kai.s02"]


def test_an_item_is_known_only_after_the_session_that_discloses_it_starts(gold):
    assert "kai.r01" not in known_ids(gold, "2025-09-03T19:59:00")
    assert "kai.r01" not in known_ids(gold, "2025-09-03T20:00:00")  # the session at t is not yet in the past
    assert "kai.r01" in known_ids(gold, "2025-09-03T20:01:00")


def test_a_forgotten_fact_is_known_until_the_forget_request_then_withheld(gold):
    assert "kai.f03" in known_ids(gold, "2025-09-05T00:00:00")
    assert "kai.f03" not in known_ids(gold, "2025-09-07T10:30:00")
    assert "kai.f03" in {e.id for e in gold.withheld_at(V.P, "2025-09-07T10:30:00")}


def test_never_store_is_never_known_but_withheld_once_disclosed(gold):
    for t in ("2025-09-04T00:00:00", "2025-09-21T23:00:00"):
        assert "kai.f04" not in known_ids(gold, t)
    assert "kai.f04" not in {e.id for e in gold.withheld_at(V.P, "2025-09-04T00:00:00")}
    assert "kai.f04" in {e.id for e in gold.withheld_at(V.P, "2025-09-06T00:00:00")}


def test_supersession_makes_the_old_value_stale_and_links_it_as_previous(gold):
    t = "2025-09-06T12:00:00"
    assert known(gold, t, "kai.f01").status == "stale"
    new = known(gold, t, "kai.f02")
    assert new.status == "current"
    assert new.previous is not None and new.previous.id == "kai.f01"


def test_an_ended_value_whose_change_was_not_told_yet_is_not_stale(gold):
    # f01 ended on 09-05, but the change (f02) is told only that evening: at noon the assistant can't know
    assert known(gold, "2025-09-05T12:00:00", "kai.f01").status == "ended_untold"


def test_a_plan_whose_date_passed_is_stale_without_any_successor(gold):
    assert known(gold, "2025-09-07T12:00:00", "kai.f06").status == "current"
    assert known(gold, "2025-09-08T12:00:00", "kai.f06").status == "stale"


def test_an_impression_is_known_once_enough_episodes_were_shown(gold):
    assert "kai.i01" not in known_ids(gold, "2025-09-02T00:00:00")  # one episode
    k = known(gold, "2025-09-06T00:00:00", "kai.i01")  # two episodes
    assert k.episodes == 2


def test_a_decoy_impression_is_never_known(gold):
    assert "kai.i02" not in known_ids(gold, "2025-09-21T23:00:00")


def test_cast_facts_disclosed_in_the_principals_sessions_are_known(gold):
    assert "cast.f01" in known_ids(gold, "2025-09-04T00:00:00")


def test_a_disclosure_the_verifier_flagged_missing_is_marked(gold):
    k = known(gold, "2025-09-11T00:00:00", "kai.f07")
    assert [d.flagged_missing for d in k.disclosed] == [True]


@pytest.mark.parametrize(
    ("t", "bucket"),
    [
        ("2025-09-10T09:30:00", "same_session"),
        ("2025-09-10T22:00:00", "same_day"),
        ("2025-09-11T12:00:00", "after_consolidation"),
        ("2025-09-20T12:00:00", "out_of_window"),  # window=1: s06 is now the only recent session
    ],
)
def test_lag_bucket_from_the_latest_disclosure(gold, t, bucket):
    assert gold.lag(V.P, "kai.f07", t, window=1) == bucket


def test_lag_weeks_later(gold):
    assert gold.lag(V.P, "kai.f05", "2025-09-20T12:00:00", window=1) == "weeks_later"


def test_oracle_memory_holds_current_values_marks_previous_and_omits_withheld(gold):
    text = gold.oracle_memory(V.P, "2025-09-12T12:00:00")
    assert "vegetarian" in text
    assert "previously: eats everything" in text
    assert "the landlord thing" in text and "broken heater" in text
    assert "Downplays her own achievements" in text
    for absent in ("crush on Robin", "00488219", "lazy about chores"):
        assert absent not in text


def test_excerpt_quotes_the_disclosing_user_turn(gold):
    assert "vegetarian now" in gold.excerpt(V.P, "kai.f02", "2025-09-12T12:00:00")


# ------------------------------------------------------------------ observed disclosures (D32 (e) on pilot-v1 data)


def claim(text, kind="fact", item_id=None, relation="new", about="kai"):
    return {"text": text, "about": about, "kind": kind, "item_id": item_id, "relation": relation}


CLAIMS = {
    "kai.s01": [claim("works as a barista at Marsh Light", item_id="kai.f07", relation="consistent")],
    "kai.s02": [claim("eats only fish", item_id="kai.f01", relation="contradicts"),
                claim("downplays her work", kind="other", item_id="kai.i01", relation="consistent")],
    "kai.s05": [claim("has a cat named Miso"), claim("is tired", kind="other")],
}


@pytest.fixture
def observed(tmp_path):
    return Gold(V.build(tmp_path / "village", claims=CLAIMS), observed=True)


def test_by_default_only_planned_items_are_disclosures(tmp_path):
    g = Gold(V.build(tmp_path / "village", claims=CLAIMS))
    assert [d.session_id for d in g.entry(V.P, "kai.f07").disclosures] == ["kai.s05"]
    assert "kai.s05.u00" not in g.entries(V.P)


def test_observed_mode_adds_a_verifier_confirmed_mention_as_an_unplanned_disclosure(observed):
    ds = observed.entry(V.P, "kai.f07").disclosures
    assert [(d.session_id, d.mode) for d in ds] == [("kai.s01", "unplanned"), ("kai.s05", "stated")]
    assert [d.session_id for d in observed.entry(V.P, "kai.f01").disclosures] == ["kai.s01"]  # a contradiction isn't one


def test_observed_mode_never_adds_impression_evidence(observed):
    assert len(observed.entry(V.P, "kai.i01").disclosures) == 3  # the plan's three episodes, not the verifier's claim


def test_observed_mode_adds_new_claims_as_trivia_told_in_their_session(observed):
    e = observed.entry(V.P, "kai.s05.u00")
    assert (e.kind, e.cls, e.text, e.subject) == ("fact", "trivia", "has a cat named Miso", "kai")
    assert "kai.s05.u00" in known_ids(observed, "2025-09-11T00:00:00")
    assert "kai.s05.u00" not in known_ids(observed, "2025-09-09T00:00:00")
    assert observed.entry(V.P, "kai.s05.u01").text == "is tired"


# ------------------------------------------------------------------ the pilot-v2 files (D32 c, e)


@pytest.fixture
def v2(tmp_path):
    return Gold(V.build(tmp_path / "village", v2=True))


def test_a_planner_minted_never_store_fact_is_withheld_once_disclosed_and_never_known(v2):
    e = v2.entry(V.P, "kai.w1.ns01")
    assert (e.cls, e.text) == ("never_store", "checking account 6027 4415 83")
    assert [d.session_id for d in e.disclosures] == ["kai.s04"]
    assert "kai.w1.ns01" not in {x.id for x in v2.withheld_at(V.P, "2025-09-07T10:00:00")}  # s04 itself isn't past yet
    assert "kai.w1.ns01" in {x.id for x in v2.withheld_at(V.P, "2025-09-07T10:30:00")}
    assert "kai.w1.ns01" not in known_ids(v2, "2025-09-21T00:00:00")


def test_a_session_fact_is_trivia_told_in_its_source_session(v2):
    e = v2.entry(V.P, "kai.s02.t01")
    assert (e.cls, e.text) == ("trivia", "Kai's heater is a 1998 wall unit")
    assert [(d.session_id, d.mode) for d in e.disclosures] == [("kai.s02", "unplanned")]
    assert "kai.s02.t01" in known_ids(v2, "2025-09-04T00:00:00")
    assert "kai.s02.t01" not in known_ids(v2, "2025-09-03T19:00:00")


def test_a_village_without_the_v2_files_still_loads(gold):
    assert "kai.w1.ns01" not in gold.entries(V.P) and "kai.s02.t01" not in gold.entries(V.P)


def test_the_horizon_covers_every_session_even_if_the_manifest_says_less(tmp_path):
    # full13-v1's manifest kept the pilot horizon (to 2025-09-21) while its sessions run to 2025-11-23
    root = V.build(tmp_path / "village")
    (root / "manifest.json").write_text(json.dumps({"horizon": ["2025-09-01", "2025-09-10"]}))
    assert Gold(root).horizon == ("2025-09-01", "2025-09-20")


def test_the_planned_items_of_a_session_with_no_transcript_are_not_disclosed(tmp_path):
    # full13's arjun.s67 failed to render twice: what it planned was never said to the assistant
    g = Gold(V.build(tmp_path / "village", unrendered=True))
    assert [d.session_id for d in g.entry(V.P, "kai.f07").disclosures] == ["kai.s05"]
    assert g.entry(V.P, "kai.w1.n01").disclosures == ()
    assert "kai.w1.n01" not in known_ids(g, "2025-09-21T00:00:00")
