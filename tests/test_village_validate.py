"""village.validate: the code-checkable ground-truth rules (design §3), on hand-built fixtures.

`clean()` is a small village that breaks no rule; every other test breaks exactly one thing and asserts the rule
that must catch it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from village.schema import (
    Bible, Cast, DisclosureItem, Episode, Fact, Impression, LifeLogEntry, LifeLogWeek, Link, Person, Reference,
    ScheduledSession, SessionPlan, Usage, Voice,
)  # fmt: skip
from village.validate import ProbeRef, validate

HORIZON = ("2025-08-25", "2025-09-21")


def F(id, attribute, value, valid_from=None, valid_to=None, superseded_by=None, class_="useful", subject="kai") -> Fact:
    return Fact(id, subject, attribute, value, valid_from, valid_to, superseded_by, class_)


def item(session_id, item_id, mode="stated", forget_request=False) -> DisclosureItem:
    return DisclosureItem(session_id, item_id, mode, "practical_guidance", "", forget_request)


@dataclass
class V:
    cast: Cast
    bibles: list[Bible]
    life_logs: list[LifeLogWeek]
    schedule: list[ScheduledSession]
    plans: list[SessionPlan]

    def rules(self, probes=()) -> set[str]:
        return {v.rule for v in self.violations(probes)}

    def violations(self, probes=()):
        return validate(horizon=HORIZON, cast=self.cast, bibles=self.bibles, life_logs=self.life_logs,
                        schedule=self.schedule, plans=self.plans, probes=list(probes))  # fmt: skip

    @property
    def bible(self) -> Bible:
        return self.bibles[0]

    def plan(self, session_id: str) -> SessionPlan:
        return next(p for p in self.plans if p.session_id == session_id)


def clean() -> V:
    people = [
        Person("kai", "Kai Reyes", "principal", 19, "CC student", "", []),
        Person("nadia", "Nadia Reyes", "principal", 26, "night nurse", "", []),
        Person("dani", "Dani Cho", "inner", 20, "roommate", "", []),
    ]
    cast = Cast(people, [Link("kai", "nadia", "siblings")], [F("cast.f01", "job", "line cook", subject="dani")])
    facts = [
        F("kai.f01", "job", "barista", valid_to="2025-09-08", superseded_by="kai.f02"),
        F("kai.f02", "job", "shift lead", valid_from="2025-09-08"),
        F("kai.f03", "favorite_snack", "takis", class_="trivia"),
        F("kai.f04", "bank_account_number", "0012345", class_="never_store"),
        F("kai.f05", "failed_driving_test", "twice", class_="forgotten"),
    ]
    imp = Impression(
        "kai.i01", "Downplays her own wins", "kai",
        [Episode(f"kai.i01.e{n}", d, "") for n, d in ((1, "2025-08-26"), (2, "2025-09-02"), (3, "2025-09-09"))],
        None, None, "", False, False,
    )  # fmt: skip
    bible = Bible(
        "kai", "", facts, [imp], [Reference("kai.r01", "the usual", "oat latte", "2025-08-26")], [],
        Voice("", "", [], []), Usage("daily", [8, 21]),
    )  # fmt: skip
    log = LifeLogWeek(
        "kai", 1,
        [LifeLogEntry("kai.w1.l01", "2025-08-26", "shift", ["dani"], "", "", ["kai.f01"], ["kai.i01"])],
        [F("kai.w1.f01", "bike", "used blue fixie", valid_from="2025-08-27", class_="trivia")],
    )  # fmt: skip
    schedule = [
        ScheduledSession("kai.s01", "kai", "2025-08-26T08:00:00", 1, "practical_guidance"),
        ScheduledSession("kai.s02", "kai", "2025-09-02T21:00:00", 2, "practical_guidance"),
        ScheduledSession("kai.s03", "kai", "2025-09-09T08:00:00", 3, "practical_guidance"),
        ScheduledSession("kai.s04", "kai", "2025-09-16T08:00:00", 4, "practical_guidance"),
    ]
    plans = [
        SessionPlan("kai.s01", "practical_guidance", "", [
            item("kai.s01", "kai.f01"), item("kai.s01", "kai.f05"), item("kai.s01", "kai.i01", "hinted"),
            item("kai.s01", "kai.r01"),
        ]),
        SessionPlan("kai.s02", "practical_guidance", "", [item("kai.s02", "kai.f05", forget_request=True)]),
        SessionPlan("kai.s03", "practical_guidance", "", [item("kai.s03", "kai.f02")]),
        SessionPlan("kai.s04", "practical_guidance", "", [item("kai.s04", "kai.w1.f01"), item("kai.s04", "cast.f01")]),
    ]  # fmt: skip
    return V(cast, [bible], [log], schedule, plans)


def test_clean_village_has_no_violations():
    assert clean().violations() == []


# --------------------------------------------------------------------------- validity windows


def test_overlapping_windows_for_same_subject_and_attribute():
    v = clean()
    v.bible.facts[1] = replace(v.bible.facts[1], valid_from="2025-09-01")  # starts before kai.f01 ends (09-08)
    assert v.rules() == {"overlap"}


def test_back_to_back_windows_do_not_overlap():
    """kai.f01 ends 09-08 and kai.f02 starts 09-08: [from, to) windows touch but do not overlap."""
    assert "overlap" not in clean().rules()


def test_back_to_back_windows_do_not_overlap_in_either_list_order():
    v = clean()
    v.bible.facts[0], v.bible.facts[1] = v.bible.facts[1], v.bible.facts[0]
    assert "overlap" not in v.rules()


def test_overlap_detected_across_cast_sheet_and_bible():
    v = clean()
    v.bible.facts.append(F("kai.f09", "job", "dishwasher", subject="dani", class_="trivia"))
    assert v.rules() == {"overlap"}


def test_same_attribute_on_different_subjects_is_not_an_overlap():
    v = clean()
    v.bible.facts.append(F("kai.f09", "favorite_snack", "chips", subject="dani", class_="trivia"))
    assert v.rules() == set()


# --------------------------------------------------------------------------- changes


def test_change_disclosed_before_it_happens():
    v = clean()
    v.plan("kai.s02").items.append(item("kai.s02", "kai.f02"))  # s02 is 09-02, the new job starts 09-08
    assert v.rules() == {"disclosed_before_valid"}


def test_change_disclosed_on_its_first_day_is_fine():
    v = clean()
    v.schedule[2] = replace(v.schedule[2], start="2025-09-08T06:00:00")
    assert v.rules() == set()


def test_useful_change_never_disclosed():
    v = clean()
    v.plan("kai.s03").items.clear()
    assert v.rules() == {"change_not_disclosed"}


# --------------------------------------------------------------------------- forget requests


def test_forgotten_fact_without_a_forget_request():
    v = clean()
    v.plan("kai.s02").items.clear()
    assert v.rules() == {"forget_order"}


def test_forget_request_before_the_fact_was_ever_disclosed():
    v = clean()
    v.plan("kai.s01").items = [i for i in v.plan("kai.s01").items if i.item_id != "kai.f05"]
    v.plan("kai.s03").items.append(item("kai.s03", "kai.f05"))  # told only after the forget request (s02)
    assert v.rules() == {"forget_order"}


def test_forget_request_in_the_same_session_as_the_disclosure():
    v = clean()
    v.plan("kai.s02").items.clear()
    v.plan("kai.s01").items.append(item("kai.s01", "kai.f05", forget_request=True))
    assert v.rules() == {"forget_order"}


def test_forgotten_fact_disclosed_again_after_the_forget_request():
    v = clean()
    v.plan("kai.s04").items.append(item("kai.s04", "kai.f05"))
    assert v.rules() == {"forget_order"}


def test_forget_request_for_a_fact_not_classed_forgotten():
    v = clean()
    v.plan("kai.s04").items.append(item("kai.s04", "kai.f03", forget_request=True))
    assert v.rules() == {"forget_order"}


def test_forget_request_after_a_probe_that_tests_it():
    probe = ProbeRef("p1", "2025-09-01T12:00:00", must_use=[], must_not=["kai.f05"])  # forget request is 09-02
    assert clean().rules([probe]) == {"forget_order"}


def test_forget_request_before_the_probe_that_tests_it():
    probe = ProbeRef("p1", "2025-09-10T12:00:00", must_use=[], must_not=["kai.f05"])
    assert clean().rules([probe]) == set()


# --------------------------------------------------------------------------- ids


def test_plan_item_with_unknown_item_id():
    v = clean()
    v.plan("kai.s04").items.append(item("kai.s04", "kai.f99"))
    assert v.rules() == {"unknown_id"}


def test_plan_item_with_unknown_session_id():
    v = clean()
    v.plan("kai.s04").items.append(item("kai.s99", "kai.f03"))
    assert v.rules() == {"unknown_id"}


def test_superseded_by_unknown_fact():
    v = clean()
    v.bible.facts[2] = replace(v.bible.facts[2], superseded_by="kai.f77")
    assert v.rules() == {"unknown_id"}


def test_fact_subject_must_be_a_cast_member():
    v = clean()
    v.bible.facts[2] = replace(v.bible.facts[2], subject="stranger")
    assert v.rules() == {"unknown_id"}


def test_duplicate_fact_ids():
    v = clean()
    v.life_logs[0].new_facts.append(F("kai.f03", "shoe_size", "8", class_="trivia"))
    assert v.rules() == {"unknown_id"}


# --------------------------------------------------------------------------- session timestamps


def test_session_before_the_horizon():
    v = clean()
    v.schedule[0] = replace(v.schedule[0], start="2025-08-24T23:59:00")
    assert v.rules() == {"session_time"}


def test_session_after_the_horizon():
    v = clean()
    v.schedule[3] = replace(v.schedule[3], start="2025-09-22T00:00:00")
    assert v.rules() == {"session_time"}


def test_session_on_the_last_horizon_day_is_inside():
    v = clean()
    v.schedule[3] = replace(v.schedule[3], start="2025-09-21T23:59:00")
    assert v.rules() == set()


def test_sessions_out_of_order():
    v = clean()
    v.schedule[1], v.schedule[2] = v.schedule[2], v.schedule[1]
    assert v.rules() == {"session_time"}


# --------------------------------------------------------------------------- never-store, impressions


def test_never_store_fact_planned_for_disclosure():
    v = clean()
    v.plan("kai.s04").items.append(item("kai.s04", "kai.f04", "hinted"))
    assert v.rules() == {"never_store_planned"}


def test_impression_stated_outright():
    v = clean()
    v.plan("kai.s01").items[2] = item("kai.s01", "kai.i01", "stated")
    assert v.rules() == {"impression_stated"}


# --------------------------------------------------------------------------- hook: needed fact before its probe


def test_needed_fact_disclosed_after_its_probe():
    probe = ProbeRef("p1", "2025-09-05T12:00:00", must_use=["kai.w1.f01"], must_not=[])  # told on 09-16
    assert clean().rules([probe]) == {"needed_after_probe"}


def test_needed_fact_never_disclosed_before_its_probe():
    probe = ProbeRef("p1", "2025-09-20T12:00:00", must_use=["kai.f03"], must_not=[])  # never told
    assert clean().rules([probe]) == {"needed_after_probe"}


def test_needed_fact_disclosed_before_its_probe():
    probe = ProbeRef("p1", "2025-09-20T12:00:00", must_use=["kai.w1.f01", "kai.i01"], must_not=[])
    assert clean().rules([probe]) == set()


def test_violation_names_the_offending_ids():
    v = clean()
    v.plan("kai.s04").items.append(item("kai.s04", "kai.f04"))
    (violation,) = v.violations()
    assert "kai.f04" in violation.ids and "kai.s04" in violation.ids


# --------------------------------------------------------------------------- D32(c): marked never-store disclosures


def test_never_store_marked_and_stated_is_a_planned_leak_test_not_a_violation():
    v = clean()
    v.plan("kai.s04").items.append(replace(item("kai.s04", "kai.f04"), never_store=True))
    assert v.rules() == set()


def test_never_store_marked_but_only_hinted_is_a_violation():
    v = clean()
    v.plan("kai.s04").items.append(replace(item("kai.s04", "kai.f04", "hinted"), never_store=True))
    assert v.rules() == {"never_store_planned"}


def test_never_store_marker_on_an_ordinary_fact_is_a_violation():
    v = clean()
    v.plan("kai.s04").items.append(replace(item("kai.s04", "kai.f03"), never_store=True))
    assert v.rules() == {"never_store_planned"}


def test_facts_minted_after_the_life_logs_are_known_to_the_validator():
    v = clean()
    minted = Fact("kai.w2.ns01", "kai", "bank_account", "4471", None, None, None, "never_store")
    v.plan("kai.s04").items.append(replace(item("kai.s04", "kai.w2.ns01"), never_store=True))
    assert "unknown_id" in v.rules()
    out = validate(horizon=HORIZON, cast=v.cast, bibles=v.bibles, life_logs=v.life_logs, schedule=v.schedule,
                   plans=v.plans, extra_facts=[minted])  # fmt: skip
    assert out == []
