"""A tiny hand-built village on disk, in the generator's real file layout, for the scorer tests.

Principal `kai`, horizon 2025-09-01..2025-09-21, six sessions:

  s01 09-01 09:00  diet kai.f01 "eats everything" (stated); impression i01 episode 1; trivia f05
  s02 09-03 20:00  crush kai.f03 (forgotten class); reference r01 first use; roommate cast.f01
  s03 09-05 19:00  diet change kai.f02 "vegetarian" (supersedes f01 from 09-05); student id f04 (never_store);
                   i01 episode 2; decoy i02 episode
  s04 09-07 10:00  forget request for f03; dentist plan f06 (ends 09-08, no successor)
  s05 09-10 09:00  i01 episode 3; job f07 (verifier flagged it missing); decoy i02 again
  s06 09-20 09:00  reference r01 again
"""

from __future__ import annotations

import json
from pathlib import Path

from village.schema import (
    Bible, Cast, DisclosureItem, Episode, Fact, Impression, LifeLogWeek, Person, Reference, ScheduledSession,
    Session, SessionFact, SessionPlan, Turn, Usage, Voice, dump,
)  # fmt: skip

P = "kai"
SESSIONS = {
    "kai.s01": "2025-09-01T09:00:00",
    "kai.s02": "2025-09-03T20:00:00",
    "kai.s03": "2025-09-05T19:00:00",
    "kai.s04": "2025-09-07T10:00:00",
    "kai.s05": "2025-09-10T09:00:00",
    "kai.s06": "2025-09-20T09:00:00",
}


def fact(id, attribute, value, cls="useful", valid_from=None, valid_to=None, superseded_by=None, subject=P):
    return Fact(id, subject, attribute, value, valid_from, valid_to, superseded_by, cls)


FACTS = [
    fact("kai.f01", "diet", "eats everything, loves carnitas", valid_to="2025-09-05", superseded_by="kai.f02"),
    fact("kai.f02", "diet", "vegetarian", valid_from="2025-09-05"),
    fact("kai.f03", "crush", "has a crush on Robin from chemistry", cls="forgotten"),
    fact("kai.f04", "student_id", "student ID 00488219", cls="never_store"),
    fact("kai.f05", "favorite_song", "hums 'Golden Hour' all day", cls="trivia"),
    fact("kai.f06", "dentist", "dentist appointment on Sept 8 at 4 pm", valid_from="2025-09-01", valid_to="2025-09-08"),
    fact("kai.f07", "job", "barista at Marsh Light Coffee"),
]
CAST_FACTS = [fact("cast.f01", "job", "night-shift nurse", subject="mo"),
              fact("cast.f02", "study_buddy", "studies with Robin from chemistry on Sundays", cls="trivia", subject="mo")]
LIFE_LOG_FACTS = [fact("kai.w1.n01", "bike", "bike chain snapped", cls="trivia", valid_from="2025-09-02")]

IMPRESSIONS = [
    Impression("kai.i01", "Downplays her own achievements.", P,
               [Episode("kai.i01.e1", "2025-08-31", "calls her award 'luck'"),
                Episode("kai.i01.e2", "2025-09-05", "skips her promotion in a bio"),
                Episode("kai.i01.e3", "2025-09-09", "credits the team only")], None, None, "", False, False),
    Impression("kai.i02", "Is lazy about chores.", P,
               [Episode("kai.i02.e1", "2025-09-05", "skipped dishes while sick")], None, None, "", True, False),
]  # fmt: skip
REFERENCES = [Reference("kai.r01", "the landlord thing", "the dispute over the broken heater", "2025-09-03")]


def item(sid, iid, mode="stated", forget=False):
    return DisclosureItem(sid, iid, mode, "practical_guidance", f"note for {iid}", forget)


PLANS = {
    "kai.s01": [item("kai.s01", "kai.f01"), item("kai.s01", "kai.i01", "hinted"), item("kai.s01", "kai.f05")],
    "kai.s02": [item("kai.s02", "kai.f03"), item("kai.s02", "kai.r01"), item("kai.s02", "cast.f01"),
                item("kai.s02", "cast.f02")],
    "kai.s03": [item("kai.s03", "kai.f02"), item("kai.s03", "kai.f04"), item("kai.s03", "kai.i01", "hinted"),
                item("kai.s03", "kai.i02", "hinted")],
    "kai.s04": [item("kai.s04", "kai.f03", forget=True), item("kai.s04", "kai.f06")],
    "kai.s05": [item("kai.s05", "kai.i01", "hinted"), item("kai.s05", "kai.f07"), item("kai.s05", "kai.i02", "hinted")],
    "kai.s06": [item("kai.s06", "kai.r01")],
}  # fmt: skip
USER_TURNS = {
    "kai.s01": "ok so I eat everything, carnitas forever. can you plan my week?",
    "kai.s02": "update on the landlord thing: still no heat. also I have a crush on Robin from chemistry lol",
    "kai.s03": "I'm vegetarian now as of today. my student ID 00488219 for the form",
    "kai.s04": "please forget what I said about Robin. also dentist Sept 8 at 4",
    "kai.s05": "help me with a shift note for my barista job at Marsh Light",
    "kai.s06": "the landlord thing is finally fixed!",
}


# pilot-v2 format (D32 c, e): a planner-minted never-store fact the user pastes in s04 (plan item marked never_store),
# and an unplanned detail of s02 recorded as SessionFact trivia
V2_NEVER_STORE = fact("kai.w1.ns01", "bank_account_number", "checking account 6027 4415 83", cls="never_store")
V2_SESSION_FACT = SessionFact("kai.s02", "my heater is a 1998 wall unit",
                              fact("kai.s02.t01", "mentioned_s02_01", "Kai's heater is a 1998 wall unit", cls="trivia",
                                   valid_from="2025-09-03"))  # fmt: skip


def build(root: Path, *, flag_missing: tuple[str, str] | None = ("kai.s05", "kai.f07"),
          claims: dict[str, list[dict]] | None = None, v2: bool = False, unrendered: bool = False) -> Path:
    """Write the village into `root` and return it (`v2`: also the pilot-v2 files never_store_facts.json and
    session_facts.json, and the never-store marker on the plan item that discloses the minted fact)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({"horizon": ["2025-09-01", "2025-09-21"]}))
    voice = Voice("lowercase, quick bursts", "gives lots of context", ["says ok so"], ["ok so help"])
    bible = Bible(P, "summary not shown to the generator", FACTS, IMPRESSIONS, REFERENCES, [], voice,
                  Usage("daily", [9, 20]))  # fmt: skip
    dump(root / "bibles" / f"{P}.json", bible)
    people = [Person(P, "Kai Ramos", "principal", 19, "student", "s", []),
              Person("mo", "Mo Patel", "inner", 24, "roommate", "s", [])]  # fmt: skip
    dump(root / "cast.json", Cast(people, [], CAST_FACTS))
    dump(root / "life_logs" / f"{P}.w1.json", LifeLogWeek(P, 1, [], LIFE_LOG_FACTS))
    dump(root / "schedule.json", [ScheduledSession(s, P, t, 1, "practical_guidance") for s, t in SESSIONS.items()])
    plans = {s: list(items) for s, items in PLANS.items()}
    if v2:
        plans["kai.s04"].append(DisclosureItem("kai.s04", V2_NEVER_STORE.id, "stated", "practical_guidance",
                                               "pastes her account number into a form", never_store=True))  # fmt: skip
        dump(root / "never_store_facts.json", [V2_NEVER_STORE])
        dump(root / "session_facts.json", [V2_SESSION_FACT])
    dump(root / "plans.json", [SessionPlan(s, "practical_guidance", "task", items) for s, items in plans.items()])
    for sid, start in SESSIONS.items():
        dump(root / "sessions" / f"{sid}.json", Session(sid, P, start, [Turn("user", USER_TURNS[sid]),
                                                                          Turn("assistant", "Sure, here you go.")]))  # fmt: skip
        flags = []
        if flag_missing and flag_missing[0] == sid:
            flags = [{"kind": "missing", "item_id": flag_missing[1], "detail": "planned stated item never came up"}]
        (root / "verify").mkdir(exist_ok=True)
        (root / "verify" / f"{sid}.json").write_text(json.dumps(
            {"session_id": sid, "model": "m", "attempt": 1, "claims": (claims or {}).get(sid, []), "flags": flags,
             "passed": not flags}))  # fmt: skip
    (root / "sessions" / "attempt1").mkdir()  # generator subfolders hold alternates, never sessions
    if unrendered:  # full13's arjun.s67: planned and scheduled, but rendering failed twice, so no transcript exists
        plans = json.loads((root / "plans.json").read_text())
        plans.append({"session_id": "kai.s07", "task_type": "practical_guidance", "task": "t", "items": [
            {"session_id": "kai.s07", "item_id": i, "mode": "stated", "task_type": "practical_guidance", "note": "n",
             "forget_request": False, "never_store": False} for i in ("kai.f07", "kai.w1.n01")]})
        (root / "plans.json").write_text(json.dumps(plans))
        (root / "verify" / "kai.s07.json").write_text(json.dumps({"session_id": "kai.s07", "model": "m", "attempt": 2,
            "claims": [], "flags": [{"kind": "unparseable", "item_id": None, "detail": "render failed twice"}],
            "passed": False}))  # fmt: skip
    return root


# --------------------------------------------------------------------------- a longer village, for lag planning (D39)

LONG = "lee"
LONG_DAYS = 16  # one session a day, 09-01..09-16 at 09:00: at 09-16 23:59 the 10-session window is s07..s16


def build_long(root: Path) -> Path:
    """Principal `lee`. Old (before the window at 09-16): facts f01-f06 told in s01-s06, reference r01 (s02), impression
    i01 (episodes s01, s03), forgotten f11 (told s03, forget request s05), never-store f12 (s04). In the window: facts
    f07-f10 (s07-s10), r02 (s12), impression i02 (episodes s12, s14), facts f13-f16 (s13-s16)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({"horizon": ["2025-09-01", "2025-09-16"]}))
    sid = {d: f"{LONG}.s{d:02d}" for d in range(1, LONG_DAYS + 1)}
    starts = {sid[d]: f"2025-09-{d:02d}T09:00:00" for d in sid}
    facts = [Fact(f"{LONG}.f{i:02d}", LONG, f"attr{i:02d}", f"value number {i}", None, None, None, "useful")
             for i in [*range(1, 11), *range(13, 17)]]  # fmt: skip
    facts += [Fact(f"{LONG}.f11", LONG, "secret", "dates a coworker", None, None, None, "forgotten"),
              Fact(f"{LONG}.f12", LONG, "ssn", "social security 123-45-6789", None, None, None, "never_store")]
    imps = [Impression(f"{LONG}.i01", "Plans obsessively.", LONG, [Episode(f"{LONG}.i01.e1", "2025-08-31", "a"),
                       Episode(f"{LONG}.i01.e2", "2025-09-02", "b")], None, None, "", False, False),
            Impression(f"{LONG}.i02", "Avoids conflict.", LONG, [Episode(f"{LONG}.i02.e1", "2025-09-11", "a"),
                       Episode(f"{LONG}.i02.e2", "2025-09-13", "b")], None, None, "", False, False)]  # fmt: skip
    refs = [Reference(f"{LONG}.r01", "the move", "moving flats", "2025-09-02"),
            Reference(f"{LONG}.r02", "the audit", "the tax audit", "2025-09-12")]
    plan: dict[int, list] = {d: [] for d in sid}
    for i in [*range(1, 11), *range(13, 17)]:
        plan[i].append(f"{LONG}.f{i:02d}")
    plan[2].append(f"{LONG}.r01")
    plan[12] += [f"{LONG}.r02", f"{LONG}.i02"]
    plan[14].append(f"{LONG}.i02")
    plan[1].append(f"{LONG}.i01")
    plan[3] += [f"{LONG}.i01", f"{LONG}.f11"]
    plan[4].append(f"{LONG}.f12")
    items = {sid[d]: [item(sid[d], i, "hinted" if ".i" in i else "stated") for i in ids] for d, ids in plan.items()}
    items[sid[5]].append(item(sid[5], f"{LONG}.f11", forget=True))
    voice = Voice("plain", "direct", [], ["help me"])
    dump(root / "bibles" / f"{LONG}.json", Bible(LONG, "s", facts, imps, refs, [], voice, Usage("daily", [9])))
    dump(root / "cast.json", Cast([Person(LONG, "Lee Park", "principal", 30, "nurse", "s", [])], [], []))
    (root / "life_logs").mkdir()
    dump(root / "plans.json", [SessionPlan(s, "practical_guidance", "task", items[s]) for s in starts])
    for s, start in starts.items():
        dump(root / "sessions" / f"{s}.json", Session(s, LONG, start, [Turn("user", f"hi from {s}"),
                                                                         Turn("assistant", "ok")]))  # fmt: skip
    return root
