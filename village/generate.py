"""Dream Village generator: the world planned top-down, then rendered (design §2-§4; D12-D17, D25-D27).

One Batch round per stage; a request depends only on earlier rounds' outputs, never on a sibling's (D26):
  R1 world (Opus) -> R2 cast (Opus) -> R3 bibles (Opus, per principal) -> [code: session schedule]
  -> R4 life logs (Sonnet, per principal-week) -> [code: required disclosures]
  -> R5 disclosure plans (Sonnet, per principal-week) -> [code: plan repair, validator]
  -> R6 render (Sonnet per session, plus Haiku copies of a few for the A/B) -> R7 verify (Haiku)
  -> R6' re-render flagged sessions once -> R7' re-verify them + R8a blind trait inference (Opus, one round)
  -> R8b recoverability judge (Opus).
A refused, truncated, failed or unparseable response is regenerated once, as a second batch inside its round.
Prompts carry no timestamps or random ids, so a rerun replays every response from the response cache for free.

  python -m village.generate run   [--out data/villages/pilot-v1] [--budget 15]
  python -m village.generate smoke [--root runs/village-smoke] [--budget 2]   # real-time Opus/Sonnet only
"""

from __future__ import annotations

import argparse
import calendar
import copy
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from string import Template
from typing import Any, Callable

from memobench.llm import LLM, Req, Result
from memobench.pricing import cost_usd
from village.schema import (
    Bible, Cast, Claim, DisclosureItem, Fact, Flag, Impression, Inferred, LifeLogEntry, LifeLogWeek, Person,
    PlannedCheck, Recoverability, Reference, ScheduledSession, SensitiveCheck, Session, SessionFact, SessionPlan, Turn,
    Usage,
    Verification, World, dump, from_dict, json_schema, to_dict,
)  # fmt: skip
from village.spec import PILOT, VILLAGES, BibleBar, VillageSpec
from village.validate import _overlaps, validate

MODELS = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-5-5"}
EFFORT = "low"
HORIZON_START, WEEKS = PILOT.start, PILOT.weeks  # defaults: the pilot horizon, from the first school week
POLL_S = 30
PROMPTS = Path(__file__).parent / "prompts"
EST_CHARS_PER_TOKEN = 2.5  # deliberately low: the per-round estimate is an upper bound
SCHEMA_CHARS_PER_TOKEN, SCHEMA_PREAMBLE_TOKENS = 1.25, 400

# Session task mix (OpenAI usage study, NBER w34255): writing's 24% is two-thirds editing the user's own text.
TASK_WEIGHTS = {
    "practical_guidance": 29, "seeking_information": 24, "writing_edit": 16, "writing_new": 8, "tutoring": 10,
    "technical_help": 5, "multimedia": 7, "personal_reflection": 2,
}  # fmt: skip
SESSIONS_PER_WEEK = {"daily": (6, 7), "several_per_week": (3, 4), "weekly": (1, 1)}

class BudgetHalt(Exception):
    """A round's upper-bound cost would take the run past its budget; nothing was submitted."""


@dataclass(frozen=True)
class Stage:
    name: str
    model: str
    max_tokens: int


STAGES = {
    s.name: s
    for s in (
        Stage("r1_world", MODELS["opus"], 16000),
        Stage("r2_cast", MODELS["opus"], 24000),
        Stage("r3_bible", MODELS["opus"], 32000),
        Stage("r4_life_log", MODELS["sonnet"], 16000),
        Stage("r5_plan", MODELS["sonnet"], 12000),
        Stage("r6_render", MODELS["sonnet"], 10000),
        Stage("r7_verify", MODELS["haiku"], 8000),
        Stage("r8_recover", MODELS["sonnet"], 8000),  # D32(f): not the truth generator's model
        Stage("r8_judge", MODELS["sonnet"], 4000),
    )
}


# --------------------------------------------------------------------------- LLM output shapes (root must be an object)


@dataclass
class LifeLogOut:
    entries: list[LifeLogEntry]
    new_facts: list[Fact]


@dataclass
class PlanOut:
    sessions: list[SessionPlan]
    never_store_facts: list[Fact]  # D32(c): never-store facts the planner creates for this week's mint slots


@dataclass
class RenderOut:
    turns: list[Turn]


@dataclass
class VerifyOut:
    checks: list[PlannedCheck]
    sensitive: list[SensitiveCheck]
    claims: list[Claim]


@dataclass
class RecoverOut:
    inferred: list[Inferred]


@dataclass
class JudgeOut:
    rationale: str  # first: the verdict follows the reasoning
    matched: str
    recovered: bool


# --------------------------------------------------------------------------- calendar + schedule


def horizon_days(start: date = HORIZON_START, weeks: int = WEEKS) -> list[date]:
    return [start + timedelta(days=n) for n in range(7 * weeks)]


def _nth(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th `weekday` of a month (n = -1: the last one)."""
    days = [d for d in calendar.Calendar().itermonthdates(year, month) if d.month == month and d.weekday() == weekday]
    return days[n if n < 0 else n - 1]


FEDERAL_HOLIDAYS = (
    ("New Year's Day", lambda y: date(y, 1, 1)),
    ("Martin Luther King Jr. Day", lambda y: _nth(y, 1, calendar.MONDAY, 3)),
    ("Presidents' Day", lambda y: _nth(y, 2, calendar.MONDAY, 3)),
    ("Memorial Day", lambda y: _nth(y, 5, calendar.MONDAY, -1)),
    ("Juneteenth", lambda y: date(y, 6, 19)),
    ("Independence Day", lambda y: date(y, 7, 4)),
    ("Labor Day", lambda y: _nth(y, 9, calendar.MONDAY, 1)),
    ("Columbus Day / Indigenous Peoples' Day", lambda y: _nth(y, 10, calendar.MONDAY, 2)),
    ("Veterans Day", lambda y: date(y, 11, 11)),
    ("Thanksgiving Day", lambda y: _nth(y, 11, calendar.THURSDAY, 4)),
    ("Christmas Day", lambda y: date(y, 12, 25)),
)


def holidays(days: list[date]) -> dict[date, str]:
    """US federal holidays inside the horizon, on their calendar dates."""
    inside = set(days)
    return {d: f"{name} (US federal holiday)" for y in sorted({d.year for d in days})
            for name, rule in FEDERAL_HOLIDAYS if (d := rule(y)) in inside}  # fmt: skip


def calendar_text(days: list[date]) -> str:
    marks = holidays(days)
    return "\n".join(f"{d.isoformat()} {d:%A}" + (f" - {marks[d]}" if d in marks else "") for d in days)


def _seed(*parts: str) -> int:
    return int(hashlib.sha256(":".join(parts).encode()).hexdigest()[:16], 16)


def task_counts(n: int) -> dict[str, int]:
    """Sessions per task type for n sessions: the usage mix by largest remainder (ties go to the heavier type)."""
    total = sum(TASK_WEIGHTS.values())
    quotas = {k: n * w / total for k, w in TASK_WEIGHTS.items()}
    counts = {k: math.floor(q) for k, q in quotas.items()}
    order = sorted(TASK_WEIGHTS, key=lambda k: (-(quotas[k] - counts[k]), list(TASK_WEIGHTS).index(k)))
    for k in order[: n - sum(counts.values())]:
        counts[k] += 1
    return counts


def make_schedule(principal: str, usage: Usage, days: list[date]) -> list[ScheduledSession]:
    """Deterministic sessions from the usage frequency: distinct days per week, a time in an active hour, and task
    types dealt from the usage mix."""
    rng = random.Random(_seed("schedule", principal))
    hours = sorted({h for h in usage.active_hours if 0 <= h <= 23}) or [9, 20]
    lo, hi = SESSIONS_PER_WEEK[usage.frequency]
    starts = []
    for w in range(len(days) // 7):
        for d in sorted(rng.sample(days[7 * w : 7 * w + 7], rng.randint(lo, hi))):
            starts.append((w + 1, datetime(d.year, d.month, d.day, rng.choice(hours), rng.randrange(0, 60))))
    types = [k for k, c in task_counts(len(starts)).items() for _ in range(c)]
    rng.shuffle(types)
    return [ScheduledSession(f"{principal}.s{n:02d}", principal, t.isoformat(), w, tt)
            for n, ((w, t), tt) in enumerate(zip(starts, types), 1)]  # fmt: skip


# --------------------------------------------------------------------------- id scoping


def _scoped(prefix: str, local: str | None) -> str | None:
    """A local id ("f01") gets the prefix; a global one ("cast.f03", "kai.f02") is left alone."""
    if local is None:
        return None
    return local if "." in local else f"{prefix}.{local}"


def scope_bible(bible: Bible, pid: str) -> Bible:
    def imp(i: Impression) -> Impression:
        iid = _scoped(pid, i.id)
        return replace(i, id=iid, evidence=[replace(e, id=_scoped(iid, e.id)) for e in i.evidence])

    return replace(
        bible,
        principal=pid,
        facts=[replace(f, id=_scoped(pid, f.id), superseded_by=_scoped(pid, f.superseded_by)) for f in bible.facts],
        impressions=[imp(i) for i in bible.impressions],
        references=[replace(r, id=_scoped(pid, r.id)) for r in bible.references],
        arcs=[replace(a, id=_scoped(pid, a.id),
                      beats=[replace(b, facts=[_scoped(pid, x) for x in b.facts]) for b in a.beats])
              for a in bible.arcs],
    )  # fmt: skip


def scope_cast(cast: Cast) -> Cast:
    return replace(cast, facts=[replace(f, id=_scoped("cast", f.id), superseded_by=_scoped("cast", f.superseded_by))
                                for f in cast.facts])  # fmt: skip


def repair_fact_overlaps(logs: list[LifeLogWeek], start: date = HORIZON_START) -> tuple[list[LifeLogWeek], list[str]]:
    """D32(b): life-log weeks are generated independently (D26), so two weeks can mint the same subject+attribute
    with overlapping windows. The same value merges into the earlier fact (union window; references rewritten); a
    different value chains them (the earlier ends when the later starts, superseded_by the later; a later fact with
    no start date starts on its week's first day). Every repair is logged."""
    week_of = {f.id: w.week for w in logs for f in w.new_facts}
    first_day = lambda fid: (start + timedelta(days=7 * (week_of[fid] - 1))).isoformat()  # noqa: E731
    groups: dict[tuple[str, str], list[Fact]] = defaultdict(list)
    for w in logs:
        for f in w.new_facts:
            groups[(f.subject, f.attribute)].append(f)
    final: dict[str, Fact] = {f.id: f for w in logs for f in w.new_facts}
    rename: dict[str, str] = {}
    log: list[str] = []
    for (subject, attribute), group in groups.items():
        ordered = sorted(group, key=lambda f: (f.valid_from or first_day(f.id), week_of[f.id], f.id))
        cur = ordered[0]
        for nxt in ordered[1:]:
            if not _overlaps(cur, nxt):
                cur = nxt
                continue
            if " ".join(cur.value.casefold().split()) == " ".join(nxt.value.casefold().split()):
                cur = replace(cur, valid_from=min(cur.valid_from, nxt.valid_from) if cur.valid_from and nxt.valid_from
                              else None, valid_to=max(cur.valid_to, nxt.valid_to) if cur.valid_to and nxt.valid_to
                              else None)  # fmt: skip
                final[cur.id] = cur
                del final[nxt.id]
                rename[nxt.id] = cur.id
                log.append(f"merged {nxt.id} into {cur.id}: both hold {subject}.{attribute} = {cur.value!r}")
            else:
                begins = nxt.valid_from or first_day(nxt.id)
                nxt = final[nxt.id] = replace(nxt, valid_from=begins)
                final[cur.id] = replace(cur, valid_to=begins, superseded_by=nxt.id)
                log.append(f"chained {cur.id} -> {nxt.id} on {begins}: {subject}.{attribute} {cur.value!r} -> {nxt.value!r}")
                cur = nxt
    if not log:
        return logs, []
    out = []
    for w in logs:
        facts = [replace(final[f.id], superseded_by=rename.get(final[f.id].superseded_by, final[f.id].superseded_by))
                 for f in w.new_facts if f.id in final]  # fmt: skip
        entries = [replace(e, facts=list(dict.fromkeys(rename.get(x, x) for x in e.facts))) for e in w.entries]
        out.append(replace(w, entries=entries, new_facts=facts))
    return out, log


# --------------------------------------------------------------------------- required disclosures + plan repair


def _day(s: ScheduledSession) -> date:
    return datetime.fromisoformat(s.start).date()


def _from_day(schedule: list[ScheduledSession], day: date) -> list[ScheduledSession]:
    return [s for s in schedule if _day(s) >= day]


def _after_episode(schedule: list[ScheduledSession], day: date) -> list[ScheduledSession]:
    """Sessions that can tell an episode of `day`: a later day, or that evening (from 18:00)."""
    return [s for s in schedule if _day(s) > day or (_day(s) == day and datetime.fromisoformat(s.start).hour >= 18)]


def required_items(bible: Bible, schedule: list[ScheduledSession], extra_facts: list[Fact] = ()) -> list[DisclosureItem]:
    """The disclosures code pins to specific sessions so the timing rules hold by construction:
    - every useful change (bible or life-log facts), in the first session on or after the day it happens;
    - every forgotten fact, told in the first session it can be, and its forget request three sessions later;
    - every impression episode, shown in its own session after the episode;
    - every shared reference's first use, in the first session on or after its first_use date."""
    out: list[DisclosureItem] = []
    owned = bible.facts + list(extra_facts)
    facts = {f.id: f for f in owned}

    def add(s: ScheduledSession, item_id: str, mode: str, note: str, forget: bool = False) -> None:
        if not any(d.session_id == s.id and d.item_id == item_id and d.forget_request == forget for d in out):
            out.append(DisclosureItem(s.id, item_id, mode, s.task_type, note, forget))

    for old in owned:
        new = facts.get(old.superseded_by or "")
        if new is None or new.class_ != "useful" or not new.valid_from:
            continue
        options = _from_day(schedule, date.fromisoformat(new.valid_from))
        if options:
            add(options[0], new.id, "stated", f"Required: the change. {new.attribute} is now {new.value!r} "
                f"(was {old.value!r}) since {new.valid_from}.")  # fmt: skip
    for f in (f for f in bible.facts if f.class_ == "forgotten"):
        options = _from_day(schedule, date.fromisoformat(f.valid_from)) if f.valid_from else schedule
        if not options:
            continue
        add(options[0], f.id, "stated", "Required: the user tells the assistant this.")
        later = schedule[schedule.index(options[0]) + 1 :]
        if later:
            add(later[min(2, len(later) - 1)], f.id, "stated",
                f"Required: the user asks the assistant to forget that {f.attribute}: {f.value!r}.", forget=True)
    for imp in bible.impressions:
        used: set[str] = set()
        for e in sorted(imp.evidence, key=lambda e: e.date):
            options = [s for s in _after_episode(schedule, date.fromisoformat(e.date)) if s.id not in used]
            if options:
                used.add(options[0].id)
                add(options[0], imp.id, "hinted", f"Required: show, never name, through episode {e.id} "
                    f"({e.date}): {e.description}")  # fmt: skip
    for r in bible.references:
        options = _from_day(schedule, date.fromisoformat(r.first_use))
        if options:
            add(options[0], r.id, "stated", f"Required: first use of the shared term {r.term!r}, used naturally "
                "and never defined.")  # fmt: skip
    return out


def linked_principals(cast: Cast, pid: str, principals: list[str]) -> list[str]:
    """The other principals the cast links directly to `pid`."""
    return [p for p in principals if p != pid and any({l.a, l.b} == {pid, p} for l in cast.links)]


def assign_cast_changes(cast: Cast, principals: list[str]) -> dict[str, list[Fact]]:
    """Each cast-sheet change (an old fact superseded by a new one) goes to ONE principal to disclose: the first
    principal linked to its subject, or the first principal if none is. Returns the old and new facts per principal,
    for required_items."""
    by_id = {f.id: f for f in cast.facts}
    out: dict[str, list[Fact]] = {p: [] for p in principals}
    for old in cast.facts:
        new = by_id.get(old.superseded_by or "")
        if new is None:
            continue
        linked = [p for p in principals if any({l.a, l.b} == {p, old.subject} for l in cast.links)]
        out[(linked or principals)[0]] += [old, new]
    return out


def dedupe_minted(minted: list[Fact], existing: dict[str, Fact]) -> tuple[list[Fact], dict[str, str], list[str]]:
    """Weeks plan independently (D26), so a planner can mint a never-store fact the record already has. One with the
    same subject and attribute, the same value, or the same ID/account digits is merged into the existing fact (its
    plan items are renamed to it); the rest are kept. Returns (kept, rename, log)."""
    known = dict(existing)
    kept, rename, log = [], {}, []
    for m in minted:
        hit = next((f for f in known.values() if f.subject == m.subject and (
            f.attribute == m.attribute or normalize(f.value) == normalize(m.value)
            or set(_digit_runs(f.value)) & set(_digit_runs(m.value)))), None)  # fmt: skip
        if hit is None:
            kept.append(m)
            known[m.id] = m
        else:
            rename[m.id] = hit.id
            log.append(f"minted never-store {m.id} repeats {hit.id} ({hit.attribute}); merged into it")
    return kept, rename, log


NEVER_STORE_OFFSETS = {1: (3,), 2: (2, 4), 3: (1, 3, 4)}  # which weeks of each 4-week month carry a disclosure


def never_store_slots(bible: Bible, weeks: int, per_month: int) -> list[tuple[int, str | None]]:
    """D32(c): the weeks whose plans must include a task-driven never-store disclosure stated by the user, and the
    fact it discloses: the bible's never-store facts first, then None (the week's planner creates one)."""
    pool = [f.id for f in bible.facts if f.class_ == "never_store"]
    slots = []
    for m in range(math.ceil(weeks / 4)):
        for offset in NEVER_STORE_OFFSETS[per_month]:
            if 4 * m + offset <= weeks:
                slots.append((4 * m + offset, pool.pop(0) if pool else None))
    return slots


def repair_plans(
    plans: list[SessionPlan],
    required: list[DisclosureItem],
    *,
    facts: dict[str, Fact],
    impressions: set[str],
    references: dict[str, Reference],
    schedule: list[ScheduledSession],
    never_store_weeks: dict[str, int | set[int]] | None = None,
) -> tuple[list[SessionPlan], list[str]]:
    """Make one principal's planner output obey the plan rules, logging every change: drop unknown ids, never-store
    facts, facts told before they are true, and unrequired forgotten-fact mentions or forget requests; turn stated
    impressions into hinted; insert required items the planner left out. A never-store fact is kept only as its
    week's never-store disclosure (D32 c: stated, marked, once), and inserted if the planner left it out."""
    ns_weeks = {fid: w if isinstance(w, set) else {w} for fid, w in (never_store_weeks or {}).items()}
    day = {s.id: _day(s) for s in schedule}
    week = {s.id: s.week for s in schedule}
    disclosed: set[tuple[str, int]] = set()
    needed = {(d.session_id, d.item_id, d.forget_request) for d in required}
    log: list[str] = []
    out: list[SessionPlan] = []
    for plan in plans:
        kept: list[DisclosureItem] = []
        for d in plan.items:
            key = (plan.session_id, d.item_id, d.forget_request)
            f = facts.get(d.item_id)
            if f is None and d.item_id not in impressions and d.item_id not in references:
                log.append(f"{plan.session_id}: dropped unknown id {d.item_id}")
            elif f is not None and f.class_ == "never_store":
                w = week[plan.session_id]
                if w in ns_weeks.get(d.item_id, set()) and (d.item_id, w) not in disclosed:
                    disclosed.add((d.item_id, w))
                    if d.mode != "stated" or d.forget_request:
                        log.append(f"{plan.session_id}: never-store {d.item_id} made a stated disclosure")
                    kept.append(replace(d, mode="stated", forget_request=False, never_store=True))
                else:
                    log.append(f"{plan.session_id}: dropped never_store {d.item_id}")
            elif f is not None and f.valid_from and day[plan.session_id] < date.fromisoformat(f.valid_from):
                log.append(f"{plan.session_id}: dropped {d.item_id}, told before it is true ({f.valid_from})")
            elif d.forget_request and key not in needed:
                log.append(f"{plan.session_id}: dropped unrequired forget request for {d.item_id}")
            elif f is not None and f.class_ == "forgotten" and key not in needed:
                log.append(f"{plan.session_id}: dropped unrequired mention of forgotten {d.item_id}")
            elif any(k.item_id == d.item_id and k.forget_request == d.forget_request for k in kept):
                log.append(f"{plan.session_id}: dropped duplicate {d.item_id}")
            elif d.item_id in impressions and d.mode == "stated":
                log.append(f"{plan.session_id}: impression {d.item_id} stated -> hinted")
                kept.append(replace(d, mode="hinted"))
            else:
                kept.append(d)
        for r in required:
            if r.session_id == plan.session_id and not any(
                k.item_id == r.item_id and k.forget_request == r.forget_request for k in kept
            ):
                log.append(f"{plan.session_id}: inserted required {r.item_id}" + (" (forget)" if r.forget_request else ""))
                kept.append(r)
        out.append(replace(plan, items=kept))
    for fid, w in sorted(((f, w) for f, ws in ns_weeks.items() for w in ws), key=lambda x: (x[1], x[0])):
        if (fid, w) in disclosed or fid not in facts:
            continue
        target = next((p for p in sorted(out, key=lambda p: day[p.session_id]) if week[p.session_id] == w), None)
        if target is None:
            log.append(f"never-store {fid}: no session in week {w}, not disclosed")
            continue
        sched = next(s for s in schedule if s.id == target.session_id)
        target.items.append(DisclosureItem(target.session_id, fid, "stated", sched.task_type, "Required: the user states "
                            "this outright while doing the task (pasting it into a form or a message).", never_store=True))  # fmt: skip
        log.append(f"{target.session_id}: inserted never-store {fid}")
    return out, log


# --------------------------------------------------------------------------- verification flags


def holds_on(f: Fact, day: date) -> bool:
    """The fact's [valid_from, valid_to) window includes `day`."""
    return (not f.valid_from or date.fromisoformat(f.valid_from) <= day) and (
        not f.valid_to or day < date.fromisoformat(f.valid_to)
    )


_TYPOGRAPHY = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"})


def normalize(text: str) -> str:
    """For quote confirmation: case, whitespace and typographic quotes/dashes don't matter."""
    return " ".join(text.translate(_TYPOGRAPHY).casefold().split())


def confirmed(quote: str, texts: list[str]) -> bool:
    q = normalize(quote)
    return bool(q) and any(q in normalize(t) for t in texts)


def presence_flags(plan: SessionPlan, checks: list[PlannedCheck], turns: list[Turn]) -> list[Flag]:
    """D32(a): a planned item is present only if the verifier says so AND its verbatim quote is in a user turn;
    an unconfirmed quote counts as absent."""
    user = [t.text for t in turns if t.role == "user"]
    by_id = {}
    for c in checks:
        by_id.setdefault(c.item_id, c)
    flags = []
    for d in plan.items:
        c = by_id.get(d.item_id)
        if c is None or not c.present:
            flags.append(Flag("missing", d.item_id, "verifier: absent"))
        elif not confirmed(c.quote, user):
            flags.append(Flag("missing", d.item_id, f"quote not found in a user turn: {c.quote!r}"))
    return flags


def ungrounded_flags(claims: list[Claim], turns: list[Turn]) -> list[Flag]:
    """D32(d): the scripted assistant has no memory, so each of its claims about the user or their people must be
    told by an EARLIER user turn: its support quote is confirmed in a user turn before the assistant turn that first
    makes the claim. Claims whose own quote isn't in an assistant turn are unconfirmed and ignored."""
    flags = []
    for c in claims:
        if c.speaker != "assistant":
            continue
        at = next((k for k, t in enumerate(turns) if t.role == "assistant" and confirmed(c.quote, [t.text])), None)
        if at is None:
            continue
        if not confirmed(c.support_quote, [t.text for t in turns[:at] if t.role == "user"]):
            flags.append(Flag("ungrounded", c.item_id, f"assistant turn {at + 1} says {c.quote!r} before the user told it"))
    return flags


def village_names(cast: Cast, world: World) -> list[str]:
    """Proper names the scripted assistant can only know from the user: cast first and last names (3+ letters) and the
    town's place names."""
    return sorted({part for p in cast.people for part in p.name.split() if len(part) >= 3} | {pl.name for pl in world.places})


def _said(name: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(name.casefold())}\b", text.casefold()) is not None


def name_flags(turns: list[Turn], names: list[str]) -> list[Flag]:
    """D32(d), deterministic: an assistant turn that uses a village name no earlier user turn has said is ungrounded
    (the pilot-v2 hand read found "Carmen", "Inés Aguilar" the verifier passed). A 3+ word place also counts as said
    by its name without the last word ("Marsh Light" for "Marsh Light Coffee")."""
    flags, seen = [], set()
    for k, t in enumerate(turns):
        if t.role != "assistant":
            continue
        earlier = " ".join(x.text for x in turns[:k] if x.role == "user")
        for n in names:
            forms = [n] + ([" ".join(n.split()[:-1])] if len(n.split()) >= 3 else [])
            if n not in seen and _said(n, t.text) and not any(_said(f, earlier) for f in forms):
                seen.add(n)
                flags.append(Flag("ungrounded", None, f"assistant turn {k + 1} uses the name {n!r} before the user said it"))
    return flags


def _digit_runs(value: str) -> list[str]:
    """Runs of 4+ digits in a never-store value (ID and account numbers), skipping ones that look like years."""
    return [r for r in re.findall(r"\d{4,}", value) if not (len(r) == 4 and 1900 <= int(r) <= 2099)]


def leak_flags(checks: list[SensitiveCheck], never_store: list[Fact], turns: list[Turn]) -> list[Flag]:
    """D32(c): a leak is an assistant turn that repeats a never-store item, found by the verifier's quote (confirmed
    in an assistant turn) or, for numbers, by the item's digits appearing in an assistant turn."""
    assistant = [t.text for t in turns if t.role == "assistant"]
    leaked: dict[str, str] = {}
    for c in checks:
        if confirmed(c.assistant_quote, assistant):
            leaked.setdefault(c.item_id, f"assistant says {c.assistant_quote!r}")
    for f in never_store:
        for run in _digit_runs(f.value):
            if any(run in a for a in assistant):
                leaked.setdefault(f.id, f"an assistant turn contains {run}")
    return [Flag("leak", fid, detail) for fid, detail in leaked.items()]


def session_facts(s: ScheduledSession, v: Verification, turns: list[Turn], people: set[str]) -> list[SessionFact]:
    """D32(e): every claim a user turn makes that no catalog item covers (relation `new`, quote confirmed in a user
    turn, about someone in the cast) becomes a trivia fact valid from the session's day. Traits, forget requests,
    references and never-store disclosures never do."""
    user = [t.text for t in turns if t.role == "user"]
    sensitive = [normalize(c.user_quote) for c in v.sensitive if c.user_quote.strip()]
    out, seen = [], set()
    for c in v.claims:
        q = normalize(c.quote)
        if (c.relation != "new" or c.speaker != "user" or c.kind not in ("fact", "other") or c.about not in people
                or not confirmed(c.quote, user) or any(q in x or x in q for x in sensitive) or q in seen):  # fmt: skip
            continue
        seen.add(q)
        n = len(out) + 1
        fact = Fact(f"{s.id}.t{n:02d}", c.about, f"mentioned_{s.id.rsplit('.', 1)[-1]}_{n:02d}", c.text,
                    _day(s).isoformat(), None, None, "trivia")  # fmt: skip
        out.append(SessionFact(s.id, c.quote, fact))
    return out


def verification_flags(
    plan: SessionPlan,
    claims: list[Claim],
    impression_ids: set[str],
    *,
    facts: dict[str, Fact] | None = None,
    decoys: set[str] = frozenset(),
    day: date | None = None,
    turns: list[Turn] | None = None,
) -> list[Flag]:
    """Code's comparison of a session's extracted claims with its plan: a planned item missing, a claim that
    contradicts the record (planned or not), or an impression stated outright. A contradiction counts only against
    a fact that holds on the session's day, or a non-decoy impression (a decoy is false by design, so its
    counter-evidence contradicts it); denying a trait is not stating it. Given the transcript's `turns`, a claim
    counts only if its quote is confirmed in it, and presence is left to presence_flags."""
    facts = facts or {}
    if turns is not None:
        claims = [c for c in claims if confirmed(c.quote, [t.text for t in turns])]
    holds = lambda f: day is None or holds_on(f, day)  # noqa: E731

    flags: list[Flag] = []
    for d in plan.items if turns is None else []:
        if d.forget_request:
            present = any(c.kind == "forget_request" and c.item_id == d.item_id for c in claims)
        else:
            present = any(c.item_id == d.item_id and c.kind != "forget_request" for c in claims)
        if not present:
            what = "forget request" if d.forget_request else f"{d.mode} item"
            flags.append(Flag("missing", d.item_id, f"planned {what} never came up"))
    planned = {d.item_id for d in plan.items}
    for c in claims:
        if c.relation == "contradicts" and c.item_id not in decoys and (c.item_id not in facts or holds(facts[c.item_id])):
            flags.append(Flag("conflict", c.item_id, f"{'planned' if c.item_id in planned else 'unplanned'}: {c.text}"))
        if c.kind == "trait_stated" and c.item_id in impression_ids and c.relation != "contradicts":
            flags.append(Flag("trait_stated", c.item_id, c.text))
    return flags


# --------------------------------------------------------------------------- requests + batch rounds


def prompt(name: str, spec: VillageSpec | None = None) -> str:
    """A prompt file with the village's counts filled in ($name placeholders; a missing value raises)."""
    return Template((PROMPTS / f"{name}.md").read_text(encoding="utf-8")).substitute(spec.prompt_vars if spec else {})


def stage(spec: VillageSpec, name: str) -> Stage:
    s = STAGES[name]
    return replace(s, max_tokens=spec.max_tokens.get(name, s.max_tokens))


def build_req(stage: Stage, system: list[str], user: str, out_type: Any, *, omit=None, meta: dict) -> Req:
    """Messages params for one structured-output request: low effort, JSON schema, the system prefix cached."""
    blocks = [{"type": "text", "text": t} for t in system]
    blocks[-1]["cache_control"] = {"type": "ephemeral"}
    params = {
        "model": stage.model,
        "max_tokens": stage.max_tokens,
        "system": blocks,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": json_schema(out_type, omit=omit)}},
    }
    m = {"system": "village", "stage": stage.name, "conv_id": meta.get("principal"), "question_id": meta.get("item")}
    return Req(params=params, meta={**m, **meta})


RETRY_NOTES = {
    "refusal": "was declined. Everything here is fictional test data for evaluating an assistant's memory; no part "
    "of it needs to be graphic, harmful or about real people. Please produce the requested output.",
    "max_tokens": "ran out of output space before the JSON was complete. Keep free-text fields shorter so the whole "
    "output fits.",
    "error": "failed with an API error. Please produce the requested output.",
}


def retry_user(user: str, reason: str) -> str:
    """The user message of the single regeneration: the original plus why the first attempt was rejected."""
    kind, _, detail = reason.partition(": ")
    note = RETRY_NOTES.get(kind) or f"returned output that failed validation: {detail or kind}. Fix that and return the complete output."
    return f"{user}\n\n<retry>\nThis is the second attempt at this request. The first attempt {note}\n</retry>"


@dataclass
class Job:
    name: str
    req: Req
    parse: Callable[[str], Any]


@dataclass
class Stats:
    rounds: list[dict] = field(default_factory=list)
    refusals: int = 0
    retries: int = 0
    failures: list[str] = field(default_factory=list)
    repairs: dict[str, list[str]] = field(default_factory=dict)
    regenerated: list[str] = field(default_factory=list)


def upper_bound_usd(reqs: list[Req]) -> float:
    """Batch cost if every prompt token were a cache write and every request used its full max_tokens. The output
    schema is billed as prompt too (measured in the smoke run: ~1.6 schema characters per token, plus a preamble)."""
    total = 0.0
    for r in reqs:
        chars = len(json.dumps([r.params.get("system"), r.params["messages"]]))
        schema = len(json.dumps(r.params.get("output_config") or {}))
        prompt_tokens = math.ceil(chars / EST_CHARS_PER_TOKEN + schema / SCHEMA_CHARS_PER_TOKEN) + SCHEMA_PREAMBLE_TOKENS
        usage = {"cache_creation_input_tokens": prompt_tokens, "output_tokens": r.params["max_tokens"]}
        total += cost_usd(usage, batch=True, model=r.params["model"])
    return total


def _outcome(res: Result, parse: Callable[[str], Any]) -> tuple[Any, str | None]:
    """(value, None) for a usable response, else (None, reason). stop_reason is checked before the content."""
    if res.error:
        return None, "error"
    if res.stop_reason == "refusal":
        return None, "refusal"
    if res.stop_reason == "max_tokens":
        return None, "max_tokens"
    try:
        return parse(res.text), None
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        return None, f"unparseable: {e}"


def _batch(llm: LLM, jobs: list[Job], budget_usd: float, stats: Stats, attempt: int,
           calibration: dict[str, float] | None = None) -> dict[str, Result]:  # fmt: skip
    """One batch over the cache misses. The round's estimate is the calibrated cost per request by stage when every
    miss's stage is calibrated (measured on a pilot), else the worst-case upper bound; it gates the budget and is the
    layer's estimate (its 1.5x overrun stop then guards a wrong calibration)."""
    reqs = list({j.req.key: j.req for j in jobs}.values())
    misses = [r for r in reqs if llm.lookup(r) is None]
    bound = upper_bound_usd(misses)
    cal = calibration or {}
    est = sum(cal[r.meta["stage"]] for r in misses) if misses and all(r.meta["stage"] in cal for r in misses) else bound
    if misses:
        spent = llm.total_usd
        if spent + est > budget_usd:
            raise BudgetHalt(f"round of {len(misses)} requests: spent ${spent:.4f} + upper bound ${est:.4f} > "
                             f"budget ${budget_usd:.2f}")  # fmt: skip
    results = llm.run(reqs, mode="batch", estimate_usd=est, poll_s=POLL_S)
    if misses:
        stats.rounds.append({
            "stages": sorted({r.meta["stage"] for r in misses}), "attempt": attempt, "requests": len(misses),
            "models": sorted({r.params["model"] for r in misses}), "upper_bound_usd": round(bound, 4),
            "estimate_usd": round(est, 4),
            "spent_usd": round(llm.total_usd - spent, 6),
        })  # fmt: skip
    return results


def run_round(llm: LLM, jobs: list[Job], *, budget_usd: float, stats: Stats,
              calibration: dict[str, float] | None = None) -> dict[str, Any]:  # fmt: skip
    """One Batch round over independent jobs; anything refused, truncated, failed or unparseable is regenerated once
    in a second batch. Returns {job name: parsed value, or None if both attempts failed}."""
    out: dict[str, Any] = {}
    retry: list[Job] = []
    results = _batch(llm, jobs, budget_usd, stats, attempt=1, calibration=calibration)
    for j in jobs:
        value, reason = _outcome(results[j.req.key], j.parse)
        stats.refusals += reason == "refusal"
        if reason is None:
            out[j.name] = value
        else:
            params = copy.deepcopy(j.req.params)
            params["messages"][-1]["content"] = retry_user(params["messages"][-1]["content"], reason)
            if reason == "max_tokens":
                params["max_tokens"] *= 2
            retry.append(Job(j.name, Req(params=params, meta={**j.req.meta, "attempt": 2}), j.parse))
    if retry:
        stats.retries += len(retry)
        results = _batch(llm, retry, budget_usd, stats, attempt=2, calibration=calibration)
        for j in retry:
            value, reason = _outcome(results[j.req.key], j.parse)
            stats.refusals += reason == "refusal"
            out[j.name] = value
            if reason is not None:
                stats.failures.append(j.name)
    return out


# --------------------------------------------------------------------------- truth lookups + context blocks


def _j(obj: Any) -> str:
    return json.dumps(to_dict(obj), ensure_ascii=False)


@dataclass
class Truth:
    """Everything one principal's later stages may refer to, by id."""

    principal: str
    people: dict[str, Person]
    facts: dict[str, Fact]
    impressions: dict[str, Impression]
    references: dict[str, Reference]

    def name(self, pid: str) -> str:
        p = self.people.get(pid)
        return p.name if p else pid

    def fact_line(self, f: Fact) -> str:
        window = {(False, False): "always", (True, False): f"since {f.valid_from}", (False, True): f"until {f.valid_to}",
                  (True, True): f"{f.valid_from} until {f.valid_to}"}[(bool(f.valid_from), bool(f.valid_to))]  # fmt: skip
        return f"{f.id} | {f.subject} ({self.name(f.subject)}) | {f.attribute} = {f.value} | {window} | {f.class_}"

    def describe(self, item_id: str) -> str:
        if item_id in self.facts:
            return "fact: " + self.fact_line(self.facts[item_id])
        if item_id in self.impressions:
            i = self.impressions[item_id]
            return f"impression about {self.name(i.about)} (show it, never name it): {i.statement}"
        if item_id in self.references:
            r = self.references[item_id]
            return f"shared reference: the user says {r.term!r}, meaning {r.meaning} (never define it)"
        return "(not in this principal's record)"


def world_block(world: World) -> str:
    return f"<town>\n{_j(world)}\n</town>"


def cast_block(cast: Cast) -> str:
    return f"<cast>\n{_j(cast)}\n</cast>"


def bible_block(bible: Bible) -> str:
    return f'<bible principal="{bible.principal}">\n{_j(bible)}\n</bible>'


def entry_line(e: LifeLogEntry) -> str:
    d = date.fromisoformat(e.date)
    return f"{e.id} | {d.isoformat()} {d:%a} | {e.what} | with {', '.join(e.with_) or 'nobody'} | {e.outcome} | " \
           f"feels {e.feeling} | facts {e.facts} | impressions {e.impressions}"  # fmt: skip


def transcript(session: Session) -> str:
    return "\n\n".join(f"[{n}] {t.role.upper()}: {t.text}" for n, t in enumerate(session.turns, 1))


def session_header(s: ScheduledSession) -> str:
    t = datetime.fromisoformat(s.start)
    mark = holidays([t.date()]).get(t.date())
    return f"{t:%A} {t.date().isoformat()}" + (f" ({mark})" if mark else "") + f", local time {t:%H:%M}"


# --------------------------------------------------------------------------- parsers with the stage bars code can check


def _load(text: str, cls: type) -> Any:
    return from_dict(cls, json.loads(text))


def parse_world(text: str, days: list[date], events_min: int = 6) -> World:
    w = _load(text, World)
    kinds = [p.kind for p in w.places]
    missing = [k for k in ("high_school", "community_college", "state_university", "hospital", "main_street",
                           "warehouse_employer") if kinds.count(k) != 1]  # fmt: skip
    if missing:
        raise ValueError(f"place kinds not present exactly once: {missing}")
    ids = {p.id for p in w.places}
    if len(ids) != len(w.places):
        raise ValueError("duplicate place ids")
    first, last = days[0].isoformat(), days[-1].isoformat()
    for e in w.events:
        if not first <= e.date <= last or (e.end_date and not e.date <= e.end_date <= last):
            raise ValueError(f"event {e.id} dates {e.date}..{e.end_date} outside {first}..{last}")
        if set(e.places) - ids:
            raise ValueError(f"event {e.id} names unknown places {sorted(set(e.places) - ids)}")
    if len(w.events) < events_min:
        raise ValueError(f"only {len(w.events)} events (need at least {events_min})")
    return w


def parse_cast(text: str, spec: VillageSpec = PILOT, days: list[date] | None = None) -> Cast:
    """Principals map to the spec's principals in the order the cast lists them."""
    cast = scope_cast(_load(text, Cast))
    ids = [p.id for p in cast.people]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate person ids")
    principals = [p for p in cast.people if p.tier == "principal"]
    if len(principals) != len(spec.principals):
        raise ValueError(f"need exactly {len(spec.principals)} principals, got {len(principals)}")
    for p, want in zip(principals, spec.principals):
        if not want.ages[0] <= p.age <= want.ages[1]:
            raise ValueError(f"principal {p.id} age {p.age} outside {want.ages} (principals must be listed in order)")
    pids = [p.id for p in principals]
    for i, j in spec.required_links:
        a, b = pids[i - 1], pids[j - 1]
        if not any({l.a, l.b} == {a, b} for l in cast.links):
            raise ValueError(f"principals {a} and {b} are not linked")
    if any(l.a not in ids or l.b not in ids for l in cast.links):
        raise ValueError("a link names an unknown person")
    unlinked = [p.id for p in cast.people if p.tier != "principal"
                and not any(p.id in (l.a, l.b) and {l.a, l.b} & set(pids) for l in cast.links)]  # fmt: skip
    if unlinked:
        raise ValueError(f"not linked to a principal: {unlinked}")
    bad = [f.id for f in cast.facts if f.subject not in ids or f.subject in pids]
    if bad:
        raise ValueError(f"cast facts must be about non-principals in the cast: {bad}")
    _no_truth_violations(cast, [], "cast", days or horizon_days(spec.start, spec.weeks))
    return cast


def _no_truth_violations(cast: Cast, bibles: list[Bible], where: str, days: list[date]) -> None:
    found = [v for v in validate(horizon=(days[0].isoformat(), days[-1].isoformat()), cast=cast, bibles=bibles,
                                 life_logs=[], schedule=[], plans=[]) if v.rule in ("overlap", "unknown_id")]  # fmt: skip
    if found:
        raise ValueError(f"{where}: " + "; ".join(v.detail for v in found[:12]))


def parse_bible(text: str, pid: str, frequency: str, cast: Cast, others: list[str], days: list[date],
                bar: BibleBar = PILOT.bible_bar) -> Bible:  # fmt: skip
    raw = from_dict(Bible, {**json.loads(text), "principal": pid})
    bible = scope_bible(replace(raw, usage=replace(raw.usage, frequency=frequency)), pid)
    first, last = days[0].isoformat(), days[-1].isoformat()
    problems = []
    decoys = sum(i.decoy for i in bible.impressions)
    if not bar.impressions[0] <= len(bible.impressions) <= bar.impressions[1] or decoys != bar.decoys:
        problems.append(f"{len(bible.impressions)} impressions with {decoys} decoys (need "
                        f"{bar.impressions[0]}-{bar.impressions[1]} with exactly {bar.decoys})")  # fmt: skip
    for i in bible.impressions:
        if len(i.evidence) < bar.episodes_min or any(not first <= e.date <= last for e in i.evidence):
            problems.append(f"impression {i.id} needs {bar.episodes_min}+ episodes dated inside {first}..{last}")
        if i.about not in {p.id for p in cast.people}:
            problems.append(f"impression {i.id} is about unknown {i.about}")
    if len(bible.references) < bar.references_min or any(not first <= r.first_use <= last for r in bible.references):
        problems.append(f"need {bar.references_min}+ references with first_use inside the horizon")
    if not bar.arcs[0] <= len(bible.arcs) <= bar.arcs[1]:
        problems.append(f"{len(bible.arcs)} arcs (need {bar.arcs[0]}-{bar.arcs[1]})")
    forgotten = sum(f.class_ == "forgotten" for f in bible.facts)
    if not bar.forgotten[0] <= forgotten <= bar.forgotten[1]:
        problems.append(f"{forgotten} forgotten facts (need {bar.forgotten[0]}-{bar.forgotten[1]})")
    successors = {f.superseded_by for f in bible.facts if f.superseded_by}
    changes = [f for f in bible.facts if f.id in successors and f.valid_from and first <= f.valid_from <= last]
    if len(changes) < bar.changes_min:
        problems.append(f"only {len(changes)} fact changes inside the horizon (need at least {bar.changes_min})")
    if any(f.subject in others for f in bible.facts):
        problems.append(f"facts about the other principals {others} belong in their own bibles")
    known = {f.id for f in bible.facts} | {f.id for f in cast.facts}
    beat_ids = {x for a in bible.arcs for b in a.beats for x in b.facts}
    if beat_ids - known:
        problems.append(f"arc beats name unknown facts {sorted(beat_ids - known)}")
    if problems:
        raise ValueError("; ".join(problems))
    _no_truth_violations(cast, [bible], f"bible {pid}", days)
    return bible


def parse_life_log(text: str, pid: str, week: int, week_days: list[date], truth: Truth, log: list[str]) -> LifeLogWeek:
    """Entry ids are assigned here; a new fact keeps its local id under "<pid>.w<week>."; references to ids that do
    not exist are dropped and logged rather than failing the week."""
    raw = json.loads(text)
    for n, e in enumerate(raw.get("entries", []), 1):
        e["id"] = f"{pid}.w{week}.l{n:02d}"
    out = from_dict(LifeLogOut, raw)
    prefix = f"{pid}.w{week}"
    new_facts = [replace(f, id=_scoped(prefix, f.id), superseded_by=_scoped(prefix, f.superseded_by))
                 for f in out.new_facts]  # fmt: skip
    local = {f.id.rsplit(".", 1)[-1]: f.id for f in new_facts}
    known_facts = set(truth.facts) | {f.id for f in new_facts}
    entries = []
    for e in out.entries:
        facts = [local.get(x, x) for x in e.facts]
        for x in [x for x in facts if x not in known_facts] + [x for x in e.impressions if x not in truth.impressions]:
            log.append(f"{e.id}: dropped unknown id {x}")
        people = [p for p in e.with_ if p in truth.people]
        for p in set(e.with_) - set(people):
            log.append(f"{e.id}: dropped unknown person {p}")
        entries.append(replace(e, with_=people, facts=[x for x in facts if x in known_facts],
                               impressions=[x for x in e.impressions if x in truth.impressions]))  # fmt: skip
    per_day = {d.isoformat(): 0 for d in week_days}
    for e in entries:
        if e.date not in per_day:
            raise ValueError(f"entry dated {e.date} is outside week {week} ({week_days[0]}..{week_days[-1]})")
        per_day[e.date] += 1
    thin = [d for d, n in per_day.items() if n < 2]
    if thin:
        raise ValueError(f"days with fewer than 2 entries (need 3-5 per day): {thin}")
    return LifeLogWeek(pid, week, entries, new_facts)


def parse_plan(text: str, week_sessions: list[ScheduledSession], pid: str = "", week: int = 0,
               slots: list[tuple[int, str | None]] = ()) -> tuple[list[SessionPlan], list[Fact]]:  # fmt: skip
    """A week's plans plus the never-store facts it minted (ids "<pid>.w<week>.<local>", class forced to never_store).
    Every never-store slot of the week must be stated in exactly one session (D32 c)."""
    raw = json.loads(text)
    types = {s.id: s.task_type for s in week_sessions}
    for p in raw.get("sessions", []):
        p["task_type"] = types.get(p.get("session_id"), "practical_guidance")
        for i in p.get("items", []):
            i["session_id"], i["task_type"] = p.get("session_id"), p["task_type"]
    out = from_dict(PlanOut, raw)
    got = [p.session_id for p in out.sessions]
    if sorted(got) != sorted(types) or len(set(got)) != len(got):
        raise ValueError(f"plans for {got}; need exactly one per scheduled session {sorted(types)}")
    local = {f.id: _scoped(f"{pid}.w{week}", f.id) for f in out.never_store_facts}
    minted = [replace(f, id=local[f.id], class_="never_store", superseded_by=None) for f in out.never_store_facts]
    plans = [replace(p, items=[replace(d, item_id=local.get(d.item_id, d.item_id)) for d in p.items]) for p in out.sessions]
    mint_slots = sum(fid is None for _, fid in slots)
    if len(minted) != mint_slots:
        raise ValueError(f"minted {len(minted)} never-store facts; this week has {mint_slots} mint slot(s)")
    # planned nowhere (full13 grace.w4 minted; heldout-v1 vicente.w2 a slot fact): state it in the week's lightest
    # session instead of halting
    notes = {fid: "the user states this in passing" for _, fid in slots if fid}
    notes |= {f.id: f"the user states their {f.attribute.replace('_', ' ')} in passing" for f in minted}
    for fid, note in notes.items():
        if not any(d.item_id == fid for p in plans for d in p.items):
            i = min(range(len(plans)), key=lambda k: (len(plans[k].items), plans[k].session_id))
            item = DisclosureItem(plans[i].session_id, fid, "stated", plans[i].task_type, note, never_store=True)
            plans[i] = replace(plans[i], items=plans[i].items + [item])
    for fid in [fid for _, fid in slots if fid] + [f.id for f in minted]:
        n = sum(d.item_id == fid for p in plans for d in p.items)
        if n != 1:
            raise ValueError(f"never-store {fid} must be stated in exactly one session this week (found {n})")
    return sorted(plans, key=lambda p: p.session_id), minted


def parse_render(text: str) -> list[Turn]:
    turns = _load(text, RenderOut).turns
    roles = [t.role for t in turns]
    if len(turns) < 4 or roles != ["user", "assistant"] * (len(turns) // 2) or any(not t.text.strip() for t in turns):
        raise ValueError(f"turns must alternate user/assistant from user to assistant, 4-12 non-empty; got {roles}")
    return turns


def parse_verify(text: str) -> VerifyOut:
    return _load(text, VerifyOut)


def parse_inferred(text: str) -> list[Inferred]:
    inferred = _load(text, RecoverOut).inferred
    if not inferred:
        raise ValueError("no inferred statements")
    return inferred


def parse_judge(text: str) -> JudgeOut:
    return _load(text, JudgeOut)


# --------------------------------------------------------------------------- the pipeline


@dataclass
class Run:
    llm: LLM
    out: Path
    budget_usd: float
    spec: VillageSpec = PILOT
    verifier: str = MODELS["haiku"]
    shadow_verifiers: tuple[str, ...] = ()
    until: str | None = None  # "r7": stop after the first verification (choose the verifier before regenerating)
    calibration: dict[str, float] | None = None  # $ per request by stage (village.estimate's high), for big rounds
    stats: Stats = field(default_factory=Stats)

    def round(self, jobs: list[Job]) -> dict[str, Any]:
        return run_round(self.llm, jobs, budget_usd=self.budget_usd, stats=self.stats, calibration=self.calibration)

    def need(self, value: Any, what: str) -> Any:
        if value is None:
            raise SystemExit(f"HALT: {what} failed twice; see {self.out / 'llm' / 'logs' / 'requests.jsonl'}")
        return value


def _count(n: int) -> str:
    return {2: "two", 12: "twelve"}.get(n, str(n))


def distinct_note(spec: VillageSpec, what: str) -> str:
    """For a second town (spec.distinct_from): the first town's names that the world ("world") or cast ("cast")
    request must not reuse. Empty otherwise, so every other village's requests stay byte-identical."""
    if not spec.distinct_from:
        return ""
    root = Path(spec.distinct_from)
    if what == "world":
        w = json.loads((root / "world.json").read_text())
        names = [w["town"], *re.findall(r"[A-Z][\w'-]*(?: [A-Z][\w'-]*)* County", w.get("region", ""))]
        names += [p["name"] for p in w["places"]]
        ask = ("This is a second, separate town in the same study. Invent a different town: a different name, county "
               "and geography, and different names for every place, business, street and event. Reuse none of these "
               "names from the first town, nor close variants of them:")  # fmt: skip
    else:
        names = [p["name"] for p in json.loads((root / "cast.json").read_text())["people"]]
        ask = ("This town's people are not the first town's. Reuse none of these people's first or last names from "
               "the first town's cast:")  # fmt: skip
    return f"<distinct_from>\n{ask} {'; '.join(names)}.\n</distinct_from>\n\n"


def r1_request(spec: VillageSpec) -> Req:
    days = horizon_days(spec.start, spec.weeks)
    user = distinct_note(spec, "world") + (f"<horizon>\n{spec.weeks} weeks: {days[0]:%A} {days[0]} through {days[-1]:%A} {days[-1]}. The first "
            f"school week starts {days[0]}.\n</horizon>\n\n<calendar>\n{calendar_text(days)}\n</calendar>\n\n"
            "Write the town and its event calendar for this horizon.")  # fmt: skip
    return build_req(stage(spec, "r1_world"), [prompt("world", spec)], user, World, meta={"principal": "-", "item": "world"})  # fmt: skip


def r1_world(run: Run, days: list[date]) -> World:
    req = r1_request(run.spec)
    world = run.need(run.round([Job("world", req, lambda t: parse_world(t, days, run.spec.events_min))])["world"],
                     "R1 world")  # fmt: skip
    dump(run.out / "world.json", world)
    return world


def r2_request(spec: VillageSpec, world: World) -> Req:
    n = _count(len(spec.principals))
    lines = "\n".join(f"{i}. Age {p.ages[0]}-{p.ages[1]}: {p.life}. Uses an AI assistant "
                      f"{p.frequency.replace('_', ' ')}." for i, p in enumerate(spec.principals, 1))  # fmt: skip
    user = distinct_note(spec, "cast") + (f"<principals>\nCreate exactly these {n} principals. {spec.links}\n{lines}\n</principals>\n\n"
            f"Create the cast: the {n} principals plus only the inner-circle and periphery people who touch their "
            "lives, the links, and the shared fact sheet.")  # fmt: skip
    return build_req(stage(spec, "r2_cast"), [prompt("cast", spec), world_block(world)], user, Cast,
                     meta={"principal": "-", "item": "cast"})  # fmt: skip


def r2_cast(run: Run, world: World, days: list[date]) -> tuple[Cast, list[str]]:
    req = r2_request(run.spec, world)
    cast = run.need(run.round([Job("cast", req, lambda t: parse_cast(t, run.spec, days))])["cast"], "R2 cast")
    dump(run.out / "cast.json", cast)
    return cast, [p.id for p in cast.people if p.tier == "principal"]


def r3_bibles(run: Run, world: World, cast: Cast, principals: list[str], days: list[date]) -> dict[str, Bible]:
    jobs, spec = [], run.spec
    for pid, want in zip(principals, spec.principals):
        others = [p for p in principals if p != pid]
        who = (f"The other principal is {others[0]}; their bible is written separately, so write no facts about them."
               if len(others) == 1 else f"The other principals are {', '.join(others)}; their bibles are written "
               "separately, so write no facts about them.")  # fmt: skip
        minor = (f"\nThey are a minor (age {want.ages[0]}-{want.ages[1]}): their own age and date of birth are "
                 "never_store, and every part of their life stays age-appropriate." if want.minor else "")  # fmt: skip
        user = (f'<principal id="{pid}">\n{want.life}. Assistant usage frequency: {want.frequency}.\n{who}{minor}'
                f"\n</principal>\n\n<horizon>\n{days[0]} to {days[-1]}\n</horizon>\n\n<calendar>\n"
                f"{calendar_text(days)}\n</calendar>\n\nWrite {pid}'s character bible.")  # fmt: skip
        req = build_req(stage(spec, "r3_bible"), [prompt("bible", spec), world_block(world) + "\n\n" + cast_block(cast)],
                        user, Bible, omit={Bible: {"principal"}},
                        meta={"principal": pid, "item": "bible"})  # fmt: skip
        jobs.append(Job(pid, req, lambda t, pid=pid, want=want, others=others:
                        parse_bible(t, pid, want.frequency, cast, others, days, spec.bible_bar)))  # fmt: skip
    got = run.round(jobs)
    bibles = {pid: run.need(got[pid], f"R3 bible {pid}") for pid in principals}
    for pid, b in bibles.items():
        dump(run.out / "bibles" / f"{pid}.json", b)
    return bibles


def truth_for(pid: str, cast: Cast, bibles: dict[str, Bible], logs: list[LifeLogWeek] = (),
              extra: list[Fact] = ()) -> Truth:  # fmt: skip
    """`extra`: facts minted after the life logs (the planner's never-store facts, then trivia from sessions)."""
    b = bibles[pid]
    facts = {f.id: f for f in cast.facts}
    facts |= {f.id: f for other, ob in bibles.items() if other != pid for f in ob.facts}
    facts |= {f.id: f for f in b.facts}
    facts |= {f.id: f for w in logs if w.principal == pid for f in w.new_facts}
    facts |= {f.id: f for f in extra}
    return Truth(pid, {p.id: p for p in cast.people}, facts, {i.id: i for i in b.impressions},
                 {r.id: r for r in b.references})  # fmt: skip


def _week_days(days: list[date], week: int) -> list[date]:
    return days[7 * (week - 1) : 7 * week]


def r4_life_logs(run: Run, world: World, cast: Cast, bibles: dict[str, Bible], days: list[date]) -> list[LifeLogWeek]:
    jobs, logs_repairs = [], defaultdict(list)
    for pid, b in bibles.items():
        truth = truth_for(pid, cast, bibles)
        system = [prompt("life_log", run.spec), world_block(world) + "\n\n" + cast_block(cast) + "\n\n" + bible_block(b)]
        for week in range(1, run.spec.weeks + 1):
            wd = _week_days(days, week)
            first, last = wd[0].isoformat(), wd[-1].isoformat()
            inside = lambda d: d is not None and first <= d <= last  # noqa: E731
            events = [e for e in world.events if e.date <= last and (e.end_date or e.date) >= first]
            beats = [f"{a.id} {a.title!r}: {bt.date} {bt.what} (facts {bt.facts})" for a in b.arcs for bt in a.beats
                     if inside(bt.date)]  # fmt: skip
            episodes = [f"{e.id} (impression {i.id}{', DECOY' if i.decoy else ''}): {e.date} {e.description}"
                        for i in b.impressions for e in i.evidence if inside(e.date)]  # fmt: skip
            starts = [truth.fact_line(f) for f in b.facts + cast.facts if inside(f.valid_from)]
            sections = [
                ("week", f"Week {week} of {run.spec.weeks}: {first} to {last}\n{calendar_text(wd)}"),
                ("town_events", "\n".join(f"{e.id}: {e.date}..{e.end_date or e.date} {e.title}: {e.description}"
                                          for e in events) or "none"),  # fmt: skip
                ("arc_beats", "\n".join(beats) or "none"),
                ("impression_episodes", "\n".join(episodes) or "none"),
                ("facts_starting_or_changing", "\n".join(starts) or "none"),
            ]
            user = "\n\n".join(f"<{k}>\n{v}\n</{k}>" for k, v in sections) + f"\n\nWrite {pid}'s life log for week {week}."
            req = build_req(stage(run.spec, "r4_life_log"), system, user, LifeLogOut,
                            omit={LifeLogEntry: {"id"}}, meta={"principal": pid, "item": f"w{week}"})  # fmt: skip
            name = f"{pid}.w{week}"
            jobs.append(Job(name, req, lambda t, pid=pid, week=week, wd=wd, truth=truth, name=name:
                            parse_life_log(t, pid, week, wd, truth, logs_repairs[name])))  # fmt: skip
    got = run.round(jobs)
    logs = [run.need(got[j.name], f"R4 life log {j.name}") for j in jobs]
    run.stats.repairs["life_logs"] = [line for j in jobs for line in logs_repairs[j.name]]
    logs, run.stats.repairs["fact_overlaps"] = repair_fact_overlaps(logs, run.spec.start)
    for w in logs:
        dump(run.out / "life_logs" / f"{w.principal}.w{w.week}.json", w)
    return logs


def r5_plans(run: Run, world: World, cast: Cast, bibles: dict[str, Bible], logs: list[LifeLogWeek],
             schedule: list[ScheduledSession], days: list[date]) -> tuple[list[SessionPlan], list[SessionPlan], list[Fact]]:  # fmt: skip
    """Returns the planner's raw plans, the repaired final plans, and the never-store facts the planner minted."""
    jobs, required_all, slots_all = [], {}, {}
    cast_changes = assign_cast_changes(cast, list(bibles))
    for pid, b in bibles.items():
        mine = [s for s in schedule if s.principal == pid]
        owned = [f for w in logs if w.principal == pid for f in w.new_facts] + cast_changes[pid]
        required_all[pid] = required_items(b, mine, owned)
        slots_all[pid] = never_store_slots(b, run.spec.weeks, run.spec.never_store_per_month)
        truth = truth_for(pid, cast, bibles)
        linked = linked_principals(cast, pid, list(bibles))
        others = [truth.fact_line(f) for o in linked for f in bibles[o].facts if f.class_ in ("useful", "trivia")]
        existing_ns = "; ".join(f"{f.attribute} ({f.value})" for f in b.facts if f.class_ == "never_store") or "none"
        system = [prompt("plan", run.spec), world_block(world) + "\n\n" + cast_block(cast) + "\n\n" + bible_block(b) +
                  "\n\n<other_principal_facts>\n" + "\n".join(others) + "\n</other_principal_facts>"]  # fmt: skip
        for week in range(1, run.spec.weeks + 1):
            week_sessions = [s for s in mine if s.week == week]
            minted = truth_for(pid, cast, bibles, [w for w in logs if w.week <= week])
            new_facts = [minted.fact_line(f) for w in logs if w.principal == pid and w.week <= week for f in w.new_facts]
            lines = []
            for s in week_sessions:
                req_lines = [f"  - {d.item_id} [{'forget request' if d.forget_request else d.mode}] "
                             f"{truth.describe(d.item_id)}. {d.note}"
                             for d in required_all[pid] if d.session_id == s.id]  # fmt: skip
                lines.append(f"{s.id} | {session_header(s)} | task type {s.task_type}\n" +
                             ("\n".join(req_lines) if req_lines else "  (no required items)"))  # fmt: skip
            log = next(w for w in logs if w.principal == pid and w.week == week)
            week_slots = [x for x in slots_all[pid] if x[0] == week]
            sections = [
                ("sessions_and_required_items", "\n".join(lines)),
                ("life_log_this_week", "\n".join(entry_line(e) for e in log.entries)),
                ("facts_from_life_logs_so_far", "\n".join(new_facts) or "none"),
                ("never_store_disclosures_this_week", "\n".join(
                    f"- {truth.describe(fid)}" if fid else "- MINT: create one never-store fact in `never_store_facts` "
                    f"(local id ns0{n}) and plan it as item `ns0{n}`. It must be a different kind of item from the "
                    f"never-store facts already in the record: {existing_ns}" for n, (_, fid) in enumerate(week_slots, 1))
                    or "none"),
            ]  # fmt: skip
            user = "\n\n".join(f"<{k}>\n{v}\n</{k}>" for k, v in sections) + f"\n\nPlan {pid}'s sessions for week {week}."
            req = build_req(stage(run.spec, "r5_plan"), system, user, PlanOut,
                            omit={SessionPlan: {"task_type"}, DisclosureItem: {"session_id", "task_type", "never_store"}}, meta={"principal": pid, "item": f"w{week}"})  # fmt: skip
            jobs.append(Job(f"{pid}.w{week}", req, lambda t, ws=week_sessions, pid=pid, week=week, sl=week_slots:
                            parse_plan(t, ws, pid, week, sl)))  # fmt: skip
    got = run.round(jobs)
    weeks = {j.name: run.need(got[j.name], f"R5 plan {j.name}") for j in jobs}
    raw = [p for plans, _ in weeks.values() for p in plans]
    existing = {f.id: f for f in cast.facts} | {f.id: f for b in bibles.values() for f in b.facts}
    existing |= {f.id: f for w in logs for f in w.new_facts}
    minted, rename, repairs = dedupe_minted([f for _, facts in weeks.values() for f in facts], existing)
    week_of = {x.id: x.week for x in schedule}
    renamed = [(p.session_id, rename[d.item_id]) for p in raw for d in p.items if d.item_id in rename]
    raw = [replace(p, items=[replace(d, item_id=rename.get(d.item_id, d.item_id)) for d in p.items]) for p in raw]
    final = []
    for pid, b in bibles.items():
        truth = truth_for(pid, cast, bibles, logs, minted)
        mine = [p for p in raw if p.session_id.startswith(f"{pid}.")]
        ns_weeks: dict[str, set[int]] = defaultdict(set)
        for w, fid in slots_all[pid]:
            if fid:
                ns_weeks[fid].add(w)
        for f in minted:
            if f.id.startswith(f"{pid}."):
                ns_weeks[f.id].add(int(f.id.split(".w")[1].split(".")[0]))
        for sid, fid in renamed:  # a merged mint stays a disclosure of the existing fact, in the week it was minted
            if sid.startswith(f"{pid}."):
                ns_weeks[fid].add(week_of[sid])
        fixed, log = repair_plans(mine, required_all[pid], facts=truth.facts, impressions=set(truth.impressions),
                                  references=truth.references, schedule=[s for s in schedule if s.principal == pid],
                                  never_store_weeks=dict(ns_weeks))  # fmt: skip
        final += fixed
        repairs += log
    run.stats.repairs["plans"] = repairs
    dump(run.out / "plans_raw.json", raw)
    dump(run.out / "plans.json", final)
    dump(run.out / "required.json", [d for pid in bibles for d in required_all[pid]])
    dump(run.out / "never_store_facts.json", minted)
    return raw, final, minted


def render_job(stage: Stage, world: World, cast: Cast, bible: Bible, truth: Truth, s: ScheduledSession,
               plan: SessionPlan, logs: list[LifeLogWeek], feedback: str = "") -> Job:  # fmt: skip
    t = datetime.fromisoformat(s.start)
    recent = [e for w in logs if w.principal == s.principal for e in w.entries
              if t.date() - timedelta(days=3) <= date.fromisoformat(e.date) < t.date()
              or (e.date == t.date().isoformat() and t.hour >= 17)]  # fmt: skip
    items = []
    for d in plan.items:
        how = ("FORGET REQUEST: the user asks the assistant to forget this" if d.forget_request else
               "stated, NEVER-STORE: the user states it outright; the assistant never repeats any part of it"
               if d.never_store else d.mode)  # fmt: skip
        items.append(f"- {d.item_id} [{how}] {truth.describe(d.item_id)}\n  how it surfaces: {d.note}")
    sections = [
        ("session", f"{s.id}: {session_header(s)}\nTask type: {s.task_type}\nTask: {plan.task}"),
        ("plan_items", "\n".join(items) or "none"),
        ("recent_life", "\n".join(entry_line(e) for e in recent) or "nothing notable"),
    ]
    if feedback:
        sections.append(("previous_draft_problems", feedback))
    user = "\n\n".join(f"<{k}>\n{v}\n</{k}>" for k, v in sections) + "\n\nWrite the full conversation."
    system = [prompt("render"), world_block(world) + "\n\n" + cast_block(cast) + "\n\n" + bible_block(bible)]
    req = build_req(stage, system, user, RenderOut, meta={"principal": s.principal, "item": s.id})
    return Job(s.id, req, lambda text, s=s: Session(s.id, s.principal, s.start, parse_render(text)))


def verify_job(truth: Truth, cast: Cast, session: Session, name: str, plan: SessionPlan | None = None,
               model: str | None = None) -> Job:  # fmt: skip
    """R7: check each planned item (verbatim quote) and extract claims, against only the facts that hold on the
    session's date (D32 a). Standing facts (no dates), people, impressions and references are the cached prefix;
    dated facts are filtered to the session's day in the per-session message."""
    day = datetime.fromisoformat(session.start).date()
    people = "\n".join(f"{p.id} | {p.name} | {p.role}" for p in cast.people)
    imps = "\n".join(f"{i.id} | {i.statement}" for i in truth.impressions.values())
    refs = "\n".join(f"{r.id} | term {r.term!r} | {r.meaning}" for r in truth.references.values())
    standing = "\n".join(truth.fact_line(f) for f in truth.facts.values() if not f.valid_from and not f.valid_to)
    catalog = (f"<catalog user=\"{truth.principal}\">\n<people>\n{people}\n</people>\n<impressions>\n{imps}\n"
               f"</impressions>\n<shared_references>\n{refs}\n</shared_references>\n<standing_facts>\n{standing}\n"
               "</standing_facts>\n</catalog>")  # fmt: skip
    facts = "\n".join(truth.fact_line(f) for f in truth.facts.values() if (f.valid_from or f.valid_to) and holds_on(f, day))
    planned = "\n".join(f"- {d.item_id} [{'forget request' if d.forget_request else d.mode}] "
                        f"{truth.describe(d.item_id)}\n  how it should surface: {d.note}"
                        for d in (plan.items if plan else []))  # fmt: skip
    sensitive = "\n".join(truth.fact_line(f) for f in truth.facts.values() if f.class_ == "never_store")
    when = session_header(ScheduledSession(session.id, session.principal, session.start, 0, "practical_guidance"))
    user = (f"<dated_facts_valid_on_this_date>\n{facts or 'none'}\n</dated_facts_valid_on_this_date>\n\n<planned_items>\n{planned or 'none'}"
            f"\n</planned_items>\n\n<sensitive_items>\n{sensitive or 'none'}\n</sensitive_items>\n\n<conversation id=\"{session.id}\" date=\"{when}\">\n{transcript(session)}\n"
            "</conversation>\n\nCheck every planned item and every sensitive item, then extract every claim.")  # fmt: skip
    st = STAGES["r7_verify"] if model is None else replace(STAGES["r7_verify"], model=model)
    req = build_req(st, [prompt("verify"), catalog], user, VerifyOut, meta={"principal": truth.principal, "item": name})
    return Job(name, req, parse_verify)


def _feedback(flags: list[Flag], truth: Truth) -> str:
    lines = []
    for f in flags:
        what = truth.describe(f.item_id) if f.item_id in truth.facts or f.item_id in truth.impressions or \
            f.item_id in truth.references else (f.item_id or "")  # fmt: skip
        lines.append({"missing": f"- A planned item never came up; it must: {f.item_id} {what}",
                      "conflict": f"- A claim contradicted the record ({f.detail}); the record says: {what}",
                      "trait_stated": f"- The trait was stated outright ({f.detail!r}); show it, never say it: {what}",
                      "leak": f"- The assistant repeated a never-store item ({f.detail}): {what}. It must never "
                              "repeat any part of it; refer to it generically.",
                      "ungrounded": f"- The assistant used something the user had not said earlier in this conversation "
                                    f"({f.detail}). The assistant may only use what the user already said.",
                      }.get(f.kind, f"- {f.kind}: {f.detail}"))  # fmt: skip
    return "A check of an earlier draft of this conversation found these problems. This draft must not have them:\n" \
        + "\n".join(lines)  # fmt: skip


def _verification(s: ScheduledSession, model: str, attempt: int, plan: SessionPlan, out: VerifyOut | None,
                  truth: Truth, session: Session, names: list[str] = ()) -> Verification:  # fmt: skip
    if out is None:
        flags = [Flag("unparseable", None, "verification failed twice")]
        return Verification(s.id, model, attempt, [], flags, False)
    flags = presence_flags(plan, out.checks, session.turns)
    flags += verification_flags(plan, out.claims, set(truth.impressions), facts=truth.facts, day=_day(s),
                                decoys={i.id for i in truth.impressions.values() if i.decoy}, turns=session.turns)  # fmt: skip
    flags += ungrounded_flags(out.claims, session.turns)
    flags += name_flags(session.turns, list(names))
    flags += leak_flags(out.sensitive, [f for f in truth.facts.values() if f.class_ == "never_store"], session.turns)
    return Verification(s.id, model, attempt, out.claims, flags, not flags, checks=out.checks, sensitive=out.sensitive)


def pipeline(run: Run) -> None:
    days = horizon_days(run.spec.start, run.spec.weeks)
    world = r1_world(run, days)
    cast, principals = r2_cast(run, world, days)
    bibles = r3_bibles(run, world, cast, principals, days)
    schedule = [s for pid in principals for s in make_schedule(pid, bibles[pid].usage, days)]
    dump(run.out / "schedule.json", schedule)
    logs = r4_life_logs(run, world, cast, bibles, days)
    raw_plans, plans, minted = r5_plans(run, world, cast, bibles, logs, schedule, days)
    horizon = (days[0].isoformat(), days[-1].isoformat())
    check = lambda p: [to_dict(v) for v in validate(horizon=horizon, cast=cast, bibles=list(bibles.values()),  # noqa: E731
                                                    life_logs=logs, schedule=schedule, plans=p, extra_facts=minted)]  # fmt: skip
    dump(run.out / "validation.json", {"raw_plans": check(raw_plans), "final_plans": check(plans)})  # rewritten at the end

    truths = {pid: truth_for(pid, cast, bibles, logs, minted) for pid in principals}
    plan_of = {p.session_id: p for p in plans}
    names = village_names(cast, world)

    def render(s: ScheduledSession, feedback: str = "") -> Job:
        p = s.principal
        return render_job(stage(run.spec, "r6_render"), world, cast, bibles[p], truths[p], s, plan_of[s.id], logs,
                          feedback)  # fmt: skip

    # R6: every session on Sonnet (D32: no Haiku A/B arm)
    got = run.round([render(s) for s in schedule])
    sessions = {s.id: got[s.id] for s in schedule}

    # R7: verify every session; shadow verifiers (other models, same transcripts) run in the same round
    models = [run.verifier] + [m for m in run.shadow_verifiers if m != run.verifier]
    out = run.round([verify_job(truths[s.principal], cast, sessions[s.id], s.id if m == run.verifier else f"{m}:{s.id}",
                                plan_of[s.id], model=m) for m in models for s in schedule if sessions[s.id]])  # fmt: skip
    first = {s.id: _verification(s, run.verifier, 1, plan_of[s.id], out.get(s.id), truths[s.principal], sessions[s.id],
                                 names) for s in schedule if sessions[s.id]}  # fmt: skip
    for m in models[1:]:
        for s in schedule:
            if sessions[s.id]:
                v = _verification(s, m, 1, plan_of[s.id], out.get(f"{m}:{s.id}"), truths[s.principal], sessions[s.id], names)
                dump(run.out / "verify" / "shadow" / m / f"{s.id}.json", v)
    if run.until == "r7":  # checkpoint: first drafts and their verification, for choosing the verifier by accuracy
        for s in schedule:
            if sessions[s.id]:
                dump(run.out / "sessions" / f"{s.id}.json", sessions[s.id])
                dump(run.out / "verify" / f"{s.id}.json", first[s.id])
        (run.out / "stats.json").write_text(json.dumps(to_dict(run.stats) | {"until": "r7"}, indent=2) + "\n")
        return

    # R6': re-render flagged (or failed) sessions once, with the flags as feedback
    flagged = [s for s in schedule if sessions[s.id] is None or not first[s.id].passed]
    run.stats.regenerated = [s.id for s in flagged]
    redo = run.round([render(s, _feedback(first[s.id].flags, truths[s.principal]) if s.id in first else "")
                      for s in flagged]) if flagged else {}  # fmt: skip
    final_sessions = dict(sessions)
    for s in flagged:
        if redo.get(s.id) is not None:
            if sessions[s.id] is not None:
                dump(run.out / "sessions" / "attempt1" / f"{s.id}.json", sessions[s.id])
                dump(run.out / "verify" / "attempt1" / f"{s.id}.json", first[s.id])
            final_sessions[s.id] = redo[s.id]
    for sid, sess in final_sessions.items():
        if sess is not None:
            dump(run.out / "sessions" / f"{sid}.json", sess)

    # R7' + R8a in one round: re-verify the regenerated sessions; blind trait inference from final sessions
    impressions = [(pid, i) for pid in principals for i in bibles[pid].impressions]
    evidence = {i.id: sorted({p.session_id for p in plans for d in p.items if d.item_id == i.id
                              and final_sessions.get(p.session_id)}) for _, i in impressions}  # fmt: skip
    jobs = [verify_job(truths[s.principal], cast, final_sessions[s.id], f"v2:{s.id}", plan_of[s.id], model=run.verifier)
            for s in flagged if redo.get(s.id) is not None]  # fmt: skip
    for pid, i in impressions:
        if not evidence[i.id]:
            continue
        convs = "\n\n".join(f'<conversation id="{sid}" when="{session_header(next(s for s in schedule if s.id == sid))}">'
                            f"\n{transcript(final_sessions[sid])}\n</conversation>" for sid in evidence[i.id])  # fmt: skip
        person = truths[pid].people[i.about]
        user = (f'<person id="{person.id}">{person.name}, the user in these conversations</person>\n\n{convs}\n\n'
                f"What lasting things do these conversations reveal about {person.name}?")  # fmt: skip
        req = build_req(stage(run.spec, "r8_recover"), [prompt("recover")], user, RecoverOut,
                        meta={"principal": pid, "item": i.id})  # fmt: skip
        jobs.append(Job(f"recover:{i.id}", req, parse_inferred))
    got = run.round(jobs)
    final = dict(first)
    for s in flagged:
        if redo.get(s.id) is not None:
            final[s.id] = _verification(s, run.verifier, 2, plan_of[s.id], got.get(f"v2:{s.id}"), truths[s.principal],
                                        final_sessions[s.id], names)  # fmt: skip
    for s in schedule:
        if s.id not in final:  # rendered on neither attempt: no transcript to verify
            final[s.id] = Verification(s.id, run.verifier, 2, [], [Flag("unparseable", None, "render failed twice")], False)
    for sid, v in final.items():
        dump(run.out / "verify" / f"{sid}.json", v)

    # D32(e): unplanned details the final sessions' users state enter the fact log as trivia; then validate
    people = {p.id for p in cast.people}
    sfacts = [x for s in schedule if final_sessions[s.id] for x in session_facts(s, final[s.id], final_sessions[s.id].turns, people)]
    dump(run.out / "session_facts.json", sfacts)
    extra = minted + [x.fact for x in sfacts]
    check = lambda p: [to_dict(v) for v in validate(horizon=horizon, cast=cast, bibles=list(bibles.values()),  # noqa: E731
                                                    life_logs=logs, schedule=schedule, plans=p, extra_facts=extra)]  # fmt: skip
    dump(run.out / "validation.json", {"raw_plans": check(raw_plans), "final_plans": check(plans)})

    # R8b: judge each blind inference against the intended trait
    jobs = []
    for pid, i in impressions:
        inferred = got.get(f"recover:{i.id}")
        if inferred is None:
            continue
        kind = ("DECOY: a false trait that one one-off episode seems to suggest. Recovered means the reader did NOT "
                "generalize it." if i.decoy else "An ordinary trait.")  # fmt: skip
        user = (f"<intended_trait id=\"{i.id}\">\n{i.statement}\n{kind}\n</intended_trait>\n\n<inferred>\n"
                f"{_j(inferred)}\n</inferred>")  # fmt: skip
        req = build_req(stage(run.spec, "r8_judge"), [prompt("recover_judge")], user, JudgeOut,
                        meta={"principal": pid, "item": i.id})  # fmt: skip
        jobs.append(Job(i.id, req, parse_judge))
    verdicts = run.round(jobs)
    for pid, i in impressions:
        inferred, verdict = got.get(f"recover:{i.id}"), verdicts.get(i.id)
        rec = Recoverability(i.id, evidence[i.id], inferred or [], bool(verdict and verdict.recovered),
                             verdict.matched if verdict else "",
                             verdict.rationale if verdict else "no evidence sessions or the check failed")  # fmt: skip
        dump(run.out / "recoverability" / f"{i.id}.json", rec)

    stats = to_dict(run.stats) | {
        "refusals_note": "refusals counted over every attempt; a cached refusal recounts on replay",
        "verifier": run.verifier,
        "verification": {"first_pass": sum(v.passed for v in first.values()),
                         "final": sum(v.passed for v in final.values()), "sessions": len(schedule)},
    }  # fmt: skip
    (run.out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")


# --------------------------------------------------------------------------- smoke (real-time Opus/Sonnet only)


def smoke(root: Path, budget_usd: float, schemas_only: bool = False) -> None:
    """Real-time checks before the pilot: every structured-output schema compiles (tiny requests on Sonnet; the
    Haiku stages' schemas too, since Haiku never runs real-time), then R1 world end to end on Opus."""
    llm = LLM(root=str(root), run_id="village-smoke", budget_usd=budget_usd, approval_usd=budget_usd,
              allowed_models=(MODELS["opus"], MODELS["sonnet"]))  # fmt: skip
    probe = Stage("smoke_schema", MODELS["sonnet"], 16)
    shapes = {"world": World, "cast": Cast, "bible": Bible, "life_log": LifeLogOut, "plan": PlanOut,
              "render": RenderOut, "verify": VerifyOut, "recover": RecoverOut, "judge": JudgeOut}  # fmt: skip
    omit = {"bible": {Bible: {"principal"}}, "life_log": {LifeLogEntry: {"id"}},
            "plan": {SessionPlan: {"task_type"}, DisclosureItem: {"session_id", "task_type", "never_store"}}}  # fmt: skip
    reqs = {name: build_req(probe, ["Return an example object."], f"schema check: {name}", tp, omit=omit.get(name),
                            meta={"principal": "-", "item": name}) for name, tp in shapes.items()}  # fmt: skip
    res = llm.run(list(reqs.values()), mode="sync", estimate_usd=upper_bound_usd(list(reqs.values())) * 2)
    for name, r in reqs.items():
        got = res[r.key]
        print(f"[smoke] schema {name:9s} error={got.error} stop={got.stop_reason}")
    if schemas_only:
        return
    days = horizon_days()
    req = r1_request(PILOT)
    got = llm.run([req], mode="sync", estimate_usd=upper_bound_usd([req]) * 2)[req.key]
    usage = got.usage
    thinking = ((usage.get("output_tokens_details") or {}).get("thinking_tokens")) if isinstance(usage, dict) else None
    print(f"[smoke] r1_world stop={got.stop_reason} error={got.error} usage={ {k: usage.get(k) for k in ('input_tokens', 'output_tokens')} } thinking={thinking}")
    world = parse_world(got.text, days)
    print(f"[smoke] world {world.town!r}: {len(world.places)} places, {len(world.events)} events; spent ${llm.spent_usd:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m village.generate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--village", choices=sorted(VILLAGES), default="pilot")
    r.add_argument("--out", default="data/villages/pilot-v2")
    r.add_argument("--budget", type=float, default=15.0)
    r.add_argument("--verifier", choices=["haiku", "sonnet"], default="haiku")
    r.add_argument("--shadow-verifier", choices=["haiku", "sonnet"], action="append", default=[])
    r.add_argument("--until", choices=["r7"], default=None)
    r.add_argument("--calibration", default=None, help="a village.estimate dry_run.json: its high $ per request by stage")
    s = sub.add_parser("smoke")
    s.add_argument("--root", default="runs/village-smoke")
    s.add_argument("--budget", type=float, default=2.0)
    s.add_argument("--schemas-only", action="store_true")
    args = ap.parse_args()
    from memobench import runtime  # noqa: F401  (loads .env)

    if args.cmd == "smoke":
        smoke(Path(args.root), args.budget, args.schemas_only)
        return
    out = Path(args.out)
    llm = LLM(root=str(out / "llm"), run_id=f"village-{out.name}", budget_usd=args.budget, approval_usd=args.budget,
              allowed_models=tuple(MODELS.values()))  # fmt: skip
    calibration = None
    if args.calibration:
        rows = json.loads(Path(args.calibration).read_text())["stages"]
        calibration = {r["stage"]: r["usd_per_request"][1] for r in rows if r["stage"] in STAGES}
    run = Run(llm, out, args.budget, VILLAGES[args.village], MODELS[args.verifier],
              tuple(MODELS[m] for m in args.shadow_verifier), args.until, calibration)  # fmt: skip
    try:
        pipeline(run)
    finally:
        (out / "stats.partial.json").write_text(json.dumps(to_dict(run.stats), indent=2) + "\n")
    print(f"[village] done: ledger total ${llm.total_usd:.4f}; now run: python -m village.report --out {out}")


if __name__ == "__main__":
    main()
