"""Code validator for the Dream Village ground truth (design §3, "Code validator, no LLM").

Rules (each Violation.rule):
  overlap                 two facts on the same (subject, attribute) whose [valid_from, valid_to) windows overlap
  disclosed_before_valid  a fact is disclosed in a session dated before its valid_from (a change told before it happens)
  change_not_disclosed    a `useful` change (a fact that supersedes another) is never disclosed
  forget_order            a `forgotten` fact has no forget request; or the request is not strictly after the fact's
                          first disclosure; or the fact is disclosed again after it; or a forget request names a fact
                          not classed `forgotten`; or a probe testing it comes before the request
  unknown_id              a duplicate fact id, or a disclosure item / superseded_by / fact subject / impression
                          `about` that names nothing that exists
  session_time            a session outside the horizon, or a principal's sessions not in chronological order
  never_store_planned     a `never_store` fact in the disclosure plan without the never_store marker (a planned leak
                          test, D32 c), a marked one that is not `stated`, or the marker on any other fact
  impression_stated       an impression planned as `stated` (impressions are only ever shown)
  needed_after_probe      hook: a probe's must_use item not disclosed before the probe's time (probes don't exist yet)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from village.schema import Bible, Cast, DisclosureItem, Fact, LifeLogWeek, ScheduledSession, SessionPlan


@dataclass
class Violation:
    rule: str
    detail: str
    ids: list[str] = field(default_factory=list)


@dataclass
class ProbeRef:
    """What the validator needs from a probe (design §5) once probes exist: its time and item targets."""

    id: str
    t: str
    must_use: list[str]
    must_not: list[str]


def _day(s: str | None, default: date) -> date:
    return date.fromisoformat(s) if s else default


def _overlaps(a: Fact, b: Fact) -> bool:
    """[valid_from, valid_to) windows share a day; a missing bound is open."""
    return _day(a.valid_from, date.min) < _day(b.valid_to, date.max) and _day(b.valid_from, date.min) < _day(
        a.valid_to, date.max
    )


def validate(
    *,
    horizon: tuple[str, str],
    cast: Cast,
    bibles: list[Bible],
    life_logs: list[LifeLogWeek],
    schedule: list[ScheduledSession],
    plans: list[SessionPlan],
    probes: list[ProbeRef] = (),
    extra_facts: list[Fact] = (),
) -> list[Violation]:
    """Every rule violation in the village; [] means the ground truth passes. `horizon` is (first day, last day)."""
    out: list[Violation] = []
    facts: list[Fact] = cast.facts + [f for b in bibles for f in b.facts] + [f for w in life_logs for f in w.new_facts]
    facts += list(extra_facts)
    by_id: dict[str, Fact] = {}
    for f in facts:
        if f.id in by_id:
            out.append(Violation("unknown_id", f"fact id {f.id} is used twice", [f.id]))
        by_id.setdefault(f.id, f)
    impressions = {i.id: i for b in bibles for i in b.impressions}
    references = {r.id: r for b in bibles for r in b.references}
    items = set(by_id) | set(impressions) | set(references)
    people = {p.id for p in cast.people}
    sessions = {s.id: s for s in schedule}
    disclosures = [i for p in plans for i in p.items]

    # ---- ids
    for f in facts:
        if f.superseded_by is not None and f.superseded_by not in by_id:
            out.append(Violation("unknown_id", f"{f.id} superseded_by unknown fact {f.superseded_by}", [f.id]))
        if f.subject not in people:
            out.append(Violation("unknown_id", f"{f.id} subject {f.subject!r} is not in the cast", [f.id]))
    for i in impressions.values():
        if i.about not in people:
            out.append(Violation("unknown_id", f"{i.id} is about {i.about!r}, not in the cast", [i.id]))
    known: list[DisclosureItem] = []
    for d in disclosures:
        if d.item_id not in items:
            out.append(Violation("unknown_id", f"{d.session_id} plans unknown item {d.item_id}", [d.session_id, d.item_id]))
        elif d.session_id not in sessions:
            out.append(Violation("unknown_id", f"item {d.item_id} planned in unknown session {d.session_id}",
                                 [d.session_id, d.item_id]))  # fmt: skip
        else:
            known.append(d)

    def when(d: DisclosureItem) -> datetime:
        return datetime.fromisoformat(sessions[d.session_id].start)

    told = sorted((d for d in known if not d.forget_request), key=when)
    forgets = sorted((d for d in known if d.forget_request), key=when)

    # ---- validity windows
    windows: dict[tuple[str, str], list[Fact]] = defaultdict(list)
    for f in by_id.values():
        windows[(f.subject, f.attribute)].append(f)
    for (subject, attribute), group in windows.items():
        for n, a in enumerate(group):
            for b in group[n + 1 :]:
                if _overlaps(a, b):
                    out.append(Violation("overlap", f"{a.id} and {b.id} both hold {subject}.{attribute}", [a.id, b.id]))

    # ---- changes: told only once true; every useful change told
    for d in told:
        f = by_id.get(d.item_id)
        if f is not None and f.valid_from and when(d).date() < date.fromisoformat(f.valid_from):
            out.append(Violation("disclosed_before_valid",
                                 f"{f.id} (from {f.valid_from}) disclosed in {d.session_id} on {when(d).date()}",
                                 [f.id, d.session_id]))  # fmt: skip
    successors = {f.superseded_by for f in by_id.values() if f.superseded_by}
    told_ids = {d.item_id for d in told}
    for fid in sorted(successors & set(by_id)):
        if by_id[fid].class_ == "useful" and fid not in told_ids:
            out.append(Violation("change_not_disclosed", f"the change to {fid} is never disclosed", [fid]))

    # ---- forget requests
    for d in forgets:
        f = by_id.get(d.item_id)
        if f is None or f.class_ != "forgotten":
            out.append(Violation("forget_order", f"{d.session_id} asks to forget {d.item_id}, not a forgotten fact",
                                 [d.session_id, d.item_id]))  # fmt: skip
    for f in (f for f in by_id.values() if f.class_ == "forgotten"):
        requests = [d for d in forgets if d.item_id == f.id]
        mentions = [d for d in told if d.item_id == f.id]
        if not requests:
            out.append(Violation("forget_order", f"forgotten fact {f.id} has no forget request", [f.id]))
            continue
        first_request = when(requests[0])
        if not mentions or when(mentions[0]) >= first_request:
            out.append(Violation("forget_order", f"{f.id}: forget request in {requests[0].session_id} is not after "
                                 "its first disclosure", [f.id, requests[0].session_id]))  # fmt: skip
        for m in mentions:
            if when(m) > first_request:
                out.append(Violation("forget_order", f"{f.id} disclosed again in {m.session_id} after the forget "
                                     f"request in {requests[0].session_id}", [f.id, m.session_id]))  # fmt: skip
        for p in probes:
            if f.id in p.must_use + p.must_not and datetime.fromisoformat(p.t) <= first_request:
                out.append(Violation("forget_order", f"probe {p.id} tests {f.id} before its forget request",
                                     [f.id, p.id, requests[0].session_id]))  # fmt: skip

    # ---- session timestamps
    first = datetime.fromisoformat(horizon[0])
    end = datetime.fromisoformat(horizon[1]) + timedelta(days=1)
    last: dict[str, ScheduledSession] = {}
    for s in schedule:
        t = datetime.fromisoformat(s.start)
        if not first <= t < end:
            out.append(Violation("session_time", f"{s.id} at {s.start} is outside {horizon[0]}..{horizon[1]}", [s.id]))
        prev = last.get(s.principal)
        if prev is not None and t <= datetime.fromisoformat(prev.start):
            out.append(Violation("session_time", f"{s.id} at {s.start} is not after {prev.id} at {prev.start}",
                                 [prev.id, s.id]))  # fmt: skip
        last[s.principal] = s

    # ---- never-store, impressions
    for d in known:
        f = by_id.get(d.item_id)
        if f is not None and f.class_ == "never_store" and not (d.never_store and d.mode == "stated"):
            out.append(Violation("never_store_planned", f"never-store {f.id} planned in {d.session_id} without a "
                                 "stated never_store marker", [f.id, d.session_id]))  # fmt: skip
        if d.never_store and (f is None or f.class_ != "never_store"):
            out.append(Violation("never_store_planned", f"{d.item_id} in {d.session_id} is marked never_store but is "
                                 "not a never_store fact", [d.item_id, d.session_id]))  # fmt: skip
        if d.item_id in impressions and d.mode == "stated":
            out.append(Violation("impression_stated", f"impression {d.item_id} planned as stated in {d.session_id}",
                                 [d.item_id, d.session_id]))  # fmt: skip

    # ---- hook: every needed item is disclosed before the probe that needs it
    for p in probes:
        t = datetime.fromisoformat(p.t)
        for iid in p.must_use:
            if not any(d.item_id == iid and when(d) < t for d in told):
                out.append(Violation("needed_after_probe", f"probe {p.id} needs {iid}, not disclosed before {p.t}",
                                     [p.id, iid]))  # fmt: skip
    return out
