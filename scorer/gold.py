"""Gold from the village files, by code only (design §3): the disclosure record, `known_at`, the oracle memory.

This is the ONLY module that reads village files, so a format change (pilot-v2, D32) is a one-place fix. Files read:
manifest.json (horizon), cast.json, bibles/<p>.json, life_logs/*.json (new_facts), plans.json, sessions/*.json (top
level only; subfolders hold generator alternates), verify/<sid>.json (missing-item flags, optional), and the pilot-v2
files when present: never_store_facts.json (never-store facts the planner minted, D32 c; class never_store, so they
are withheld once pasted and `leaked` in memory) and session_facts.json (unplanned user details as SessionFact trivia,
D32 e; each is disclosed, mode `unplanned`, in its source session).

Rules (each a reading of design §3/§5/§6.2, see the scorer report for the ambiguities):
- "Before t" means a session that started before t (strictly), as in memsys.archive: the session at t isn't past yet.
- A fact holds on [valid_from, valid_to) by date. At t it is `current`; `stale` once ended AND the assistant could
  know it (no successor, e.g. a plan whose date passed, or the successor was disclosed before t); `ended_untold`
  when it ended but the change wasn't told yet (as told, it still looks current; probes may not target it).
- An impression is known once EVIDENCE_NEEDED sessions have shown it; decoys are never known. An impression that
  changes (`changed`, `changed_to`) becomes two entries: `<id>` ending on `changed`, and `<id>.changed` from then.
- A `forgotten` fact is known until its first forget request, then withheld. A `never_store` fact is never known; it
  is withheld once disclosed. Withheld items are what probes may only target as `must_not`.
- observed=True (pilot-v1, which predates D32 (e)) also reads the verifier's claims (verify/<sid>.json): a claim
  that matches an existing fact or reference (not a contradiction) is an `unplanned` disclosure in that session, and
  a new claim matching no entry becomes a `trivia` fact `<sid>.u<nn>` told there. Impression evidence stays
  plan-only. Off by default: pilot-v2 puts unplanned claims in the fact log itself.
- Lag (design §5) from the latest disclosure before t: same_session / same_day (no 03:00 night boundary crossed) /
  after_consolidation (a night crossed, still in the last RECENT_WINDOW sessions) / out_of_window (< 14 days) /
  weeks_later (< 60 days) / months_later.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path

from village.schema import (
    Bible, Cast, Fact, LifeLogWeek, Session, SessionFact, SessionPlan, from_dict, load, load_list,
)  # fmt: skip

EVIDENCE_NEEDED = 2  # sessions showing an impression before it is a pattern (v0 files a pattern once it recurs)
RECENT_WINDOW = 10  # sessions, the same window v0 and the recent_window baseline use
SAME_SESSION_H = 3
NIGHT_HOUR = 3  # the simulated consolidation boundary (v0: nightly at 03:00); lag is system-independent
LAG_BUCKETS = ("same_session", "same_day", "after_consolidation", "out_of_window", "weeks_later", "months_later")


def when(t: str | datetime) -> datetime:
    return t if isinstance(t, datetime) else datetime.fromisoformat(t)


@dataclass(frozen=True)
class Disclosure:
    session_id: str
    at: str  # the session's start
    mode: str  # stated | hinted
    flagged_missing: bool = False  # the verifier says this planned item never came up (often a verifier error, D32)


@dataclass(frozen=True)
class Entry:
    id: str
    kind: str  # fact | impression | reference
    subject: str  # a cast person id
    text: str  # fact value, impression statement, or reference meaning
    attribute: str = ""  # facts: the attribute; references: the term
    cls: str = ""  # facts: useful | trivia | never_store | forgotten; impressions: trait | decoy | sensitive
    valid_from: str | None = None
    valid_to: str | None = None
    superseded_by: str | None = None
    supersedes: str | None = None
    disclosures: tuple[Disclosure, ...] = ()
    forget_at: str | None = None  # the first forget request

    def label(self) -> str:
        """One line naming the entry, for prompts and cards."""
        if self.kind == "reference":
            return f'"{self.attribute}": {self.text}'
        if self.kind == "impression":
            return self.text
        return f"{self.attribute}: {self.text}"


@dataclass(frozen=True)
class Known:
    entry: Entry
    status: str  # current | stale | ended_untold
    disclosed: tuple[Disclosure, ...]  # the disclosures before t
    episodes: int  # sessions that showed it before t (impressions; for others the disclosure count)
    previous: Entry | None  # the entry it superseded, when that was disclosed before t
    lag: str


class Gold:
    def __init__(self, village: Path | str, *, observed: bool = False) -> None:
        self.root = Path(village)
        self.observed = observed
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.horizon: tuple[str, str] = tuple(manifest["horizon"])
        cast = load(self.root / "cast.json", Cast)
        self.names = {p.id: p.name for p in cast.people}
        self.bibles = {b.principal: b for b in (load(p, Bible) for p in sorted((self.root / "bibles").glob("*.json")))}
        self.principals = sorted(self.bibles)
        weeks = [load(p, LifeLogWeek) for p in sorted((self.root / "life_logs").glob("*.json"))]
        facts: list[Fact] = cast.facts + [f for b in self.bibles.values() for f in b.facts]
        facts += [f for w in weeks for f in w.new_facts]
        facts += self._optional_list("never_store_facts.json", Fact)
        session_facts: list[SessionFact] = self._optional_list("session_facts.json", SessionFact)
        facts += [sf.fact for sf in session_facts]
        plans = load_list(self.root / "plans.json", SessionPlan)
        self._sessions: dict[str, list[Session]] = {p: [] for p in self.principals}
        for path in sorted((self.root / "sessions").glob("*.json")):
            s = from_dict(Session, json.loads(path.read_text(encoding="utf-8")), path.name)
            self._sessions.setdefault(s.principal, []).append(s)
        for ss in self._sessions.values():
            ss.sort(key=lambda s: when(s.start))
        start = {s.id: s.start for ss in self._sessions.values() for s in ss}
        if start:  # the sessions bound the horizon too: full13-v1's manifest kept the pilot's end date
            days = sorted(x[:10] for x in start.values())
            self.horizon = (min(self.horizon[0], days[0]), max(self.horizon[1], days[-1]))
        verify = self._verify()
        missing = {(sid, f["item_id"]) for sid, v in verify.items() for f in v.get("flags", []) if f.get("kind") == "missing"}
        owner = {s.id: s.principal for ss in self._sessions.values() for s in ss}

        told: dict[tuple[str, str], list[Disclosure]] = {}
        forgets: dict[tuple[str, str], list[str]] = {}
        for plan in plans:
            if plan.session_id not in start:
                continue  # a planned session that was never rendered discloses nothing
            p = owner[plan.session_id]
            for d in plan.items:
                if d.forget_request:
                    forgets.setdefault((p, d.item_id), []).append(start[plan.session_id])
                else:
                    told.setdefault((p, d.item_id), []).append(Disclosure(
                        plan.session_id, start[plan.session_id], d.mode,
                        (plan.session_id, d.item_id) in missing))  # fmt: skip
        for sf in session_facts:
            if sf.source in start:
                told.setdefault((owner[sf.source], sf.fact.id), []).append(Disclosure(sf.source, start[sf.source], "unplanned"))
        trait_ids = {i.id for b in self.bibles.values() for i in b.impressions}
        unplanned: dict[str, list[Entry]] = {}
        if observed:
            for sid in sorted(set(verify) & set(start), key=lambda x: when(start[x])):
                p, n = owner[sid], 0
                for c in verify[sid].get("claims", []):
                    iid = c.get("item_id")
                    if iid and c["kind"] in ("fact", "reference_use", "other") and c["relation"] != "contradicts":
                        if iid not in trait_ids and not any(d.session_id == sid for d in told.get((p, iid), [])):
                            told.setdefault((p, iid), []).append(Disclosure(sid, start[sid], "unplanned"))
                    elif iid is None and c["kind"] in ("fact", "other") and c["relation"] == "new":
                        subject = c["about"] if c.get("about") in self.names else p
                        unplanned.setdefault(p, []).append(Entry(
                            f"{sid}.u{n:02d}", "fact", subject, c["text"], "unplanned", "trivia", start[sid][:10],
                            disclosures=(Disclosure(sid, start[sid], "unplanned"),)))  # fmt: skip
                        n += 1

        self._entries: dict[str, dict[str, Entry]] = {}
        by_fact = {f.id: f for f in facts}
        supersedes = {f.superseded_by: f.id for f in facts if f.superseded_by}
        for p in self.principals:
            bible = self.bibles[p]
            mine = {iid for (q, iid) in told if q == p} | {iid for (q, iid) in forgets if q == p}
            fact_ids = [f.id for f in facts if f.subject == p or f.id.startswith("cast.") or f.id in mine]
            out: dict[str, Entry] = {}

            def add(e: Entry) -> None:
                ds = tuple(sorted(told.get((p, e.id.removesuffix(".changed")), []), key=lambda d: when(d.at)))
                if e.kind == "impression" and e.supersedes:  # a changed trait: only episodes from the change on
                    ds = tuple(d for d in ds if when(d.at).date() >= date.fromisoformat(e.valid_from))
                fs = sorted(forgets.get((p, e.id), []), key=when)
                out[e.id] = replace(e, disclosures=ds, forget_at=fs[0] if fs else None)

            for fid in dict.fromkeys(fact_ids):
                f = by_fact[fid]
                add(Entry(f.id, "fact", f.subject, f.value, f.attribute, f.class_, f.valid_from, f.valid_to,
                          f.superseded_by, supersedes.get(f.id)))  # fmt: skip
            for i in bible.impressions:
                cls = "decoy" if i.decoy else "sensitive" if i.sensitive else "trait"
                if i.changed:
                    add(Entry(i.id, "impression", i.about, i.statement, "trait", cls, i.formed, i.changed,
                              f"{i.id}.changed"))  # fmt: skip
                    add(Entry(f"{i.id}.changed", "impression", i.about, i.changed_to, "trait", cls, i.changed,
                              None, None, i.id))  # fmt: skip
                else:
                    add(Entry(i.id, "impression", i.about, i.statement, "trait", cls, i.formed))
            for r in bible.references:
                add(Entry(r.id, "reference", p, r.meaning, r.term, "reference", r.first_use))
            out.update({e.id: e for e in unplanned.get(p, [])})
            self._entries[p] = out

    def _optional_list(self, name: str, cls: type) -> list:
        path = self.root / name
        return load_list(path, cls) if path.exists() else []

    def _verify(self) -> dict[str, dict]:
        """The verification record of each session (top level only), when the village has one."""
        d = self.root / "verify"
        files = sorted(d.glob("*.json")) if d.is_dir() else []
        return {v["session_id"]: v for v in (json.loads(f.read_text(encoding="utf-8")) for f in files)}

    # ------------------------------------------------------------------ lookups

    def sessions(self, principal: str) -> list[Session]:
        return self._sessions[principal]

    def entries(self, principal: str) -> dict[str, Entry]:
        return self._entries[principal]

    def entry(self, principal: str, item_id: str) -> Entry:
        return self._entries[principal][item_id]

    def name(self, person: str) -> str:
        return self.names.get(person, person)

    def voice(self, principal: str) -> dict:
        v = self.bibles[principal].voice
        return {"texting_style": v.texting_style, "with_assistant": v.with_assistant, "sample_lines": v.sample_lines}

    # ------------------------------------------------------------------ time rules

    @staticmethod
    def disclosed_by(e: Entry, t: str | datetime, *, inclusive: bool = False) -> tuple[Disclosure, ...]:
        """Disclosures in sessions that started before t (or at t, for a commit made by that session's write)."""
        cut = when(t)
        return tuple(d for d in e.disclosures if when(d.at) < cut or (inclusive and when(d.at) == cut))

    def ended(self, e: Entry, t: str | datetime) -> bool:
        return e.valid_to is not None and when(t).date() >= date.fromisoformat(e.valid_to)

    def status(self, principal: str, e: Entry, t: str | datetime, *, inclusive: bool = False) -> str:
        if not self.ended(e, t):
            return "current"
        nxt = self._entries[principal].get(e.superseded_by or "")
        if nxt is None or self.disclosed_by(nxt, t, inclusive=inclusive):
            return "stale"
        return "ended_untold"

    def forgotten_by(self, e: Entry, t: str | datetime, *, inclusive: bool = False) -> bool:
        return e.forget_at is not None and (when(e.forget_at) < when(t) or (inclusive and when(e.forget_at) == when(t)))

    def knowable(self, principal: str, e: Entry, t: str | datetime, *, inclusive: bool = False) -> bool:
        """Disclosed, not withheld, and (impressions) shown often enough, by t."""
        ds = self.disclosed_by(e, t, inclusive=inclusive)
        if not ds or e.cls in ("never_store", "decoy", "sensitive"):
            return False
        if e.kind == "impression" and len(ds) < EVIDENCE_NEEDED:
            return False
        return not self.forgotten_by(e, t, inclusive=inclusive)

    def known_at(self, principal: str, t: str | datetime) -> list[Known]:
        """What the assistant could know at t, each with its status; withheld items excluded."""
        out = []
        for e in self._entries[principal].values():
            if not self.knowable(principal, e, t):
                continue
            prev = self._entries[principal].get(e.supersedes or "")
            if prev is not None and not self.disclosed_by(prev, t):
                prev = None
            ds = self.disclosed_by(e, t)
            out.append(Known(e, self.status(principal, e, t), ds, len(ds), prev, self.lag(principal, e.id, t)))
        return out

    def withheld_at(self, principal: str, t: str | datetime) -> list[Entry]:
        """Never-store items disclosed before t, and forgotten items whose forget request came before t."""
        return [
            e for e in self._entries[principal].values()
            if (e.cls == "never_store" and self.disclosed_by(e, t)) or self.forgotten_by(e, t)
        ]  # fmt: skip

    def lag(self, principal: str, item_id: str, t: str | datetime, *, window: int = RECENT_WINDOW) -> str | None:
        ds = self.disclosed_by(self._entries[principal][item_id], t)
        if not ds:
            return None
        last, now = ds[-1], when(t)
        at = when(last.at)
        past = [s.id for s in self._sessions[principal] if when(s.start) < now]
        night = at.replace(hour=NIGHT_HOUR, minute=0, second=0, microsecond=0)
        night += timedelta(days=1) if night <= at else timedelta(0)
        if now < night:
            same = past and past[-1] == last.session_id and now - at < timedelta(hours=SAME_SESSION_H)
            return "same_session" if same else "same_day"
        if last.session_id in past[-window:]:
            return "after_consolidation"
        age = now - at
        return "out_of_window" if age < timedelta(days=14) else "weeks_later" if age < timedelta(days=60) else "months_later"

    # ------------------------------------------------------------------ renderings

    def excerpt(self, principal: str, item_id: str, t: str | datetime, chars: int = 400) -> str:
        """The user turn of the latest disclosing session before t that shares the most words with the entry."""
        e = self._entries[principal][item_id]
        ds = self.disclosed_by(e, t) or e.disclosures
        if not ds:
            return ""
        session = next(s for s in self._sessions[principal] if s.id == ds[-1].session_id)
        words = set(re.findall(r"\w{3,}", (e.text + " " + e.attribute).lower()))
        turns = [turn.text for turn in session.turns if turn.role == "user"] or [turn.text for turn in session.turns]
        best = max(turns, key=lambda x: len(words & set(re.findall(r"\w{3,}", x.lower()))))
        return f"{session.id} ({session.start[:10]}): {best[:chars]}"

    def oracle_memory(self, principal: str, t: str | datetime) -> str:
        """known_at(t) as clean markdown: the oracle baseline's memory."""
        now = when(t)
        known = self.known_at(principal, t)
        facts = [k for k in known if k.entry.kind == "fact" and k.status != "stale"]
        ended = [k for k in known if k.entry.kind == "fact" and k.status == "stale"]
        superseded = {k.previous.id for k in facts if k.previous is not None}
        lines = [f"# What you know about {self.name(principal)} (as of {now:%Y-%m-%d})", ""]
        by_subject: dict[str, list[Known]] = {}
        for k in facts:
            by_subject.setdefault(k.entry.subject, []).append(k)
        lines.append("## Facts")
        for subject in sorted(by_subject, key=lambda s: (s != principal, self.name(s))):
            lines.append(f"### {self.name(subject)}")
            for k in by_subject[subject]:
                e = k.entry
                since = f" (since {e.valid_from})" if e.valid_from else ""
                prev = f" (previously: {k.previous.text}, until {k.previous.valid_to})" if k.previous else ""
                lines.append(f"- {e.attribute}: {e.text}{since}{prev} [told {k.disclosed[-1].at[:10]}]")
        past = [k for k in ended if k.entry.id not in superseded]
        if past:
            lines += ["", "## Ended or past (no longer true)"]
            lines += [f"- {self.name(k.entry.subject)}, {k.entry.attribute}: {k.entry.text} (ended {k.entry.valid_to})"
                      for k in past]  # fmt: skip
        traits = [k for k in known if k.entry.kind == "impression" and k.status != "stale"]
        if traits:
            lines += ["", "## Patterns you have noticed (impressions; let them shape how you help, never label them)"]
            lines += [f"- {k.entry.text} (seen in {k.episodes} conversations)" for k in traits]
        refs = [k for k in known if k.entry.kind == "reference"]
        if refs:
            lines += ["", "## Shared references (what the user's shorthand means)"]
            lines += [f'- "{k.entry.attribute}": {k.entry.text}' for k in refs]
        return "\n".join(lines) + "\n"
