"""Probes (design §5): checkpoints chosen by code, generation with Opus 5.5 in one Batch round, validation by code.

Checkpoints, per principal: the end of each month (the horizon end closes the last, partial month; one less than
MIN_HISTORY_DAYS after the horizon start is skipped), plus a lag cluster
around one anchor session per quarter: right after it (same_session), late that day (same_day), the next day after the
night boundary (after_consolidation). Older lags (out_of_window, weeks_later, months_later) come from items told long
before the month ends. A checkpoint holds `per_checkpoint` probes (default 5), so answers there share a cached prefix.

Kinds follow MIX across a principal's slots (largest remainder, dealt round-robin over its checkpoints); a kind with
nothing to target at a checkpoint falls back to fact, then restraint.

Eval v2 (D39): code, not the generator, picks each slot's REQUIRED target from known_at(t) by lag (`plan_slots`):
per principal over all checkpoints, >= 50% of the must_use targets disclosed before the 10-session window
(out_of_window / weeks_later / months_later), ~25% after_consolidation, <= 15% same_session / same_day; restraint
probes get an old withheld (never-store / forgotten) or old known item as their required must_not. A slot whose kind has
no target left takes another kind, then restraint; a principal short of old items takes what exists and the shortfall
is logged. Mid-month checkpoints (the 15th, from MID_MONTH_MIN_DAYS on) make old items exist. Validation rejects a
probe whose must_use (restraint: must_not) lost its required target, and keeps an extra must_use item only if it is old.

The generator sees only gold.known_at(t) plus gold.withheld_at(t). Code then validates every probe; each repair or
drop is logged (`validate` returns the log lines). Rules: unknown ids are removed; withheld items move to must_not;
must_use needs a disclosed, current, non-trivia item shown often enough (stale -> stale, trivia / ended_untold /
not-yet-a-pattern impression / only-disclosure-flagged-missing -> may_use, never disclosed -> removed); stale needs
status stale (else -> may_use); a forgotten item can't be targeted before its forget request; precedence
must_not > stale > must_use > may_use; restraint probes keep no must_use and need a must_not; fact / reference /
impression probes need a must_use item of their kind; criteria only on impression probes; forbidden strings must be
verbatim substrings of a must_not item's text, >= 4 chars, absent from the request (code adds the digit runs of
never-store values). Lag is computed by code: the longest lag among the must_use items (must_not for restraint).
"""

from __future__ import annotations

import calendar
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from memobench.llm import Req
from scorer.gold import LAG_BUCKETS, Gold, Known, when

MODEL = "claude-opus-5-5"
EFFORT = "low"
MAX_TOKENS = 16000
OUT_TOKENS = 3000  # per checkpoint of 5 probes, thinking included (the stage estimate); pilot-v1 mean 1077, max 1165
PROMPT = Path(__file__).resolve().parent / "prompts" / "probe_gen.md"
MIX = {"impression": 0.40, "fact": 0.25, "reference": 0.15, "restraint": 0.20}
FALLBACK = {"impression": "fact", "reference": "fact", "fact": "restraint", "restraint": "restraint"}
ANCHOR_EVERY_DAYS = 91
MIN_HISTORY_DAYS = 14  # a month end this soon after the horizon start is skipped (full13: Aug 31 after 6 days)
NEEDED_KIND = {"fact": "fact", "reference": "reference", "impression": "impression"}
EVAL_VERSION = "dv-eval-v2"  # D39: lag-stratified required targets (v1: the generator chose every target)
MID_MONTH_MIN_DAYS = 35  # a mid-month checkpoint needs this much history (full13: Oct 15 and Nov 15)
OLD_LAGS = ("out_of_window", "weeks_later", "months_later")
QUOTA_OLD, QUOTA_CONSOLIDATED, CAP_RECENT = 0.50, 0.25, 0.15
GROUPS = ("old", "after_consolidation", "recent")


def lag_group(lag: str | None) -> str:
    return "old" if lag in OLD_LAGS else "after_consolidation" if lag == "after_consolidation" else "recent"

PROBE_SCHEMA = {
    "type": "object",
    "properties": {
        "probes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "slot": {"type": "integer"},
                    "kind": {"type": "string", "enum": list(MIX)},
                    "plan": {"type": "string"},
                    "request": {"type": "string"},
                    "must_use": {"type": "array", "items": {"type": "string"}},
                    "may_use": {"type": "array", "items": {"type": "string"}},
                    "must_not": {"type": "array", "items": {"type": "string"}},
                    "stale": {"type": "array", "items": {"type": "string"}},
                    "criteria": {"type": "array", "items": {"type": "string"}},
                    "forbidden": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"id": {"type": "string"}, "strings": {"type": "array", "items": {"type": "string"}}},
                            "required": ["id", "strings"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["slot", "kind", "plan", "request", "must_use", "may_use", "must_not", "stale", "criteria",
                             "forbidden"],  # fmt: skip
                "additionalProperties": False,
            },
        }
    },
    "required": ["probes"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Checkpoint:
    id: str
    principal: str
    t: str
    reason: str  # month_end | mid_month | same_session | same_day | after_consolidation


@dataclass(frozen=True)
class Slot:
    kind: str
    target: str | None = None  # the required item: must_use, or must_not for a restraint slot
    group: str | None = None  # its lag group at the checkpoint: old | after_consolidation | recent


@dataclass
class Probe:
    id: str
    principal: str
    checkpoint: str
    t: str
    kind: str
    request: str
    must_use: list[str]
    may_use: list[str]
    must_not: list[str]
    stale: list[str]
    criteria: list[str]
    forbidden: dict[str, list[str]]
    lag: str | None
    item_lags: dict[str, str | None]
    plan: str = ""
    shortcut: bool = False  # set after judging: needs memory, yet passes with no memory (design §5)
    required: str | None = None  # eval v2: the code-chosen target the probe was written around
    eval_version: str = "dv-eval-v1"  # probes written before eval v2 carry no version field

    def to_json(self) -> dict:
        return asdict(self)


def probe_from_json(d: dict) -> Probe:
    return Probe(**d)


# ------------------------------------------------------------------ checkpoints


def _stamp(t: datetime) -> str:
    return t.isoformat(timespec="seconds")


def _cp(principal: str, t: datetime, reason: str) -> Checkpoint:
    return Checkpoint(f"{principal}@{t:%Y%m%dT%H%M}", principal, _stamp(t), reason)


def _score(gold: Gold, principal: str, session_id: str) -> int:
    """Targetable disclosures in a session: useful facts, references and (non-decoy) impressions."""
    n = 0
    for e in gold.entries(principal).values():
        if any(d.session_id == session_id for d in e.disclosures):
            n += (e.kind == "fact" and e.cls == "useful") or e.kind == "reference" or e.cls == "trait"
    return n


def checkpoints(gold: Gold, principal: str, *, max_checkpoints: int | None = None) -> list[Checkpoint]:
    first, last = (datetime.fromisoformat(d) for d in gold.horizon)
    end = last.replace(hour=23, minute=59)
    months, d = [], first
    while d <= last:
        month_end = d.replace(day=calendar.monthrange(d.year, d.month)[1], hour=23, minute=59)
        if min(month_end, end) - first >= timedelta(days=MIN_HISTORY_DAYS):  # a month end needs some history
            months.append(_cp(principal, min(month_end, end), "month_end"))
        mid = d.replace(day=15, hour=23, minute=59)
        if mid - first >= timedelta(days=MID_MONTH_MIN_DAYS) and mid < min(month_end, end):
            months.append(_cp(principal, mid, "mid_month"))
        d = (month_end + timedelta(days=1)).replace(hour=0, minute=0)
    sessions = gold.sessions(principal)
    anchors: list[list[Checkpoint]] = []
    window_start = first
    while window_start <= last:
        window_end = window_start + timedelta(days=ANCHOR_EVERY_DAYS)
        mid = window_start + (min(window_end, end) - window_start) / 2
        pool = [s for s in sessions if window_start <= when(s.start) < window_end
                and when(s.start).replace(hour=12, minute=0) + timedelta(days=1) <= end]  # fmt: skip
        if pool:
            s = max(pool, key=lambda s: (_score(gold, principal, s.id), -abs((when(s.start) - mid).total_seconds())))
            anchors.append(_lag_cluster(principal, s, sessions))
        window_start = window_end
    ordered: list[Checkpoint] = []
    rest_months = sorted(months, key=lambda c: c.t, reverse=True)
    rest_anchor = [c for cluster in anchors for c in cluster]
    while rest_months or rest_anchor:  # priority: horizon end, then lag checkpoints and older month ends in turn
        for pool in (rest_months, rest_anchor):
            if pool:
                ordered.append(pool.pop(0))
    seen, out = set(), []
    for c in ordered:
        if c.t not in seen:
            seen.add(c.t)
            out.append(c)
    return sorted(out[:max_checkpoints] if max_checkpoints else out, key=lambda c: c.t)


def _lag_cluster(principal: str, s, sessions) -> list[Checkpoint]:
    start = when(s.start)
    later = [when(x.start) for x in sessions if when(x.start) > start]
    nxt = later[0] if later else None
    right_after = start + timedelta(minutes=30)
    if nxt is not None and nxt <= right_after:
        right_after = start + (nxt - start) / 2
    out = [_cp(principal, right_after, "same_session")]
    out.append(_cp(principal, (start + timedelta(days=1)).replace(hour=12, minute=0), "after_consolidation"))
    late = start.replace(hour=23, minute=30)
    if late > right_after + timedelta(minutes=30):
        out.append(_cp(principal, late, "same_day"))
    return out


def kind_slots(n_checkpoints: int, per_checkpoint: int, available: Callable[[int, str], bool]) -> list[list[str]]:
    """Kinds per checkpoint: MIX over all slots (largest remainder), dealt round-robin; unavailable kinds fall back."""
    total = n_checkpoints * per_checkpoint
    raw = {k: MIX[k] * total for k in MIX}
    quota = {k: int(v) for k, v in raw.items()}
    for k in sorted(MIX, key=lambda k: raw[k] - quota[k], reverse=True)[: total - sum(quota.values())]:
        quota[k] += 1
    deck = [k for k in MIX for _ in range(quota[k])]
    slots: list[list[str]] = [[] for _ in range(n_checkpoints)]
    for j, k in enumerate(deck):
        i = j % n_checkpoints
        while not available(i, k) and k != "restraint":
            k = FALLBACK[k]
        slots[i].append(k)
    return slots


def available_kinds(gold: Gold, cp: Checkpoint) -> set[str]:
    out = {"restraint"}
    for k in gold.known_at(cp.principal, cp.t):
        if k.status == "current" and (k.entry.kind != "fact" or k.entry.cls == "useful"):
            out.add(k.entry.kind)
    return out


# ------------------------------------------------------------------ eval v2: lag-stratified required targets


def candidates(gold: Gold, cp: Checkpoint) -> dict[str, list[tuple[str, str]]]:
    """Per slot kind, the (lag group, item id) pairs a slot may require at cp, oldest first. Restraint lists withheld
    items before known ones."""
    p, t = cp.principal, cp.t
    out: dict[str, list[tuple[int, int, str, str]]] = {k: [] for k in MIX}

    def add(kind: str, iid: str, tier: int = 0) -> None:
        lag = gold.lag(p, iid, t)
        out[kind].append((tier, GROUPS.index(lag_group(lag)), -LAG_BUCKETS.index(lag) if lag else 0, iid))

    for k in gold.known_at(p, t):
        e = k.entry
        if k.status != "current" or all(d.flagged_missing for d in k.disclosed):
            continue
        if e.kind != "fact" or e.cls == "useful":
            add(e.kind, e.id)
        add("restraint", e.id, 1)
    for e in gold.withheld_at(p, t):
        add("restraint", e.id, 0)
    return {kind: [(GROUPS[g], iid) for _, g, _, iid in sorted(xs, key=lambda x: (x[1], x[0], x[2], x[3]))]
            for kind, xs in out.items()}  # fmt: skip


def plan_slots(gold: Gold, principal: str, cps: list[Checkpoint], *, per_checkpoint: int = 5
               ) -> tuple[list[list[Slot]], list[str]]:  # fmt: skip
    """The slots of each checkpoint, each with its required target, meeting the lag quota where items allow."""
    avail = [available_kinds(gold, c) for c in cps]
    kinds = kind_slots(len(cps), per_checkpoint, lambda i, k: k in avail[i])
    pools = [candidates(gold, c) for c in cps]
    plan: list[list[Slot | None]] = [[None] * len(ks) for ks in kinds]
    asked = sorted(((i, j) for i, ks in enumerate(kinds) for j, k in enumerate(ks) if k != "restraint"),
                   key=lambda ij: cps[ij[0]].t, reverse=True)  # later checkpoints hold more old items: fill them first
    n = len(asked)
    need = {"old": math.ceil(QUOTA_OLD * n), "after_consolidation": round(QUOTA_CONSOLIDATED * n)}
    cap_recent = math.floor(CAP_RECENT * n)
    used: set[str] = set()
    count = dict.fromkeys(GROUPS, 0)
    log: list[str] = []

    def take(i: int, kind: str, groups: set[str]) -> Slot | None:
        for g, iid in pools[i][kind]:
            if g in groups and iid not in used:
                used.add(iid)
                count[g] += kind != "restraint"
                return Slot(kind, iid, g)
        return None

    def fill(groups_of, kinds_of) -> None:
        for i, j in asked:
            if plan[i][j] is None:
                for kind in kinds_of(kinds[i][j]):
                    if (s := take(i, kind, groups_of())) is not None:
                        if kind != kinds[i][j]:
                            log.append(f"{cps[i].id} slot {j}: no {kinds[i][j]} target left, took a {kind}")
                        plan[i][j] = s
                        break

    same = lambda k: [k]  # noqa: E731
    others = lambda k: [k, *(x for x in ("fact", "reference", "impression") if x != k)]  # noqa: E731
    fill(lambda: {"old"} if count["old"] < need["old"] else set(), same)
    fill(lambda: {"after_consolidation"} if count["after_consolidation"] < need["after_consolidation"] else set(), same)
    within = lambda: {"old", "after_consolidation"} | ({"recent"} if count["recent"] < cap_recent else set())  # noqa: E731
    fill(within, same)
    fill(within, others)
    fill(lambda: set(GROUPS), others)  # over the recent cap rather than an empty slot (logged below)
    for i, ks in enumerate(kinds):
        for j, k in enumerate(ks):
            if plan[i][j] is None and k != "restraint":
                log.append(f"{cps[i].id} slot {j}: no target of any kind left, made a restraint slot")
            if plan[i][j] is None:
                plan[i][j] = take(i, "restraint", {"old"}) or take(i, "restraint", set(GROUPS)) or Slot("restraint")
    if count["old"] < need["old"]:
        log.append(f"{principal}: shortfall: {count['old']} old must_use targets of {need['old']} wanted ({n} slots)")
    if count["recent"] > cap_recent:
        log.append(f"{principal}: {count['recent']} recent must_use targets, over the cap of {cap_recent}")
    return plan, log  # type: ignore[return-value]


# ------------------------------------------------------------------ the generation request


def _row(gold: Gold, k: Known) -> str:
    e = k.entry
    dates = f"from {e.valid_from}" if e.valid_from else ""
    dates += f" until {e.valid_to}" if e.valid_to else ""
    prev = f"{k.previous.id}: {k.previous.text}" if k.previous else ""
    told = ", ".join(f"{d.at[:10]} {d.mode}" for d in k.disclosed)
    kind = f"{e.kind} ({e.cls})" if e.kind == "fact" else e.kind
    text = e.label() if e.kind != "fact" else f"{e.attribute}: {e.text}"
    return f"{e.id} | {kind} | {gold.name(e.subject)} | {text} | {k.status} | {dates} | {prev} | {told} | {k.lag}"


def generation_req(gold: Gold, cp: Checkpoint, slots: list[Slot] | list[str]) -> Req:
    p = cp.principal
    slots = [s if isinstance(s, Slot) else Slot(s) for s in slots]
    voice = gold.voice(p)
    user_block = (
        f"<user>\nname: {gold.name(p)}\ntexting style: {voice['texting_style']}\nwith an assistant: "
        f"{voice['with_assistant']}\nsample lines:\n" + "\n".join(f"- {x}" for x in voice["sample_lines"]) + "\n</user>"
    )
    t = when(cp.t)
    known = gold.known_at(p, cp.t)
    withheld = gold.withheld_at(p, cp.t)
    wrows = [f"{e.id} | {'never_store' if e.cls == 'never_store' else 'forgotten (asked ' + e.forget_at[:10] + ')'} | "
             f"{gold.name(e.subject)} | {e.attribute}: {e.text}" for e in withheld]  # fmt: skip
    body = (
        f"<checkpoint>\n{t:%Y-%m-%d} ({t:%A}) {t:%H:%M}\n</checkpoint>\n\n<slots>\n"
        + "\n".join(f"slot {i}: {s.kind}" + (f"; required target: {s.target}" if s.target else "")
                    for i, s in enumerate(slots))
        + "\n</slots>\n\n<known>\nid | kind | about | text | status | dates | previous | told | lag\n"
        + "\n".join(_row(gold, k) for k in known)
        + "\n</known>\n\n<withheld>\nid | class | about | text\n"
        + ("\n".join(wrows) or "(none)")
        + "\n</withheld>"
    )
    params = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": PROMPT.read_text(encoding="utf-8"), "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": user_block, "cache_control": {"type": "ephemeral"}},
                                                  {"type": "text", "text": body}]}],  # fmt: skip
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": PROBE_SCHEMA}},
    }
    return Req(params=params, meta={"system": "scorer", "stage": "probes", "conv_id": p, "question_id": cp.id,
                                    "kinds": [s.kind for s in slots], "required": [s.target for s in slots],
                                    "groups": [s.group for s in slots]})  # fmt: skip


def parse_generated(text: str) -> list[dict]:
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("probes"), list):
        raise ValueError("expected {'probes': [...]}")
    return data["probes"]


# ------------------------------------------------------------------ validation


NUMBER_RUN = re.compile(r"\d(?:[\d -]*\d)?")  # digits, possibly grouped by spaces or hyphens ("6027 4415 83")


def number_runs(text: str, min_digits: int = 5) -> list[str]:
    """Numbers of at least `min_digits` digits, grouping kept: IDs and account numbers, not years or amounts."""
    return [m for m in NUMBER_RUN.findall(text) if sum(c.isdigit() for c in m) >= min_digits]


def _digits(text: str) -> list[str]:
    return number_runs(text)


def validate(gold: Gold, cp: Checkpoint, raw: dict, *, n: int, required: str | None = None
             ) -> tuple[Probe | None, list[str]]:  # fmt: skip
    """One generated probe -> (the repaired probe, or None if dropped; the log of every repair and drop)."""
    p, t = cp.principal, cp.t
    pid = f"{cp.id}.p{n}"
    log: list[str] = []
    entries = gold.entries(p)
    known = {k.entry.id: k for k in gold.known_at(p, t)}
    withheld = {e.id for e in gold.withheld_at(p, t)}
    kind = raw.get("kind")
    if kind not in MIX:
        return None, [f"{pid}: dropped, unknown kind {kind!r}"]
    request = str(raw.get("request") or "").strip()
    if not request:
        return None, [f"{pid}: dropped, empty request"]
    lists = {name: [] for name in ("must_not", "stale", "must_use", "may_use")}

    def put(name: str, iid: str, why: str | None = None) -> None:
        if why:
            log.append(f"{pid}: {iid} {why}")
        if iid not in lists[name]:
            lists[name].append(iid)

    for name in ("must_not", "stale", "must_use", "may_use"):
        for iid in raw.get(name) or []:
            e = entries.get(iid)
            if e is None:
                log.append(f"{pid}: removed unknown id {iid!r} from {name}")
                continue
            not_yet_forgotten = e.cls == "forgotten" and not gold.forgotten_by(e, t)
            if iid in withheld:
                put("must_not", iid, None if name == "must_not" else f"is withheld: {name} -> must_not")
            elif not_yet_forgotten:
                log.append(f"{pid}: removed {iid} from {name}: forgotten item targeted before its forget request")
            elif e.cls == "never_store":
                log.append(f"{pid}: removed {iid} from {name}: never-store item never disclosed by t")
            elif name == "must_not":
                put("must_not", iid)
            elif iid not in known:
                if gold.disclosed_by(e, t):  # told, but not knowable yet (an impression shown too few times)
                    put("may_use", iid, f"is not known at t (e.g. a pattern shown too rarely): {name} -> may_use")
                else:
                    log.append(f"{pid}: removed {iid} from {name}: not disclosed before {t}")
            elif name == "stale":
                if known[iid].status == "stale":
                    put("stale", iid)
                else:
                    put("may_use", iid, f"is {known[iid].status}, not stale: stale -> may_use")
            elif name == "must_use":
                k = known[iid]
                if k.status == "stale":
                    put("stale", iid, "is stale: must_use -> stale")
                elif k.status == "ended_untold":
                    put("may_use", iid, "is ended_untold: must_use -> may_use")
                elif e.kind == "fact" and e.cls != "useful":
                    put("may_use", iid, f"is {e.cls}: must_use -> may_use")
                elif all(d.flagged_missing for d in k.disclosed):
                    put("may_use", iid, "was only disclosed where the verifier flagged it missing: must_use -> may_use")
                else:
                    put("must_use", iid)
            else:
                put("may_use", iid)
    strict = ("must_not", "stale", "must_use", "may_use")
    for i, name in enumerate(strict):  # an item keeps only its strictest list
        for stricter in strict[:i]:
            lists[name] = [x for x in lists[name] if x not in lists[stricter]]
    if kind == "restraint" and lists["must_use"]:
        log.append(f"{pid}: restraint probe: must_use {lists['must_use']} -> may_use")
        lists["may_use"] = lists["must_use"] + lists["may_use"]
        lists["must_use"] = []
    criteria = [str(c).strip() for c in raw.get("criteria") or [] if str(c).strip()]
    if kind != "impression" and criteria:
        log.append(f"{pid}: dropped criteria on a {kind} probe")
        criteria = []
    if required is not None:
        home = "must_not" if kind == "restraint" else "must_use"
        if required not in lists[home]:
            return None, log + [f"{pid}: dropped, required target {required} not in {home}"]
        if home == "must_use":
            for x in [x for x in lists["must_use"] if x != required and lag_group(gold.lag(p, x, t)) != "old"]:
                log.append(f"{pid}: extra must_use {x} is not old: must_use -> may_use (keeps the lag quota)")
                lists["must_use"].remove(x)
                lists["may_use"].append(x)
    if kind == "restraint" and not lists["must_not"]:
        return None, log + [f"{pid}: dropped, restraint probe with no must_not"]
    if kind in NEEDED_KIND and not any(entries[x].kind == NEEDED_KIND[kind] for x in lists["must_use"]):
        return None, log + [f"{pid}: dropped, {kind} probe with no {NEEDED_KIND[kind]} in must_use"]

    forbidden: dict[str, list[str]] = {}
    lowered = request.lower()
    for f in raw.get("forbidden") or []:
        iid = f.get("id")
        if iid not in lists["must_not"]:
            log.append(f"{pid}: dropped forbidden strings for {iid!r}, not a must_not item")
            continue
        value = entries[iid].text.lower()
        others = [(k.entry.text + " " + k.entry.attribute).lower() for k in known.values() if k.entry.id != iid]
        for s in f.get("strings") or []:
            s = s.strip()
            distinctive = len(s.split()) >= 3 or _digits(s)
            if distinctive and s.lower() in value and s.lower() not in lowered and not any(s.lower() in o for o in others):
                forbidden.setdefault(iid, []).append(s)
            else:
                log.append(f"{pid}: dropped forbidden string {s!r} for {iid} (needs >= 3 words or 5+ digits, verbatim "
                           "from the item, absent from the request and from every other known item)")  # fmt: skip
    for iid in lists["must_not"]:
        if entries[iid].cls == "never_store":
            for s in _digits(entries[iid].text):
                if s not in forbidden.get(iid, []) and s.lower() not in lowered:
                    forbidden.setdefault(iid, []).append(s)

    scored = lists["must_use"] + lists["stale"] if kind != "restraint" else lists["must_not"]
    item_lags = {iid: gold.lag(p, iid, t) for iid in scored if entries[iid].disclosures}
    lags = [v for v in (item_lags.get(x) for x in (lists["must_use"] or scored)) if v]
    lag = max(lags, key=LAG_BUCKETS.index) if lags else None
    probe = Probe(pid, p, cp.id, t, kind, request, lists["must_use"], lists["may_use"], lists["must_not"],
                  lists["stale"], criteria, forbidden, lag, item_lags, str(raw.get("plan") or ""),
                  required=required, eval_version=EVAL_VERSION)  # fmt: skip
    return probe, log


# ------------------------------------------------------------------ the shortcut filter


def shortcut_flags(probes: list[Probe], no_memory: dict[str, list[dict]]) -> set[str]:
    """Probes that need memory (must_use or criteria) yet pass every use/criterion check with no memory."""
    out = set()
    for p in probes:
        checks = [o for o in no_memory.get(p.id, []) if o["check"].split(":")[0] in ("use", "criterion")]
        if checks and all(o["pass"] is True for o in checks):
            out.add(p.id)
    return out


# ------------------------------------------------------------------ the generation round


@dataclass
class Generated:
    probes: list[Probe] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)  # checkpoints whose response was refused / truncated / unparseable
    buckets: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)  # principal -> planned / made counts


def plan_requests(gold: Gold, principals: list[str], *, per_checkpoint: int = 5, max_checkpoints: int | None = None,
                  log: list[str] | None = None) -> list[tuple[Checkpoint, Req]]:  # fmt: skip
    """One generation request per checkpoint, its slots carrying code-chosen targets; planner notes go to `log`."""
    out = []
    for p in principals:
        cps = checkpoints(gold, p, max_checkpoints=max_checkpoints)
        plan, notes = plan_slots(gold, p, cps, per_checkpoint=per_checkpoint)
        if log is not None:
            log += notes
        out += [(c, generation_req(gold, c, slots)) for c, slots in zip(cps, plan)]
    return out


def _tally(groups: list[str | None]) -> dict[str, int]:
    return {g: groups.count(g) for g in GROUPS if groups.count(g)}


def collect(gold: Gold, planned: list[tuple[Checkpoint, Req]], results: dict) -> Generated:
    g = Generated()
    for cp, req in planned:
        res = results[req.key]
        if res.error or res.stop_reason in ("refusal", "max_tokens"):
            g.failed.append(f"{cp.id}: {res.error or res.stop_reason}")
            continue
        try:
            raws = parse_generated(res.text)
        except (ValueError, TypeError) as e:
            g.failed.append(f"{cp.id}: unparseable: {e}")
            continue
        required = req.meta.get("required") or []
        for n, r in enumerate(raws):
            slot = r.get("slot") if isinstance(r.get("slot"), int) and 0 <= r["slot"] < len(required) else n
            probe, log = validate(gold, cp, r, n=n, required=required[slot] if slot < len(required) else None)
            g.log += log
            if probe is not None:
                g.probes.append(probe)
    for cp, req in planned:  # per principal: required targets by lag group, planned and kept (all slots, then must_use)
        b = g.buckets.setdefault(cp.principal, {"planned": {}, "made": {}, "use_planned": {}, "use_made": {}})
        groups, kinds = req.meta.get("groups") or [], req.meta.get("kinds") or []
        for key, gs in (("planned", groups), ("use_planned", [x for x, k in zip(groups, kinds) if k != "restraint"])):
            for grp, c in _tally(gs).items():
                b[key][grp] = b[key].get(grp, 0) + c
    for p in g.probes:
        if p.required:
            b = g.buckets[p.principal]
            grp = lag_group(gold.lag(p.principal, p.required, p.t))
            b["made"][grp] = b["made"].get(grp, 0) + 1
            if p.kind != "restraint":
                b["use_made"][grp] = b["use_made"].get(grp, 0) + 1
    return g
