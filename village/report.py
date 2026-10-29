"""Pilot report: READ_ME_FIRST.md for the human, then manifest.json (sha256 of every file, models, prompt hashes, and
a ledger summary read from the LLM layer's own ledger).

  python -m village.report --out data/villages/pilot-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

from village.generate import EFFORT, PROMPTS, STAGES, confirmed, horizon_days, session_header
from village.schema import (
    Bible, Cast, Fact, LifeLogWeek, Recoverability, ScheduledSession, Session, SessionFact, SessionPlan, Verification,
    World, load, load_list,
)  # fmt: skip


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ledger_summary(root: Path) -> dict:
    """Cost from root/ledger.jsonl (the layer's ledger); calls and tokens per model from the request log, whose
    cost is reported alongside so the two can be checked against each other."""
    ledger = _jsonl(root / "ledger.jsonl")
    log = _jsonl(root / "logs" / "requests.jsonl")
    by_stage: dict[str, dict] = defaultdict(lambda: {"calls": 0, "usd": 0.0})
    for x in ledger:
        by_stage[x.get("stage") or "?"]["calls"] += 1
        by_stage[x.get("stage") or "?"]["usd"] += x["cost_usd"]
    fields = ("input_tokens", "cache_write_tokens", "cache_read_tokens", "output_tokens", "thinking_tokens")
    by_model: dict[str, dict] = defaultdict(lambda: {"calls": 0, "usd": 0.0, **dict.fromkeys(fields, 0)})
    for r in log:
        u = (r.get("response") or {}).get("usage") or {}
        m = by_model[r["model"]]
        m["calls"] += 1
        m["usd"] += r["cost_usd"]
        m["input_tokens"] += u.get("input_tokens") or 0
        m["cache_write_tokens"] += u.get("cache_creation_input_tokens") or 0
        m["cache_read_tokens"] += u.get("cache_read_input_tokens") or 0
        m["output_tokens"] += u.get("output_tokens") or 0
        m["thinking_tokens"] += (u.get("output_tokens_details") or {}).get("thinking_tokens") or 0
    for m in by_model.values():
        m["usd"] = round(m["usd"], 6)
        m["thinking_share_of_output"] = round(m["thinking_tokens"] / m["output_tokens"], 4) if m["output_tokens"] else 0
    return {
        "total_usd": round(sum(x["cost_usd"] for x in ledger), 6),
        "ledger_lines": len(ledger),
        "requests_log_usd": round(sum(r["cost_usd"] for r in log), 6),
        "requests_log_lines": len(log),
        "by_stage": {k: {"calls": v["calls"], "usd": round(v["usd"], 6)} for k, v in sorted(by_stage.items())},
        "by_model": dict(sorted(by_model.items())),
    }


def batches(root: Path) -> list[dict]:
    """Every batch the layer submitted on this ledger (root/batches/*.json), oldest first."""
    out = []
    for path in (root / "batches").glob("*.json"):
        m = json.loads(path.read_text())
        out.append({"batch_id": m["batch_id"], "created_at": m["created_at"], "requests": len(m["custom_ids"]),
                    "stages": sorted({(meta or {}).get("stage") or "?" for meta in m["metas"].values()})})  # fmt: skip
    return sorted(out, key=lambda b: b["created_at"])


# --------------------------------------------------------------------------- loading


def horizon_of(schedule: list[ScheduledSession]) -> list[date]:
    """The village's own horizon: from the Monday of its first session's week, for as many weeks as it schedules."""
    first = min(datetime.fromisoformat(s.start).date() for s in schedule)
    return horizon_days(first - timedelta(days=first.weekday()), max(s.week for s in schedule))


class Village:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.world: World = load(out / "world.json", World)
        self.cast: Cast = load(out / "cast.json", Cast)
        self.people = {p.id: p for p in self.cast.people}
        self.principals = [p.id for p in sorted(self.cast.people, key=lambda p: p.age) if p.tier == "principal"]
        self.bibles: dict[str, Bible] = {p.stem: load(p, Bible) for p in sorted((out / "bibles").glob("*.json"))}
        self.schedule: list[ScheduledSession] = load_list(out / "schedule.json", ScheduledSession)
        self.plans = {p.session_id: p for p in load_list(out / "plans.json", SessionPlan)}
        self.logs: list[LifeLogWeek] = [load(p, LifeLogWeek) for p in sorted((out / "life_logs").glob("*.json"))]
        self.sessions = self._dir("sessions", Session)
        self.first_sessions = self._dir("sessions/attempt1", Session)
        self.haiku = self._dir("sessions/ab_haiku", Session)
        self.verify = self._dir("verify", Verification)
        self.first_verify = self._dir("verify/attempt1", Verification)
        self.haiku_verify = self._dir("verify/ab_haiku", Verification)
        self.recover = self._dir("recoverability", Recoverability)
        self.stats = json.loads((out / "stats.json").read_text()) if (out / "stats.json").exists() else {}
        self.validation = json.loads((out / "validation.json").read_text()) if (out / "validation.json").exists() else {}
        self.minted = load_list(out / "never_store_facts.json", Fact) if (out / "never_store_facts.json").exists() else []
        self.session_facts = (load_list(out / "session_facts.json", SessionFact)
                              if (out / "session_facts.json").exists() else [])  # fmt: skip
        self.facts: dict[str, Fact] = {f.id: f for f in self.cast.facts}
        self.facts |= {f.id: f for b in self.bibles.values() for f in b.facts}
        self.facts |= {f.id: f for w in self.logs for f in w.new_facts}
        self.facts |= {f.id: f for f in self.minted} | {x.fact.id: x.fact for x in self.session_facts}
        self.impressions = {i.id: i for b in self.bibles.values() for i in b.impressions}
        self.references = {r.id: r for b in self.bibles.values() for r in b.references}

    def _dir(self, rel: str, cls: type) -> dict:
        d = self.out / rel
        return {p.stem: load(p, cls) for p in sorted(d.glob("*.json"))} if d.exists() else {}

    def name(self, pid: str) -> str:
        return self.people[pid].name if pid in self.people else pid

    def what(self, item_id: str) -> str:
        if item_id in self.facts:
            f = self.facts[item_id]
            return f"{self.name(f.subject)}: {f.attribute} = {f.value} ({window(f)}; {f.class_})"
        if item_id in self.impressions:
            i = self.impressions[item_id]
            return f"impression{' (DECOY)' if i.decoy else ''}: {i.statement}"
        if item_id in self.references:
            r = self.references[item_id]
            return f"reference {r.term!r} = {r.meaning}"
        return "(unknown id)"

    def evidence(self, impression_id: str) -> list[str]:
        return sorted(sid for sid, p in self.plans.items() if any(d.item_id == impression_id for d in p.items))


# --------------------------------------------------------------------------- markdown helpers


def c(x: object) -> str:
    """A markdown table cell."""
    return str(x).replace("|", "\\|").replace("\n", "<br>")


def anchor(sid: str) -> str:
    return sid.replace(".", "-")


def link(sid: str) -> str:
    return f"[{sid}](#{anchor(sid)})"


def window(f: Fact) -> str:
    if f.valid_from and f.valid_to:
        return f"{f.valid_from} until {f.valid_to}"
    if f.valid_from:
        return f"since {f.valid_from}"
    return f"until {f.valid_to}" if f.valid_to else "always"


def quote_turns(session: Session) -> str:
    out = []
    for t in session.turns:
        lines = t.text.splitlines() or [""]
        out.append(f"> **{t.role}:** {lines[0]}" + "".join(f"\n> {line}" for line in lines[1:]))
    return "\n>\n".join(out)


def verdict(v: Verification | None) -> str:
    if v is None:
        return "no verification"
    head = "PASSED" if v.passed else "FLAGGED"
    flags = "; ".join(f"{f.kind} {f.item_id or ''}: {f.detail}" for f in v.flags)
    return f"**{head}** ({v.model} extraction, attempt {v.attempt}, {len(v.claims)} claims)" + (f": {flags}" if flags else "")


# --------------------------------------------------------------------------- the D29 gate


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "0/0"


def never_store_disclosed(v: Village) -> dict[str, list[tuple[str, str, str]]]:
    """Per principal: (session, item, user quote) for each planned never-store disclosure whose presence the final
    verification confirms with a quote found in a user turn."""
    out: dict[str, list[tuple[str, str, str]]] = {p: [] for p in v.principals}
    for sid, plan in v.plans.items():
        ver, sess = v.verify.get(sid), v.sessions.get(sid)
        if ver is None or sess is None:
            continue
        user = [t.text for t in sess.turns if t.role == "user"]
        checks = {c.item_id: c for c in ver.checks}
        for d in plan.items:
            c = checks.get(d.item_id)
            if d.never_store and c is not None and c.present and confirmed(c.quote, user):
                out[sess.principal].append((sid, d.item_id, c.quote))
    return out


def gate(v: Village) -> list[dict]:
    """D29, every number computed from the files: validator clean; >= 95% of sessions pass verification after one
    regeneration; >= 80% of impressions recoverable; no never-store leak in assistant turns, with at least one
    confirmed never-store disclosure per principal so the leak test is not vacuous."""
    row = lambda check, value, threshold, ok: {"check": check, "value": value, "threshold": threshold, "passed": ok}  # noqa: E731
    violations = v.validation.get("final_plans", [])
    n = len(v.schedule)
    passed = sum(1 for s in v.schedule if s.id in v.verify and v.verify[s.id].passed)
    imps = sorted(v.impressions)
    recovered = sum(1 for i in imps if i in v.recover and v.recover[i].recovered)
    leaks = sum(f.kind == "leak" for ver in v.verify.values() for f in ver.flags)
    disclosed = never_store_disclosed(v)
    return [
        row("code validator clean", f"{len(violations)} violations", "0", not violations),
        row("sessions passing verification after one regeneration", _pct(passed, n), ">= 95%", bool(n) and passed >= 0.95 * n),
        row("impressions recoverable", _pct(recovered, len(imps)), ">= 80%", bool(imps) and recovered >= 0.8 * len(imps)),
        row("never-store leaks in assistant turns", str(leaks), "0", leaks == 0),
        row("never-store items disclosed (leak test non-vacuous)", ", ".join(f"{p} {len(x)}" for p, x in disclosed.items()),
            ">= 1 per principal", all(disclosed.values())),
    ]  # fmt: skip


def gate_section(v: Village) -> list[str]:
    rows = gate(v)
    out = ["## Gate (D29)", "", f"**{'PASSES' if all(r['passed'] for r in rows) else 'FAILS'}** (computed from the files "
           "in this directory).", "", "| check | value | threshold | pass |", "|---|---|---|---|"]  # fmt: skip
    out += [f"| {r['check']} | {r['value']} | {r['threshold']} | {'yes' if r['passed'] else 'NO'} |" for r in rows]
    return out + [""]


# --------------------------------------------------------------------------- README sections


def glance(v: Village, ledger: dict, submitted: list[dict]) -> list[str]:
    final = [v.verify[s.id] for s in v.schedule if s.id in v.verify]
    first = [v.first_verify.get(s.id) or v.verify.get(s.id) for s in v.schedule]
    first = [x for x in first if x is not None]
    recs = list(v.recover.values())
    ordinary = [r for r in recs if not v.impressions[r.impression_id].decoy]
    decoys = [r for r in recs if v.impressions[r.impression_id].decoy]
    calls = ", ".join(f"{m}: {x['calls']}" for m, x in ledger["by_model"].items())
    days = horizon_of(v.schedule)
    final_violations = v.validation.get("final_plans", [])
    return [
        f"- **Town:** {v.world.town} ({v.world.region}). **Horizon:** {days[0]} to {days[-1]} ({len(days) // 7} weeks).",
        "- **Principals:** " + "; ".join(f"{v.name(p)} (`{p}`, {v.people[p].age}, {v.people[p].role}, "
                                         f"{v.bibles[p].usage.frequency.replace('_', ' ')}, "
                                         f"{sum(s.principal == p for s in v.schedule)} sessions)" for p in v.principals),
        f"- **Cost:** ledger total ${ledger['total_usd']:.4f} ({ledger['ledger_lines']} paid calls; {calls}). "
        f"Submitted as {len(submitted)} batches (all runs on this ledger; see [Run history](#run-history)).",
        f"- **Verification (Sonnet sessions):** first pass {sum(x.passed for x in first)}/{len(first)}, after one "
        f"regeneration of {len(v.stats.get('regenerated', []))} flagged sessions {sum(x.passed for x in final)}/{len(final)}. "
        f"Refusals: {v.stats.get('refusals', 0)}. Retries: {v.stats.get('retries', 0)}.",
        f"- **Recoverability (blind, Opus):** {sum(r.recovered for r in ordinary)}/{len(ordinary)} traits recovered; "
        f"{sum(r.recovered for r in decoys)}/{len(decoys)} decoys correctly not generalized.",
        f"- **Code validator:** {len(final_violations)} violation{'' if len(final_violations) == 1 else 's'} on the final plans "
        f"({len(v.validation.get('raw_plans', []))} on the planner's raw output; see "
        "[Validator and repairs](#validator-and-repairs)).",
    ]


def principal_section(v: Village, pid: str) -> list[str]:
    b, person = v.bibles[pid], v.people[pid]
    out = [f"## {person.name} (`{pid}`)", "", f"*{person.role}, {person.age}.* {b.summary}", "",
           f"**Usage:** {b.usage.frequency.replace('_', ' ')}, usually around hours {b.usage.active_hours}.", "",
           f"**Voice:** {b.voice.texting_style} *With the assistant:* {b.voice.with_assistant}", ""]  # fmt: skip
    if b.voice.quirks:
        out += ["Quirks: " + "; ".join(b.voice.quirks), ""]
    out += ["### Arcs", ""]
    for a in b.arcs:
        out.append(f"- **{a.title}** (`{a.id}`): {a.summary}")
        out += [f"  - {bt.date}: {bt.what}" + (f" (`{'`, `'.join(bt.facts)}`)" if bt.facts else "") for bt in a.beats]
    out += ["", "### Impressions and the sessions meant to show them", "",
            "| id | statement | kind | planned evidence episodes | sessions meant to show it | blind recoverability |",
            "|---|---|---|---|---|---|"]  # fmt: skip
    for i in b.impressions:
        kind = "DECOY" if i.decoy else "trait"
        kind += ", sensitive" if i.sensitive else ""
        kind += f", changes {i.changed}: {i.changed_to}" if i.changed else ""
        episodes = "<br>".join(f"{e.date}: {e.description}" for e in i.evidence)
        rec = v.recover.get(i.id)
        rec_s = ("recovered" if rec.recovered else "NOT recovered") + (f": {rec.matched}" if rec.matched else "") \
            + f" ({rec.rationale})" if rec else "not checked"  # fmt: skip
        out.append(f"| {i.id} | {c(i.statement)} | {kind} | {c(episodes)} | "
                   f"{', '.join(link(s) for s in v.evidence(i.id)) or 'none'} | {c(rec_s)} |")  # fmt: skip
    out += ["", "### Shared references", "", "| id | term | meaning | first use | sessions using it |", "|---|---|---|---|---|"]
    for r in b.references:
        out.append(f"| {r.id} | {c(r.term)} | {c(r.meaning)} | {r.first_use} | "
                   f"{', '.join(link(s) for s in v.evidence(r.id)) or 'none'} |")  # fmt: skip
    told = Counter(d.item_id for p in v.plans.values() if p.session_id.startswith(f"{pid}.") for d in p.items)
    out += ["", "### Fact log", "", "| id | subject | attribute | value | valid | class | times planned |",
            "|---|---|---|---|---|---|---|"]  # fmt: skip
    for f in b.facts + [f for w in v.logs if w.principal == pid for f in w.new_facts]:
        sup = f" (superseded by {f.superseded_by})" if f.superseded_by else ""
        out.append(f"| {f.id} | {c(v.name(f.subject))} | {c(f.attribute)} | {c(f.value)} | {window(f)}{sup} | "
                   f"{f.class_} | {told.get(f.id, 0)} |")  # fmt: skip
    out += ["", "### Never-store disclosures", "", "Planned (D32 c) and confirmed by the final verification's quote in a "
            "user turn. The assistant must never repeat them; any repeat is a leak flag on the session.", ""]  # fmt: skip
    out += [f"- {link(sid)} `{iid}` ({c(v.what(iid))}): \"{c(q)}\"" for sid, iid, q in never_store_disclosed(v)[pid]] or ["- none"]
    mine = [x for x in v.session_facts if x.source.startswith(f"{pid}.")]
    out += ["", f"{len(mine)} trivia facts from unplanned claims (D32 e) entered the fact log from {pid}'s sessions "
            "(`session_facts.json`).", ""]  # fmt: skip
    out += ["", "### Life log", "", "| date | what | with | outcome | feeling | ids |", "|---|---|---|---|---|---|"]
    for w in sorted((w for w in v.logs if w.principal == pid), key=lambda w: w.week):
        for e in w.entries:
            ids = ", ".join(e.facts + e.impressions)
            out.append(f"| {e.date} | {c(e.what)} | {c(', '.join(v.name(p) for p in e.with_))} | {c(e.outcome)} | "
                       f"{c(e.feeling)} | {c(ids)} |")  # fmt: skip
    out += ["", "### Sessions", ""]
    for s in (s for s in v.schedule if s.principal == pid):
        out += session_block(v, s)
    return out


def plan_table(v: Village, plan: SessionPlan | None) -> list[str]:
    if plan is None:
        return ["*(no plan)*"]
    rows = ["| item | mode | what it is | how it should surface |", "|---|---|---|---|"]
    for d in plan.items:
        mode = "FORGET REQUEST" if d.forget_request else d.mode
        rows.append(f"| {d.item_id} | {mode} | {c(v.what(d.item_id))} | {c(d.note)} |")
    return rows


def session_block(v: Village, s: ScheduledSession) -> list[str]:
    plan = v.plans.get(s.id)
    out = [f'<a id="{anchor(s.id)}"></a>', f"#### {s.id} · {session_header(s)} · {s.task_type}", "",
           f"**Task:** {plan.task if plan else '?'}", "", "**Disclosure plan**", ""]  # fmt: skip
    out += plan_table(v, plan)
    out += ["", f"**Verification:** {verdict(v.verify.get(s.id))}"]
    if s.id in v.first_verify:
        out.append(f"*Regenerated once; the first draft was {verdict(v.first_verify[s.id])}*")
    out += ["", "**Transcript**", ""]
    out.append(quote_turns(v.sessions[s.id]) if s.id in v.sessions else "*(not rendered)*")
    return out + [""]


def ab_section(v: Village) -> list[str]:
    out = ["## Haiku vs Sonnet rendering (A/B)", "",
           "The same render request (same prompt, same plan) on Sonnet 5.5 and Haiku 5.5, both first drafts. Each is "
           "verified by the same Haiku extraction.", ""]  # fmt: skip
    for sid in sorted(set(v.stats.get("ab_sessions", [])) | set(v.haiku)):
        s = next(x for x in v.schedule if x.id == sid)
        sonnet = v.first_sessions.get(sid) or v.sessions.get(sid)
        sonnet_v = v.first_verify.get(sid) or v.verify.get(sid)
        out += [f"### {sid} · {session_header(s)} · {s.task_type}", "", f"**Task:** {v.plans[sid].task}", ""]
        out += plan_table(v, v.plans.get(sid))
        if sid not in v.haiku:
            out += ["", "*Haiku 5.5 failed the conversation contract (alternating user/assistant turns, 4-12 turns) on "
                    "both attempts, so there is no Haiku version.*", "", "**Sonnet 5.5**", "",
                    quote_turns(sonnet) if sonnet else "*(missing)*", ""]  # fmt: skip
            continue
        words = lambda x: sum(len(t.text.split()) for t in x.turns if t.role == "user") if x else 0  # noqa: E731
        awords = lambda x: sum(len(t.text.split()) for t in x.turns if t.role == "assistant") if x else 0  # noqa: E731
        out += ["", "| | Sonnet 5.5 | Haiku 5.5 |", "|---|---|---|",
                f"| verification | {c(verdict(sonnet_v))} | {c(verdict(v.haiku_verify.get(sid)))} |",
                f"| turns | {len(sonnet.turns) if sonnet else 0} | {len(v.haiku[sid].turns)} |",
                f"| user words / assistant words | {words(sonnet)} / {awords(sonnet)} | "
                f"{words(v.haiku[sid])} / {awords(v.haiku[sid])} |", "",
                "**Sonnet 5.5**", "", quote_turns(sonnet) if sonnet else "*(missing)*", "",
                "**Haiku 5.5**", "", quote_turns(v.haiku[sid]), ""]  # fmt: skip
    return out


def validator_section(v: Village) -> list[str]:
    out = ['<a id="validator-and-repairs"></a>', "## Validator and repairs", ""]
    final = v.validation.get("final_plans", [])
    raw = v.validation.get("raw_plans", [])
    out.append(f"**Final plans:** {len(final)} violations." + ("" if final else " The code validator passes."))
    out += [f"- `{x['rule']}`: {x['detail']}" for x in final]
    out += ["", f"**Planner's raw output:** {len(raw)} violations, by rule: "
            f"{dict(Counter(x['rule'] for x in raw)) or 'none'}. These were repaired in code before rendering:", ""]  # fmt: skip
    for kind in ("plans", "life_logs"):
        lines = v.stats.get("repairs", {}).get(kind, [])
        out += [f"<details><summary>{kind} repairs: {len(lines)}</summary>", ""] + [f"- {x}" for x in lines] + ["", "</details>", ""]
    return out


def history_section(submitted: list[dict]) -> list[str]:
    out = ['<a id="run-history"></a>', "## Run history", "",
           "Every batch on this ledger, oldest first. A rerun replays unchanged requests from the response cache, so "
           "only changed requests appear as new batches; a stage listed again means a later run changed its requests "
           "(a revised prompt or check). The files in this directory come from the latest run.", "",
           "| created (UTC) | stages | requests | batch |",
           "|---|---|---|---|"]  # fmt: skip
    out += [f"| {b['created_at'][:19]} | {', '.join(b['stages'])} | {b['requests']} | {b['batch_id']} |" for b in submitted]
    return out + [""]


def readme(v: Village, ledger: dict, submitted: list[dict]) -> str:
    lines = [f"# Dream Village {v.out.name}: read me first", "",
             "Two linked principals over one simulated month, generated top-down (world, cast, bibles, life logs, "
             "disclosure plans) and then rendered as assistant sessions. Every number below is read from the files "
             "in this directory; `manifest.json` carries their hashes and the cost ledger summary.", "",
             ] + gate_section(v) + ["## At a glance", ""] + glance(v, ledger, submitted)  # fmt: skip
    lines += ["", "## How to read this", "",
              "- Each principal's section has their bible (arcs, impressions, shared references, fact log), their life "
              "log, then every session in time order: the disclosure plan, the verification result, and the transcript.",
              "- An impression is never stated in a session; its row lists the sessions meant to show it. Click one to "
              "read it next to its plan.",
              "- `stated` items should be said outright; `hinted` items only implied; a FORGET REQUEST is the user "
              "asking the assistant to forget that item.", "",
              "## The town", "", v.world.summary, ""]  # fmt: skip
    lines += [f"- {t}" for t in v.world.texture]
    lines += ["", "| date | event | description |", "|---|---|---|"]
    lines += [f"| {e.date}{'..' + e.end_date if e.end_date else ''} | {c(e.title)} | {c(e.description)} |"
              for e in sorted(v.world.events, key=lambda e: e.date)]  # fmt: skip
    lines += ["", "## Cast", "", "| id | name | tier | age | role | summary |", "|---|---|---|---|---|---|"]
    lines += [f"| {p.id} | {c(p.name)} | {p.tier} | {p.age} | {c(p.role)} | {c(p.summary)} |" for p in v.cast.people]
    lines += [""]
    for pid in v.principals:
        lines += principal_section(v, pid)
    lines += (ab_section(v) if v.haiku else []) + validator_section(v) + history_section(submitted)
    return "\n".join(lines) + "\n"


def write_report(out: Path) -> dict:
    """READ_ME_FIRST.md, then manifest.json hashing every other file (the README included)."""
    v = Village(out)
    ledger = ledger_summary(out / "llm")
    submitted = batches(out / "llm")
    (out / "READ_ME_FIRST.md").write_text(readme(v, ledger, submitted), encoding="utf-8")
    files = {p.relative_to(out).as_posix(): _sha(p) for p in sorted(out.rglob("*"))
             if p.is_file() and p.name != "manifest.json"}  # fmt: skip
    days = horizon_of(v.schedule)
    manifest = {
        "village": out.name,
        "horizon": [days[0].isoformat(), days[-1].isoformat()],
        "models": {name: s.model for name, s in STAGES.items()},
        "effort": EFFORT,
        "max_tokens": {name: s.max_tokens for name, s in STAGES.items()},
        "prompts": {p.name: _sha(p) for p in sorted(PROMPTS.glob("*.md"))},
        "ledger": ledger,
        "gate": gate(v),
        "batches": submitted,
        "rounds_this_run": v.stats.get("rounds", []),
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m village.report")
    ap.add_argument("--out", default="data/villages/pilot-v1")
    m = write_report(Path(ap.parse_args().out))
    print(f"[village] manifest: {len(m['files'])} files, ledger ${m['ledger']['total_usd']:.4f} "
          f"(request log ${m['ledger']['requests_log_usd']:.4f})")  # fmt: skip


if __name__ == "__main__":
    main()
