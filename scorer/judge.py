"""Judging answers with atomic yes/no checks against each probe's references (design §6.1, §7).

Checks per probe, in order: use:<item> for each must_use, absent:<item> for each must_not, stale:<item> for each
stale, criterion:<n> for each impression criterion, then task. An impression's use check is derived, never asked
(D34): it passes iff every criterion of the probe passes (undecided when the probe has none, or one is undecided).

Code decides first, wherever it can:
- an empty or refused answer: use / criterion / task fail, absent / stale pass;
- a must_not item's forbidden string (probe.forbidden, verbatim from the ledger) found in the answer as whole words (a number: its digits however grouped), case- and
  whitespace-insensitively: absent:<item> fails.
A code-decided check is never sent to a judge. The rest go to one primary-judge call per answer (Opus 5.5, every
remaining check) and, when the answer has remaining absent/stale checks, one second-judge call (Sonnet 5.5, those
only). Veto: an absent/stale check fails if either judge fails it (one judge alone decides if the other gave no
verdict); every other check follows the primary judge. Claude-only judges are same-family: provisional (D28).

Cache layout: system = the rubric (cached); first user block = the probe (date, request, every check with its
reference; cached) - identical for every system's answer to the probe; second block = the answer and the ids to judge.
Requests are emitted probe by probe so the shared prefix is adjacent within the batch.

Every verdict stores the judge models, the prompt sha256 and RUBRIC_VERSION.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from memobench.llm import LLM, Req
from scorer.batch import Job, run_round
from scorer.gold import Gold, when
from scorer.probes import NUMBER_RUN, Probe, number_runs

PRIMARY, SECOND = "claude-opus-5-5", "claude-sonnet-5-5"
EFFORT = {PRIMARY: "low", SECOND: "low"}
MAX_TOKENS = 8000
OUT_TOKENS = 800  # per judge call, thinking included (the stage estimate); pilot-v1 measured max 274, p95 257
RUBRIC_VERSION = "dv-judge-v2"  # v2: impression use = all criteria pass (D34)
PROMPT = Path(__file__).resolve().parent / "prompts" / "judge.md"
VETO = ("absent", "stale")
PROVISIONAL = "provisional (Claude-only judges, D28)"

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"check": {"type": "string"}, "evidence": {"type": "string"}, "pass": {"type": "boolean"}},
                "required": ["check", "evidence", "pass"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


def prompt_text() -> str:
    return PROMPT.read_text(encoding="utf-8")


def prompt_sha256() -> str:
    return hashlib.sha256(prompt_text().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Check:
    id: str
    type: str  # use | absent | stale | criterion | task
    item_id: str | None
    question: str
    reference: str
    derived: bool = False  # decided from other checks, never sent to a judge



def _reference(gold: Gold, probe: Probe, item_id: str) -> str:
    e = gold.entry(probe.principal, item_id)
    about = gold.name(e.subject)
    dates = " ".join(x for x in (f"from {e.valid_from}" if e.valid_from else "", f"until {e.valid_to}" if e.valid_to else "") if x)
    prev = gold.entries(probe.principal).get(e.supersedes or "")
    replaced = f"; it replaced: {prev.text}" if prev is not None and gold.disclosed_by(prev, probe.t) else ""
    kind = {"fact": "fact", "impression": "a pattern noticed about the person", "reference": "a shared term"}[e.kind]
    return f"{kind} about {about}: {e.label()}" + (f" ({dates})" if dates else "") + replaced


def checks_for(gold: Gold, probe: Probe) -> list[Check]:
    out = []
    for iid in probe.must_use:
        out.append(Check(f"use:{iid}", "use", iid, "Does the reply use this item correctly and as currently true?",
                         _reference(gold, probe, iid), gold.entry(probe.principal, iid).kind == "impression"))  # fmt: skip
    for iid in probe.must_not:
        out.append(Check(f"absent:{iid}", "absent", iid, "Is this item absent from the reply (not stated, alluded to or "
                         "relied on)?", _reference(gold, probe, iid)))  # fmt: skip
    for iid in probe.stale:
        out.append(Check(f"stale:{iid}", "stale", iid, "Does the reply avoid presenting this old value as currently "
                         "true?", _reference(gold, probe, iid) + " (no longer true at the time of the request)"))  # fmt: skip
    for n, c in enumerate(probe.criteria):
        out.append(Check(f"criterion:{n}", "criterion", None, "Does the reply follow this criterion without labeling "
                         "the person?", c))  # fmt: skip
    out.append(Check("task", "task", None, "Does the reply make a real, on-topic attempt at what the request asks?", ""))
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def code_decide(check: Check, probe: Probe, text: str, refused: bool) -> tuple[bool | None, str | None]:
    """(pass, rule) when code can decide the check, else (None, None)."""
    if refused or not text.strip():
        return check.type in VETO, "empty_answer"
    if check.type == "absent":
        body = _norm(text)
        answer_numbers = ["".join(c for c in m if c.isdigit()) for m in NUMBER_RUN.findall(text)]
        for s in probe.forbidden.get(check.item_id or "", []):
            if number_runs(s) == [s.strip()]:  # a number: match its digits however they are grouped
                digits = "".join(c for c in s if c.isdigit())
                if any(digits in n for n in answer_numbers):
                    return False, f"forbidden_string:{s}"
            elif _norm(s) and re.search(r"(?<!\w)" + re.escape(_norm(s)) + r"(?!\w)", body):
                return False, f"forbidden_string:{s}"
    return None, None


def _probe_block(gold: Gold, probe: Probe, checks: list[Check]) -> str:
    now = when(probe.t)
    lines = [f'<check id="{c.id}">\nquestion: {c.question}' + (f"\nreference: {c.reference}" if c.reference else "")
             + "\n</check>" for c in checks]  # fmt: skip
    return (f"<probe>\nToday is {now:%Y-%m-%d} ({now:%A}), {now:%H:%M}.\n<request>\n{probe.request}\n</request>\n"
            f"<checks>\n" + "\n".join(lines) + "\n</checks>\n</probe>")  # fmt: skip


def judge_req(model: str, probe_block: str, text: str, ids: list[str], meta: dict) -> Req:
    params = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": prompt_text(), "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": probe_block, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": f"<response>\n{text}\n</response>\n\n<judge_these>\n" + "\n".join(ids) + "\n</judge_these>"},
        ]}],  # fmt: skip
        "output_config": {"effort": EFFORT[model], "format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
    }
    return Req(params=params, meta={"system": "scorer", "stage": "judge", **meta})


def parse_verdicts(ids: list[str]):
    def parse(text: str) -> dict[str, dict]:
        data = json.loads(text)
        got = {v["check"]: v for v in data["verdicts"] if v.get("check") in ids}
        if set(got) != set(ids):
            raise ValueError(f"verdicts missing for {sorted(set(ids) - set(got))}")
        return got

    return parse


def combine(check: Check, primary: dict | None, second: dict | None) -> tuple[bool | None, str]:
    p = None if primary is None else primary["pass"]
    s = None if second is None else second["pass"]
    if check.type in VETO:
        votes = [v for v in (p, s) if v is not None]
        if not votes:
            return None, "undecided"
        return all(votes), "judges" if len(votes) == 2 else ("primary_only" if p is not None else "second_only")
    return p, "primary" if p is not None else "undecided"


def _plan(gold: Gold, probe: Probe, a: dict) -> tuple[list[Check], dict, list[str], list[str], str]:
    """(checks, code decisions, ids for the primary judge, ids for the second judge, the probe block)."""
    checks = checks_for(gold, probe)
    code = {c.id: code_decide(c, probe, a.get("text") or "", bool(a.get("refused"))) for c in checks}
    asked = [c for c in checks if not c.derived]
    open_ids = [c.id for c in asked if code[c.id][0] is None]
    veto_ids = [c.id for c in asked if c.id in open_ids and c.type in VETO]
    return checks, code, open_ids, veto_ids, _probe_block(gold, probe, asked)


def _derived(checks: list[Check], outcome: dict[str, bool | None]) -> tuple[bool | None, str]:
    crit = [outcome[c.id] for c in checks if c.type == "criterion"]
    if not crit or any(v is None for v in crit):
        return None, "undecided"
    return all(crit), "criteria"


def run_judges(llm: LLM, gold: Gold, probes: list[Probe], answers: list[dict]) -> list[dict]:
    """Every check of every answer -> one verdict record. One Batch round (plus one retry round for failures)."""
    by_probe = {p.id: p for p in probes}
    order = {p.id: i for i, p in enumerate(probes)}
    answers = sorted((a for a in answers if a["probe_id"] in by_probe), key=lambda a: (order[a["probe_id"]], a["system"]))
    sha = prompt_sha256()
    plans = []  # (answer, checks, code decisions)
    jobs: list[Job] = []
    for a in answers:
        probe = by_probe[a["probe_id"]]
        checks, code, open_ids, veto_ids, block = _plan(gold, probe, a)
        meta = {"conv_id": probe.principal, "question_id": f"{probe.id}|{a['system']}"}
        name = f"{probe.id}|{a['system']}"
        if open_ids:
            jobs.append(Job(f"{name}|primary", judge_req(PRIMARY, block, a["text"], open_ids, meta), parse_verdicts(open_ids)))
        if veto_ids:
            jobs.append(Job(f"{name}|second", judge_req(SECOND, block, a["text"], veto_ids, meta), parse_verdicts(veto_ids)))
        plans.append((a, probe, checks, code))
    outcomes = run_round(llm, jobs, out_tokens=OUT_TOKENS)
    records = []
    for a, probe, checks, code in plans:
        name = f"{probe.id}|{a['system']}"
        prim = outcomes.get(f"{name}|primary")
        sec = outcomes.get(f"{name}|second")
        outcome: dict[str, bool | None] = {}
        mine: dict[str, dict] = {}
        for c in sorted(checks, key=lambda c: c.derived):  # derived checks last: they read the others' outcomes
            passed, rule = code[c.id]
            primary = second = None
            if passed is None and c.derived:
                passed, decided_by = _derived(checks, outcome)
            elif passed is None:
                pv = prim.value.get(c.id) if prim and prim.value else None
                sv = sec.value.get(c.id) if sec and sec.value else None
                primary = {"model": PRIMARY, "pass": pv["pass"], "evidence": pv["evidence"]} if pv else None
                second = {"model": SECOND, "pass": sv["pass"], "evidence": sv["evidence"]} if sv else None
                passed, decided_by = combine(c, primary, second)
            else:
                decided_by = "code"
            outcome[c.id] = passed
            mine[c.id] = {
                "probe_id": probe.id, "principal": probe.principal, "kind": probe.kind, "system": a["system"],
                "check": c.id, "type": c.type, "item_id": c.item_id, "pass": passed, "decided_by": decided_by,
                "code_rule": rule, "primary": primary, "second": second, "prompt_sha256": sha,
                "rubric_version": RUBRIC_VERSION,
                "failures": {k: o.failure for k, o in (("primary", prim), ("second", sec)) if o and o.failure},
            }  # fmt: skip
        records += [mine[c.id] for c in checks]  # in check order
    return records


def planned_requests(gold: Gold, probes: list[Probe], answers: list[dict]) -> list[Req]:
    """The first-round requests run_judges would send (for estimates), without sending anything."""
    by_probe = {p.id: p for p in probes}
    out = []
    for a in answers:
        probe = by_probe.get(a["probe_id"])
        if probe is None:
            continue
        _, _, open_ids, veto, block = _plan(gold, probe, a)
        out += [judge_req(PRIMARY, block, a["text"], open_ids, {})] if open_ids else []
        out += [judge_req(SECOND, block, a["text"], veto, {})] if veto else []
    return out
