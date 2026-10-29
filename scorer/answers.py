"""Answers for each (probe, system) pair, by the same Sonnet 5.5 answerer (design §6.1, §10 step 2).

Systems:
- v0: the memory checked out at t (memsys.replay.checkout) and memsys.answer's own state machine (its prompt, tools,
  injected profile/preferences/listing, 10-session recent window). Its model and effort must equal MODEL / EFFORT.
- no_memory: the date and the request alone.
- recent_window: the last WINDOW sessions before t, then the date and the request.
- oracle: gold.oracle_memory(t) + the recent window + the date and the request.
- full_context: every session before t + the date and the request; run only on a chosen subset (cost).
Baselines share one prompt (prompts/answer_baseline.md, the v0 answer rules without tools). The date line is given to
every system: it is not memory, and time effects need it.

Driving: every pair is in flight together; each Batch round sends the next request of every unfinished pair (a
baseline needs one, v0 one per tool hop), so a round is one hop. The first user block (context) carries a cache
breakpoint, so the answers at one checkpoint share their prefix (for v0 the harness adds that breakpoint to
memsys.answer's request; caching never changes the output).

Each answer records its text, refusal, hops, cost, summed usage, `context_tokens` (the last hop's prompt: input +
cache writes + cache reads) and `prompt_tokens_total` (all hops).
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from memobench.llm import LLM, USAGE_FIELDS, Req, Result
from memsys import answer as v0_answer
from memsys.archive import before, transcript
from memsys.config import V0, load
from memsys.replay import checkout
from scorer.batch import upper_usd
from scorer.gold import RECENT_WINDOW, Gold, when
from scorer.probes import Probe

SYSTEMS = ("v0", "no_memory", "recent_window", "oracle", "full_context")
BASELINES = SYSTEMS[1:]
MODEL = "claude-sonnet-5-5"
EFFORT = "medium"
MAX_TOKENS = 8000
OUT_TOKENS = 1500  # per answer hop, thinking included (the stage estimate); pilot-v1 measured mean 656, p95 1654
WINDOW = RECENT_WINDOW
POLL_S = 20
PROMPT = Path(__file__).resolve().parent / "prompts" / "answer_baseline.md"


def _prompt_tokens(usage: dict) -> int:
    return sum((usage or {}).get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))


def _context(gold: Gold, system: str, probe: Probe) -> str:
    now = when(probe.t)
    past = before(gold.sessions(probe.principal), probe.t)
    parts = []
    if system == "oracle":
        parts.append(f"<memory>\n{gold.oracle_memory(probe.principal, probe.t).rstrip()}\n</memory>")
    if system in ("recent_window", "oracle", "full_context"):
        shown = past if system == "full_context" else past[-WINDOW:] if WINDOW else []
        parts.append("<recent_sessions>\n" + ("\n".join(transcript(s) for s in shown) or "(none)") + "\n</recent_sessions>")
    parts.append(f"Today is {now:%Y-%m-%d} ({now:%A}), {now:%H:%M}.")
    return "\n\n".join(parts)


def baseline_req(gold: Gold, system: str, probe: Probe, *, salt: str | None = None) -> Req:
    if system not in BASELINES:
        raise ValueError(f"{system!r} is not a baseline")
    params = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": PROMPT.read_text(encoding="utf-8"), "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": _context(gold, system, probe), "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": f"<request>\n{probe.request}\n</request>"},
        ]}],  # fmt: skip
        "output_config": {"effort": EFFORT},
    }
    if salt is not None:
        params["metadata"] = {"user_id": f"answer-{salt}"}
    return Req(params=params, meta={"system": f"scorer-{system}", "stage": "answer", "conv_id": probe.principal,
                                    "question_id": probe.id})  # fmt: skip


def stratified_subset(probes: list[Probe], per_stratum: int) -> set[str]:
    """Up to `per_stratum` probes per (principal, kind), picked by a hash of the id (deterministic)."""
    strata: dict[tuple[str, str], list[str]] = {}
    for p in probes:
        strata.setdefault((p.principal, p.kind), []).append(p.id)
    rank = lambda pid: hashlib.sha256(pid.encode("utf-8")).hexdigest()  # noqa: E731
    return {pid for ids in strata.values() for pid in sorted(ids, key=rank)[:per_stratum]}


@dataclass
class _Pair:
    probe: Probe
    system: str
    chat: object | None = None  # memsys loop.Chat for v0
    req: Req | None = None  # the single request of a baseline
    tmp: Path | None = None
    hop_usage: list[dict] = field(default_factory=list)
    cost_usd: float = 0.0
    result: Result | None = None
    record: dict | None = None


def _v0_system(config: Path | str):
    system = load(config)
    model, effort = system.setting("models", "answer"), system.setting("effort", "answer")
    if (model, effort) != (MODEL, EFFORT):
        raise ValueError(f"v0 answers with {model}/{effort}; every system must use the same answerer {MODEL}/{EFFORT}")
    return system


def _start(gold: Gold, p: Probe, system: str, *, replay, v0, salt) -> _Pair:
    pair = _Pair(p, system)
    if system == "v0":
        if replay is None:
            raise ValueError("system v0 needs --replay")
        pair.tmp = checkout(replay, p.principal, p.t)
        pair.chat = v0_answer.answer_state(v0, pair.tmp, gold.sessions(p.principal), p.request, p.t, salt=salt)
        pair.chat.messages[0]["content"][0]["cache_control"] = {"type": "ephemeral"}
        pair.chat.meta = {**pair.chat.meta, "question_id": p.id}
    else:
        pair.req = baseline_req(gold, system, p, salt=salt)
    return pair


def _finish(pair: _Pair, final: dict) -> None:
    usage = {k: sum((u or {}).get(k) or 0 for u in pair.hop_usage) for k in USAGE_FIELDS}
    pair.record = {
        "probe_id": pair.probe.id, "principal": pair.probe.principal, "kind": pair.probe.kind, "system": pair.system,
        **final, "cost_usd": pair.cost_usd, "usage": usage,
        "context_tokens": _prompt_tokens(pair.hop_usage[-1]) if pair.hop_usage else 0,
        "prompt_tokens_total": sum(_prompt_tokens(u) for u in pair.hop_usage),
    }  # fmt: skip
    if pair.tmp is not None:
        shutil.rmtree(pair.tmp, ignore_errors=True)
        pair.tmp = None


def _next(pair: _Pair) -> Req | None:
    """The pair's next request, or None after finishing it."""
    if pair.chat is not None:
        x = v0_answer.step(pair.chat)
        if isinstance(x, v0_answer.Final):
            _finish(pair, {"text": x.text, "refused": x.refused, "stop_reason": x.stop_reason, "hops": x.hops,
                           "error": x.error, "tool_calls": [c["name"] for c in x.calls]})  # fmt: skip
            return None
        return x
    if pair.result is None:
        return pair.req
    r = pair.result
    refused = r.stop_reason == "refusal"
    _finish(pair, {"text": "" if refused else r.text.strip(), "refused": refused, "stop_reason": r.stop_reason,
                   "hops": 1, "error": r.error, "tool_calls": []})  # fmt: skip
    return None


def run_answers(
    llm: LLM,
    gold: Gold,
    probes: list[Probe],
    systems: list[str],
    *,
    replay: Path | str | None = None,
    config: Path | str = V0,
    full_context_ids: set[str] | None = None,
    salt: str | None = None,
) -> list[dict]:
    v0 = _v0_system(config) if "v0" in systems else None
    pairs = [_start(gold, p, s, replay=replay, v0=v0, salt=salt) for p in probes for s in systems
             if s != "full_context" or p.id in (full_context_ids or set())]  # fmt: skip
    try:
        while True:
            todo = [(pair, req) for pair in pairs if pair.record is None and (req := _next(pair)) is not None]
            if not todo:
                break
            reqs = [req for _, req in todo]
            results = llm.run(reqs, mode="batch", poll_s=POLL_S, estimate_usd=upper_usd(llm, reqs, OUT_TOKENS))
            for pair, req in todo:
                res = results[req.key]
                pair.hop_usage.append(res.usage or {})
                pair.cost_usd += res.cost_usd
                if pair.chat is not None:
                    v0_answer.feed(pair.chat, res)
                else:
                    pair.result = res
    finally:
        for pair in pairs:
            if pair.tmp is not None:
                shutil.rmtree(pair.tmp, ignore_errors=True)
    return [pair.record for pair in pairs]


def first_round(gold: Gold, probes: list[Probe], systems: list[str], *, replay=None, config: Path | str = V0,
                full_context_ids: set[str] | None = None, salt: str | None = None) -> list[Req]:  # fmt: skip
    """The requests of the first round (for LLM.estimate), without sending anything."""
    v0 = _v0_system(config) if "v0" in systems else None
    out = []
    for p in probes:
        for s in systems:
            if s == "full_context" and p.id not in (full_context_ids or set()):
                continue
            pair = _start(gold, p, s, replay=replay, v0=v0, salt=salt)
            out.append(_next(pair) if pair.chat is not None else pair.req)
            if pair.tmp is not None:
                shutil.rmtree(pair.tmp, ignore_errors=True)
    return out
