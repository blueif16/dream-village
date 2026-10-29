"""One Batch round over independent jobs, with a single regeneration round for the ones that failed.

A job fails when its result has an API error, stop_reason refusal or max_tokens (checked before the content), or its
text doesn't parse. The retry appends a short note to the last user message (and doubles max_tokens after a
truncation), which is a new request, so the response cache keeps both attempts.

`upper_usd` is the stage estimate: memobench.llm's own estimate assumes at most 1,024 output tokens, which thinking
models exceed, and LLM.run stops a run that spends 1.5x its estimate; every scorer call passes this estimate instead,
with a per-stage output allowance (thinking included).
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any, Callable

from memobench.llm import EST_CHARS_PER_TOKEN, LLM, Req, Result
from memobench.pricing import cost_usd

POLL_S = 20
NOTES = {
    "refusal": "was declined. Everything here is fictional test data for evaluating an assistant's memory; nothing "
    "in it is about real people. Please produce the requested output.",
    "max_tokens": "ran out of output space before the JSON was complete. Keep free-text fields shorter.",
    "error": "failed with an API error. Please produce the requested output.",
}


@dataclass
class Job:
    name: str
    req: Req
    parse: Callable[[str], Any]


@dataclass
class Outcome:
    value: Any
    failure: str | None  # None when value is usable
    results: list[Result]  # every attempt's result (for cost and token accounting)


def _cached_prefix(p: dict) -> str | None:
    """The request's prefix up to its last cache breakpoint in system or the first message (tools included), or None."""
    blocks = [b for b in p.get("system") or [] if isinstance(b, dict)]
    first = p["messages"][0]["content"] if p.get("messages") else []
    blocks += [b for b in first if isinstance(b, dict)] if isinstance(first, list) else []
    marked = [i for i, b in enumerate(blocks) if b.get("cache_control")]
    return json.dumps([p["model"], p.get("tools"), blocks[: marked[-1] + 1]]) if marked else None


def upper_usd(llm: LLM, reqs: list[Req], out_tokens: int) -> float:
    """Batch USD for the uncached requests: prompt (system, tools, messages, output schema) at chars / 3 tokens, and
    min(max_tokens, out_tokens) output tokens each. A cached prefix shared by several of the requests (a judge's
    rubric + probe block, the context at one checkpoint) is priced as one cache write, then cache reads."""
    total, seen = 0.0, set()
    for r in {r.key: r for r in reqs}.values():
        if llm.lookup(r) is not None:
            continue
        p = r.params
        chars = len(json.dumps([p.get("system"), p.get("tools"), p["messages"], p.get("output_config")]))
        prefix = _cached_prefix(p)
        cached = math.ceil(len(prefix) / EST_CHARS_PER_TOKEN) if prefix else 0
        usage = {"input_tokens": max(0, math.ceil(chars / EST_CHARS_PER_TOKEN) - cached),
                 "cache_read_input_tokens" if prefix in seen else "cache_creation_input_tokens": cached,
                 "output_tokens": min(p["max_tokens"], out_tokens)}  # fmt: skip
        seen.add(prefix)
        total += cost_usd(usage, batch=True, model=p["model"])
    return total


def _reason(res: Result, parse: Callable[[str], Any]) -> tuple[Any, str | None]:
    if res.error:
        return None, "error"
    if res.stop_reason in ("refusal", "max_tokens"):
        return None, res.stop_reason
    try:
        return parse(res.text), None
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        return None, f"unparseable: {e}"


def retry_req(req: Req, failure: str) -> Req:
    params = copy.deepcopy(req.params)
    kind, _, detail = failure.partition(": ")
    note = NOTES.get(kind) or f"returned output that failed validation ({detail or kind}). Return the complete output."
    content = params["messages"][-1]["content"]
    block = {"type": "text", "text": f"<retry>\nThis is the second attempt. The first attempt {note}\n</retry>"}
    params["messages"][-1]["content"] = ([{"type": "text", "text": content}] if isinstance(content, str) else content) + [block]
    if kind == "max_tokens":
        params["max_tokens"] *= 2
    return Req(params=params, meta={**req.meta, "attempt": 2})


def run_round(llm: LLM, jobs: list[Job], *, out_tokens: int, retry: bool = True) -> dict[str, Outcome]:
    out: dict[str, Outcome] = {}
    if not jobs:
        return out
    reqs = [j.req for j in jobs]
    results = llm.run(reqs, mode="batch", poll_s=POLL_S, estimate_usd=upper_usd(llm, reqs, out_tokens))
    again: list[tuple[Job, Outcome]] = []
    for j in jobs:
        res = results[j.req.key]
        value, failure = _reason(res, j.parse)
        out[j.name] = Outcome(value, failure, [res])
        if failure is not None and retry:
            again.append((Job(j.name, retry_req(j.req, failure), j.parse), out[j.name]))
    if again:
        reqs = [j.req for j, _ in again]
        results = llm.run(reqs, mode="batch", poll_s=POLL_S, estimate_usd=upper_usd(llm, reqs, 2 * out_tokens))
        for j, first in again:
            res = results[j.req.key]
            value, failure = _reason(res, j.parse)
            out[j.name] = Outcome(value, failure, first.results + [res])
    return out
