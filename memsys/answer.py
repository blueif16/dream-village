"""Answering a request at time t (system-v0.md §1, §3 "Every turn"), read-only on memory.

Context: the static answer prompt (cached) + <profile>, <preferences>, <memory_listing> + the last
`recent_window_sessions` raw sessions before t + today's date + the request. Tools: memory_read and archive_search
(sessions with start < t). No edit tool is offered, so answering never changes memory.

It is a resumable state machine so a harness can run many answers through Batch rounds, one round per tool hop:

    state = answer_state(...)
    while not isinstance(x := step(state), Final):
        feed(state, llm.call(x))      # or: collect step() of many states, llm.run(..., mode="batch"), feed each

After `max_tool_hops` tool rounds, one last call with tool_choice none forces a reply.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from memobench.llm import LLM, Req
from memsys import loop
from memsys.archive import before, load_forgotten, redact, transcript, when
from memsys.config import System
from memsys.render import injected
from memsys.tools import ANSWER_TOOLS, Toolbox, specs
from village.schema import Session

feed = loop.feed


@dataclass(frozen=True)
class Final:
    text: str
    refused: bool
    stop_reason: str | None
    hops: int
    cost_usd: float
    usage: dict
    calls: list
    error: str | None


def answer_state(
    system: System,
    memory_dir: Path | str,
    sessions: list[Session],
    request: str,
    t: str | datetime,
    *,
    salt: str | None = None,
) -> loop.Chat:
    store = system.store(memory_dir)
    past = redact(before(sessions, t), load_forgotten(memory_dir))  # forgotten spans masked (C4; none in v0)
    recent = past[-system.config["recent_window_sessions"] :] if system.config["recent_window_sessions"] else []
    now = when(t)
    context = (
        f"{injected(store, versions=False)}\n\n<recent_sessions>\n"
        + ("\n".join(transcript(s) for s in recent) or "(none)")
        + f"\n</recent_sessions>\n\nToday is {now:%Y-%m-%d} ({now:%A}), {now:%H:%M}."
    )
    tool = Toolbox(store, ANSWER_TOOLS, archive=past, now=now.isoformat(), search=system.config["archive_search"])
    return loop.Chat(
        params=loop.base_params(system, "answer", specs(ANSWER_TOOLS), salt),
        messages=[{"role": "user", "content": [{"type": "text", "text": context},
                                               {"type": "text", "text": f"<request>\n{request}\n</request>"}]}],  # fmt: skip
        tool=tool,
        max_hops=system.setting("max_tool_hops", "answer"),
        final_without_tools=True,
        meta={"system": system.config["system"], "stage": "answer", "conv_id": Path(memory_dir).name,
              "question_id": f"answer-{now:%Y-%m-%dT%H:%M}"},  # fmt: skip
    )


def step(state: loop.Chat) -> Req | Final:
    req = loop.step(state)
    if req is not None:
        return req
    return Final(state.text, state.refused, state.stop_reason, state.hops, state.cost_usd, dict(state.usage),
                 list(state.calls), state.error)  # fmt: skip


def answer_sync(
    llm: LLM,
    system: System,
    memory_dir: Path | str,
    sessions: list[Session],
    request: str,
    t: str | datetime,
    *,
    salt: str | None = None,
) -> Final:
    state = answer_state(system, memory_dir, sessions, request, t, salt=salt)
    while not isinstance(x := step(state), Final):
        feed(state, llm.call(x))
    return x
