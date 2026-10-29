"""The dream: slow consolidation on the simulated clock (system-v0.md §1-2; v0: nightly, small edits).

v0 cadence: at `consolidation_time` (03:00) local sim time, the first such time strictly after the principal's last
session, and only if at least `consolidation_min_sessions` sessions happened since the last dream. Context: the
static dream prompt (cached) + today's sim date + profile, preferences, listing and the full text of every memory
file (memory is small). Tools: small edits; memory_write only restructures an existing file.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta

from memobench.llm import LLM
from memsys import loop
from memsys.archive import when
from memsys.config import System
from memsys.render import injected, memory_files
from memsys.store import Store
from memsys.tools import Toolbox, dream_tools, specs
from village.schema import Session

CLOSING = "Consolidate this memory with small edits, following your instructions. End with the one-line DONE or NO CHANGES summary."  # noqa: E501


def next_dream_at(system: System, after: str | datetime) -> datetime:
    """The first consolidation time strictly after `after`."""
    after = when(after)
    at = datetime.combine(after.date(), time.fromisoformat(system.config["consolidation_time"]))
    return at if at > after else at + timedelta(days=1)


def dream_due(
    system: System, last_activity: str | datetime | None, sessions_since: int, now: str | datetime
) -> datetime | None:
    """When the dream fires, if the clock has crossed it by `now`; else None."""
    if last_activity is None or sessions_since < system.config["consolidation_min_sessions"]:
        return None
    at = next_dream_at(system, last_activity)
    return at if at <= when(now) else None


def dream_chat(
    system: System, store: Store, at: str | datetime, *, archive: list[Session] = (), salt: str | None = None
) -> loop.Chat:
    """`archive` (the principal's sessions) is searchable only when the config gives the dream archive_search."""
    at = when(at)
    names = dream_tools(system)
    text = (
        f"Today is {at:%Y-%m-%d} ({at:%A}), {at:%H:%M}.\n\n{injected(store, versions=True)}\n\n"
        f"{memory_files(store)}\n\n{CLOSING}"
    )
    return loop.Chat(
        params=loop.base_params(system, "dream", specs(names), salt),
        messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
        tool=Toolbox(store, names, archive=archive, now=at.isoformat(), search=system.config["archive_search"],
                     allow_create=("topics",) if system.get("dream_inference") == "patterns" else False),  # fmt: skip
        max_hops=system.setting("max_tool_hops", "dream"),
        meta={"system": system.config["system"], "stage": "dream", "conv_id": store.root.name,
              "question_id": f"dream-{at:%Y-%m-%dT%H:%M}"},  # fmt: skip
    )


def dream_pass(
    llm: LLM, system: System, store: Store, at: str | datetime, *, archive: list[Session] = (), salt: str | None = None
) -> loop.Chat:
    return loop.run(llm, dream_chat(system, store, at, archive=archive, salt=salt))
