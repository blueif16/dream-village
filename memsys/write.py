"""The fast write: after each finished session, a Haiku tool loop files what is durable (system-v0.md §1).

Context: the static write prompt (cached) + <profile>, <preferences>, <memory_listing> (with version tokens) + the
session transcript with its date. archive_search covers only sessions that started before this one, so the model can
check whether a passing mention recurs (the calibration rule) and cite evidence ids for [inferred] lines (D24).
The loop ends when the model stops calling tools or after `max_tool_hops` calls.
"""

from __future__ import annotations

from memobench.llm import LLM
from memsys import loop
from memsys.archive import transcript
from memsys.config import IMPLEMENTED, System
from memsys.render import injected
from memsys.store import Store
from memsys.tools import Toolbox, specs, write_tools
from village.schema import Session

CLOSING = "File what is durable from this session into memory, following your instructions. End with the one-line DONE or NO CHANGES summary."  # noqa: E501


def write_chat(system: System, store: Store, session: Session, *, archive: list[Session], salt: str | None = None) -> loop.Chat:
    text = f"{injected(store, versions=True)}\n\n{transcript(session)}\n\n{CLOSING}"
    names = write_tools(system)
    tool = Toolbox(store, names, archive=archive, now=session.start, search=system.config["archive_search"])
    return loop.Chat(
        params=loop.base_params(system, "write", specs(names), salt),
        messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
        tool=tool,
        max_hops=system.setting("max_tool_hops", "write"),
        meta={"system": system.config["system"], "stage": "write", "conv_id": session.principal, "question_id": session.id},
    )


def write_pass(
    llm: LLM, system: System, store: Store, session: Session, *, archive: list[Session], salt: str | None = None
) -> loop.Chat:
    if system.config["write_after"] != "session":
        raise NotImplementedError(f"write_after={system.config['write_after']!r}: {IMPLEMENTED['write_after'][1]}")
    return loop.run(llm, write_chat(system, store, session, archive=archive, salt=salt))
