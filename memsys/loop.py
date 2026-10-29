"""A resumable tool loop over memobench.llm: `step(chat) -> Req | None`, `feed(chat, result)`.

The same state machine runs the write pass, the dream and the answer. Driven by `run()` it is a real-time chain
(LLM.call per hop); a later harness can instead collect `step()` from many chats into one Batch round per hop and
`feed()` each Result back. The history is append-only (assistant turns go back verbatim, thinking blocks included)
so the cached prefix and thinking-block bindings survive every hop.

Prompt caching: an explicit breakpoint on the static system prompt (tools + system, shared by every principal),
plus top-level automatic caching for the growing conversation tail.

A refusal (stop_reason "refusal") ends the chat with `refused=True`; it never raises, so a replay chain logs it and
moves on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from memobench.llm import LLM, USAGE_FIELDS, Req, Result
from memsys.config import System

log = logging.getLogger("memsys")

Tool = Callable[[str, dict], tuple[str, bool, bool]]  # (name, input) -> (text, is_error, edited)


def base_params(system: System, step: str, tools: list[dict], salt: str | None) -> dict:
    """Everything but `messages`. `salt` sets metadata.user_id: a new cache key, the same prompt."""
    params = {
        "model": system.setting("models", step),
        "max_tokens": system.setting("max_tokens", step),
        "system": [{"type": "text", "text": system.prompts[step], "cache_control": {"type": "ephemeral"}}],
        "tools": tools,
        "output_config": {"effort": system.setting("effort", step)},
        "cache_control": {"type": "ephemeral"},
    }
    if salt is not None:
        params["metadata"] = {"user_id": f"replay-{salt}"}
    return params


@dataclass
class Chat:
    params: dict  # every request param except messages
    messages: list[dict]
    tool: Tool
    max_hops: int
    meta: dict = field(default_factory=dict)
    final_without_tools: bool = False  # after max_hops tool rounds, one more call with tool_choice none (answers)
    hops: int = 0  # model calls fed so far
    done: bool = False
    closing: bool = False  # the next request is that final no-tools call
    text: str = ""
    stop_reason: str | None = None
    refused: bool = False
    error: str | None = None
    edits: int = 0
    calls: list[dict] = field(default_factory=list)  # every tool call: name, input, ok, result (truncated)
    cost_usd: float = 0.0
    usage: dict = field(default_factory=lambda: dict.fromkeys(USAGE_FIELDS, 0))


def step(chat: Chat) -> Req | None:
    """The next request, or None when the chat is done."""
    if chat.done:
        return None
    params = {**chat.params, "messages": chat.messages}
    if chat.closing:
        params["tool_choice"] = {"type": "none"}
    return Req(params=params, meta={**chat.meta, "hop": chat.hops})


def feed(chat: Chat, result: Result) -> None:
    """Apply one response: run its tool calls (appending their results), or finish."""
    chat.hops += 1
    chat.cost_usd += result.cost_usd
    for k in USAGE_FIELDS:
        chat.usage[k] += (result.usage or {}).get(k) or 0
    if result.error:
        chat.error, chat.done = result.error, True
        return
    chat.stop_reason = result.stop_reason
    if result.stop_reason == "refusal":
        log.warning("refusal in %s (hop %d); skipping", chat.meta, chat.hops)
        chat.refused, chat.done = True, True
        return
    content = result.message.get("content") or []
    uses = [b for b in content if b.get("type") == "tool_use"]
    if result.stop_reason != "tool_use" or not uses or chat.closing:
        chat.text, chat.done = result.text.strip(), True
        return
    results = []
    for b in uses:
        text, is_error, edited = chat.tool(b["name"], b.get("input") or {})
        chat.edits += int(edited)
        chat.calls.append({"name": b["name"], "input": b.get("input"), "ok": not is_error, "result": text[:300]})
        results.append({"type": "tool_result", "tool_use_id": b["id"], "content": text, "is_error": is_error})
    chat.messages = [*chat.messages, {"role": "assistant", "content": content}, {"role": "user", "content": results}]
    if chat.hops >= chat.max_hops:
        if chat.final_without_tools:
            chat.closing = True
        else:
            chat.done = True


def run(llm: LLM, chat: Chat) -> Chat:
    """Drive the chat to the end in real time, one LLM.call per hop."""
    while (req := step(chat)) is not None:
        feed(chat, llm.call(req))
    return chat
