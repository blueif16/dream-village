"""A scripted stand-in for the Anthropic client, for memsys tool loops.

tests/fakes.FakeAnthropic names a request by the text of its last user message, which a tool loop can't use (after
the first hop the last message holds only tool_result blocks). This fake instead answers each messages.create call
from a script: a list of replies consumed in order, or a function of the request params. Every reply is a real SDK
`Message` built from the complete payloads in tests/fakes, so memobench.llm sees exactly what the API returns.
"""

from __future__ import annotations

import copy
import itertools
from collections import deque
from typing import Any, Callable

from anthropic.types import Message

from tests.fakes import message_payload, text_block, thinking_block, usage_payload

_ids = itertools.count(1)


def tool_use(name: str, **inputs: Any) -> dict:
    return {"type": "tool_use", "id": f"toolu_fake_{next(_ids):06d}", "name": name, "input": inputs,
            "caller": {"type": "direct"}}  # fmt: skip


def tools_reply(*blocks: dict, usage: dict | None = None) -> dict:
    """An assistant turn that thinks, then calls tools (stop_reason tool_use)."""
    return {"content": [thinking_block(""), *blocks], "stop_reason": "tool_use", "usage": usage}


def text_reply(text: str, *, stop_reason: str = "end_turn", usage: dict | None = None) -> dict:
    return {"content": [thinking_block(""), text_block(text)], "stop_reason": stop_reason, "usage": usage}


def refusal_reply() -> dict:
    return {"content": [], "stop_reason": "refusal", "usage": None}


class ScriptedClient:
    def __init__(self, script: list[dict] | Callable[[dict], dict]) -> None:
        self._queue = deque(script) if isinstance(script, list) else None
        self._fn = script if callable(script) else None
        self.calls: list[dict] = []  # params of every messages.create, in order
        self.messages = self  # client.messages.create(...)

    def create(self, **params: Any) -> Message:
        self.calls.append(copy.deepcopy(params))
        if self._fn is not None:
            reply = self._fn(params)
        elif self._queue:
            reply = self._queue.popleft()
        else:
            raise AssertionError(f"unscripted messages.create call #{len(self.calls)}")
        payload = message_payload(
            msg_id=f"msg_fake_{next(_ids):06d}",
            model=params["model"],
            content=copy.deepcopy(reply["content"]),
            usage=reply.get("usage") or usage_payload(),
            stop_reason=reply["stop_reason"],
        )
        return Message.model_validate(payload)


def last_tool_results(params: dict) -> list[dict]:
    """The tool_result blocks of the request's last (user) message."""
    content = params["messages"][-1]["content"]
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]


def context_text(params: dict) -> str:
    """All text of the first user message (the injected context)."""
    content = params["messages"][0]["content"]
    return content if isinstance(content, str) else "".join(b.get("text", "") for b in content)


def scripted_llm(root, script: list[dict] | Callable[[dict], dict]):
    """(LLM on `root` with a ScriptedClient, the client). No real sleeps."""
    from memobench.llm import LLM

    client = ScriptedClient(script)
    llm = LLM(root=str(root), run_id="memsys-test", client=client, sleep=lambda s: None,
              allowed_models=("claude-haiku-5-5", "claude-sonnet-5-5"))  # fmt: skip
    return llm, client
