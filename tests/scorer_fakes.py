"""A batch-capable fake Anthropic client that answers each request with a function of its params.

tests/fakes.FakeAnthropic already implements messages.create and the whole Message Batches surface with real SDK
objects; it names a request by its last message's text. Scorer requests share long prefixes and differ in structured
content, so this subclass replaces only how a reply is chosen: `respond(params) -> {"text" | "content", "stop_reason",
"usage"}`. Everything memobench.llm sees (MessageBatch, results JSONL, Message) stays the real SDK shape.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Callable

from anthropic.types import Message

from memobench.llm import LLM
from tests.fakes import FakeAnthropic, message_payload, text_block, thinking_block, usage_payload

MODELS = ("claude-opus-5-5", "claude-sonnet-5-5")


class RespondingFake(FakeAnthropic):
    def __init__(self, respond: Callable[[dict], dict]) -> None:
        super().__init__()
        self.respond = respond
        self.polls_until_ended = 0
        self.seen: list[dict] = []  # params of every request answered, in order

    def _message(self, params: dict, *, service_tier: str) -> tuple[dict, Message]:
        self.seen.append(copy.deepcopy(params))
        r = self.respond(params)
        content = r.get("content") if r.get("content") is not None else [thinking_block(""), text_block(r.get("text", ""))]
        usage = usage_payload(**(r.get("usage") or {}), service_tier=service_tier)  # the complete Usage shape
        payload = message_payload(msg_id=f"msg_fake_{next(self._ids):06d}", model=params["model"],
                                  content=copy.deepcopy(content), usage=usage,
                                  stop_reason=r.get("stop_reason", "end_turn"))  # fmt: skip
        return payload, Message.model_validate(payload)


def make_llm(root, respond: Callable[[dict], dict]) -> tuple[LLM, RespondingFake]:
    fake = RespondingFake(respond)
    llm = LLM(root=str(root), run_id="scorer-test", client=fake, sleep=lambda s: None, allowed_models=MODELS)
    return llm, fake


def user_texts(params: dict) -> list[str]:
    """The text blocks of the first user message, in order."""
    content = params["messages"][0]["content"]
    return [content] if isinstance(content, str) else [b.get("text", "") for b in content if isinstance(b, dict)]


def json_reply(obj: Any, **kw) -> dict:
    return {"text": json.dumps(obj), **kw}
