"""Hand-written fake of the Anthropic client surface that memobench.llm uses.

* `messages.create(**params)` and `messages.batches.{create, retrieve, results, cancel}`.
* Every return value is a REAL SDK object built with `model_validate` on a complete payload
  (Message, MessageBatch, MessageBatchIndividualResponse inside a real JSONLDecoder).
* HTTP failures are the SDK's real exception classes, built from an `httpx2.Response`
  exactly as `anthropic._client.Anthropic._make_status_error` does (the SDK uses httpx2).
* Every call is recorded in order (`calls`), plus start/end `events` for ordering checks.
* Scriptable: per-call / per-prompt failures, per-prompt replies, per-prompt batch outcomes
  (succeeded / errored_invalid / errored_api / errored_overloaded / expired / canceled),
  out-of-order results, a batch that never ends, and a simulated crash during polling.

Tests identify requests by their *prompt* (the text of the last user message), so scripting
never depends on how memobench computes keys.
"""

from __future__ import annotations

import copy
import itertools
import json
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import anthropic
import httpx2
from anthropic._decoders.jsonl import JSONLDecoder
from anthropic.types import Message
from anthropic.types.messages import MessageBatch, MessageBatchIndividualResponse

API = "https://api.anthropic.com"
CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")  # the real API's custom_id constraint


class SimulatedCrash(BaseException):
    """Stands in for the process dying (BaseException so no `except Exception` swallows it)."""


class FakeGuardTripped(BaseException):
    """Raised when the code under test polls forever; fails the test instead of hanging it."""


# --------------------------------------------------------------------------- errors


_STATUS_CLASSES: dict[int, type[anthropic.APIStatusError]] = {
    400: anthropic.BadRequestError,
    401: anthropic.AuthenticationError,
    403: anthropic.PermissionDeniedError,
    404: anthropic.NotFoundError,
    409: anthropic.ConflictError,
    413: anthropic.RequestTooLargeError,
    422: anthropic.UnprocessableEntityError,
    429: anthropic.RateLimitError,
    529: anthropic.OverloadedError,
}

_ERROR_TYPES: dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    503: "api_error",
    504: "timeout_error",
    529: "overloaded_error",
}


def api_status_error(
    status: int,
    *,
    headers: dict[str, str] | None = None,
    message: str | None = None,
    path: str = "/v1/messages",
) -> anthropic.APIStatusError:
    """Build the exact exception the SDK raises for an HTTP `status` response."""
    request = httpx2.Request("POST", API + path)
    body = {
        "type": "error",
        "error": {"type": _ERROR_TYPES.get(status, "api_error"), "message": message or f"fake {status}"},
        "request_id": f"req_fake_{status}",
    }
    hdrs = {"request-id": f"req_fake_{status}", **(headers or {})}
    response = httpx2.Response(status, headers=hdrs, json=body, request=request)
    err_msg = f"Error code: {status} - {body}"
    if status in _STATUS_CLASSES:
        return _STATUS_CLASSES[status](err_msg, response=response, body=body)
    if status >= 500:
        return anthropic.InternalServerError(err_msg, response=response, body=body)
    return anthropic.APIStatusError(err_msg, response=response, body=body)


def connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=httpx2.Request("POST", API + "/v1/messages"))


def timeout_error() -> anthropic.APITimeoutError:
    return anthropic.APITimeoutError(request=httpx2.Request("POST", API + "/v1/messages"))


# --------------------------------------------------------------------------- payloads


def usage_payload(
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    *,
    ephemeral_5m_input_tokens: int | None = None,
    ephemeral_1h_input_tokens: int | None = None,
    service_tier: str = "standard",
) -> dict:
    """Complete `Usage` payload. Breakdown defaults to all-5m, matching the API."""
    if ephemeral_5m_input_tokens is None and ephemeral_1h_input_tokens is None:
        ephemeral_5m_input_tokens, ephemeral_1h_input_tokens = cache_creation_input_tokens, 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation": {
            "ephemeral_5m_input_tokens": ephemeral_5m_input_tokens or 0,
            "ephemeral_1h_input_tokens": ephemeral_1h_input_tokens or 0,
        },
        "inference_geo": "us",
        "output_tokens_details": {"thinking_tokens": 0},
        "server_tool_use": None,
        "service_tier": service_tier,
        "speed": "standard",
    }


def text_block(text: str) -> dict:
    return {"type": "text", "text": text, "citations": None}


def thinking_block(thinking: str) -> dict:
    return {"type": "thinking", "thinking": thinking, "signature": "sig_fake_" + "0" * 16}


def message_payload(
    *,
    msg_id: str,
    model: str,
    content: list[dict],
    usage: dict,
    stop_reason: str = "end_turn",
) -> dict:
    """Complete `Message` payload (every field the SDK model declares)."""
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "stop_details": (
            {"type": "refusal", "category": None, "explanation": None} if stop_reason == "refusal" else None
        ),
        "container": None,
        "diagnostics": None,
        "usage": usage,
    }


def _error_response(err_type: str, message: str) -> dict:
    return {"type": "error", "error": {"type": err_type, "message": message}, "request_id": "req_fake_batch"}


def prompt_of(params: dict) -> str:
    """The text of the last message: how tests name a request."""
    content = params["messages"][-1]["content"]
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


# --------------------------------------------------------------------------- fake client


@dataclass
class Call:
    method: str
    kwargs: dict


@dataclass
class _Reply:
    content: list[dict] | None = None
    usage: dict | None = None
    stop_reason: str = "end_turn"


@dataclass
class _Batch:
    id: str
    created_at: datetime
    requests: list[dict]
    outcomes: dict[str, str]  # custom_id -> outcome
    retrieves: int = 0
    canceled_at: datetime | None = None
    cancel_polls_left: int = 1
    ended_at: datetime | None = None
    observed_ended: bool = False
    message_ids: dict[str, str] = field(default_factory=dict)


BATCH_OUTCOMES = {"succeeded", "errored_invalid", "errored_api", "errored_overloaded", "expired", "canceled"}


class FakeAnthropic:
    """Stand-in for `anthropic.Anthropic(max_retries=0)`; see module docstring."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._ids = itertools.count(1)
        self.calls: list[Call] = []
        self.events: list[str] = []
        self.responses: list[Message] = []  # every Message returned by messages.create, in order
        # messages.create scripting
        self._replies: dict[str, _Reply] = {}
        self._fail_next: deque[BaseException] = deque()
        self._fail_prompt: dict[str, deque[BaseException]] = defaultdict(deque)
        self.hold_s = 0.0  # real seconds each messages.create takes (for concurrency checks)
        self.in_flight = 0
        self.max_in_flight = 0
        # batch scripting
        self._batches: dict[str, _Batch] = {}
        self._outcomes: dict[str, deque[str]] = defaultdict(deque)
        self.polls_until_ended = 1  # retrieve() answers "in_progress" this many times, then "ended"
        self.never_end = False  # batch stays "in_progress" until canceled
        self.succeed_before_cancel: set[str] = set()  # prompts that finished before a cancel landed
        self.crash_on_next_retrieve = False
        self.results_reversed = True  # results come back in a different order than submitted
        self.max_retrieves = 2000
        self.messages = _Messages(self)

    # ---- scripting API
    def reply(
        self,
        prompt: str,
        *,
        text: str | None = None,
        content: list[dict] | None = None,
        usage: dict | None = None,
        stop_reason: str = "end_turn",
    ) -> None:
        if text is not None:
            content = [text_block(text)]
        self._replies[prompt] = _Reply(content=content, usage=usage, stop_reason=stop_reason)

    def fail_next(self, *errors: BaseException) -> None:
        """The next len(errors) messages.create calls raise these, in order (any prompt)."""
        self._fail_next.extend(errors)

    def fail_prompt(self, prompt: str, *errors: BaseException) -> None:
        """The next messages.create calls for `prompt` raise these, in order."""
        self._fail_prompt[prompt].extend(errors)

    def batch_outcomes(self, prompt: str, *outcomes: str) -> None:
        """Outcome of `prompt` in its 1st, 2nd, ... batch submission; afterwards 'succeeded'."""
        assert set(outcomes) <= BATCH_OUTCOMES, outcomes
        self._outcomes[prompt].extend(outcomes)

    # ---- inspection API
    def _of(self, method: str) -> list[dict]:
        return [c.kwargs for c in self.calls if c.method == method]

    @property
    def methods(self) -> list[str]:
        return [c.method for c in self.calls]

    @property
    def create_calls(self) -> list[dict]:
        return self._of("messages.create")

    @property
    def created_prompts(self) -> list[str]:
        return [prompt_of(p) for p in self.create_calls]

    @property
    def batch_creates(self) -> list[list[dict]]:
        """One entry per batches.create call: the list of {custom_id, params} submitted."""
        return [c["requests"] for c in self._of("messages.batches.create")]

    @property
    def batch_ids(self) -> list[str]:
        return list(self._batches)

    @property
    def retrieved_ids(self) -> list[str]:
        return [c["message_batch_id"] for c in self._of("messages.batches.retrieve")]

    @property
    def canceled_ids(self) -> list[str]:
        return [c["message_batch_id"] for c in self._of("messages.batches.cancel")]

    @property
    def results_ids(self) -> list[str]:
        return [c["message_batch_id"] for c in self._of("messages.batches.results")]

    def batch_observed_ended(self, batch_id: str) -> bool:
        return self._batches[batch_id].observed_ended

    def message_id_for(self, batch_id: str, custom_id: str) -> str:
        return self._batches[batch_id].message_ids[custom_id]

    # ---- internals shared by the resources
    def _record(self, method: str, kwargs: dict) -> None:
        self.calls.append(Call(method, copy.deepcopy(kwargs)))

    def _message(self, params: dict, *, service_tier: str) -> tuple[dict, Message]:
        prompt = prompt_of(params)
        r = self._replies.get(prompt, _Reply())
        content = r.content if r.content is not None else [text_block(f"answer:{prompt}")]
        usage = dict(r.usage) if r.usage is not None else usage_payload()
        usage["service_tier"] = service_tier
        payload = message_payload(
            msg_id=f"msg_fake_{next(self._ids):06d}",
            model=params["model"],
            content=copy.deepcopy(content),
            usage=usage,
            stop_reason=r.stop_reason,
        )
        return payload, Message.model_validate(payload)

    def _batch_payload(self, b: _Batch, status: str) -> MessageBatch:
        ended = status == "ended"
        counts = dict(processing=0, succeeded=0, errored=0, canceled=0, expired=0)
        if ended:
            for o in b.outcomes.values():
                counts["errored" if o.startswith("errored") else o] += 1
        else:
            counts["processing"] = len(b.requests)
        payload = {
            "id": b.id,
            "type": "message_batch",
            "processing_status": status,
            "request_counts": counts,
            "created_at": b.created_at.isoformat(),
            "expires_at": (b.created_at + timedelta(hours=24)).isoformat(),
            "ended_at": b.ended_at.isoformat() if ended and b.ended_at else None,
            "cancel_initiated_at": b.canceled_at.isoformat() if b.canceled_at else None,
            "archived_at": None,
            "results_url": f"{API}/v1/messages/batches/{b.id}/results" if ended else None,
        }
        return MessageBatch.model_validate(payload)


class _Messages:
    def __init__(self, fake: FakeAnthropic) -> None:
        self._fake = fake
        self.batches = _Batches(fake)

    def create(self, **params: Any) -> Message:
        f = self._fake
        with f._lock:
            f._record("messages.create", params)
            prompt = prompt_of(params)
            f.events.append(f"messages.create:start:{prompt}")
            err = None
            if f._fail_prompt[prompt]:
                err = f._fail_prompt[prompt].popleft()
            elif f._fail_next:
                err = f._fail_next.popleft()
            f.in_flight += 1
            f.max_in_flight = max(f.max_in_flight, f.in_flight)
        try:
            if f.hold_s:
                time.sleep(f.hold_s)
            if err is not None:
                raise err
            with f._lock:
                _, msg = f._message(params, service_tier="standard")
                f.responses.append(msg)
            return msg
        finally:
            with f._lock:
                f.in_flight -= 1
                f.events.append(f"messages.create:end:{prompt}")


class _Batches:
    def __init__(self, fake: FakeAnthropic) -> None:
        self._fake = fake

    def _get(self, batch_id: str) -> _Batch:
        if batch_id not in self._fake._batches:
            raise api_status_error(404, message=f"batch {batch_id} not found", path=f"/v1/messages/batches/{batch_id}")
        return self._fake._batches[batch_id]

    def create(self, *, requests: Any, **kwargs: Any) -> MessageBatch:
        f = self._fake
        with f._lock:
            reqs = [dict(r) for r in requests]
            f._record("messages.batches.create", {"requests": reqs, **kwargs})
            f.events.append(f"messages.batches.create:{len(reqs)}")
            ids = [r["custom_id"] for r in reqs]
            if not reqs:
                raise api_status_error(400, message="requests: must not be empty", path="/v1/messages/batches")
            if len(set(ids)) != len(ids):
                raise api_status_error(400, message="custom_id must be unique within a batch", path="/v1/messages/batches")
            if not all(isinstance(i, str) and CUSTOM_ID_RE.match(i) for i in ids):
                raise api_status_error(400, message="custom_id: invalid format", path="/v1/messages/batches")
            outcomes = {}
            for r in reqs:
                q = f._outcomes[prompt_of(r["params"])]
                outcomes[r["custom_id"]] = q.popleft() if q else "succeeded"
            b = _Batch(
                id=f"msgbatch_fake_{next(f._ids):06d}",
                created_at=datetime.now(timezone.utc),
                requests=copy.deepcopy(reqs),
                outcomes=outcomes,
            )
            f._batches[b.id] = b
            return f._batch_payload(b, "in_progress")

    def retrieve(self, message_batch_id: str, **kwargs: Any) -> MessageBatch:
        f = self._fake
        with f._lock:
            f._record("messages.batches.retrieve", {"message_batch_id": message_batch_id, **kwargs})
            if f.crash_on_next_retrieve:
                f.crash_on_next_retrieve = False
                raise SimulatedCrash(f"process died while polling {message_batch_id}")
            if len(f.retrieved_ids) > f.max_retrieves:
                raise FakeGuardTripped(f"{message_batch_id} polled > {f.max_retrieves} times")
            b = self._get(message_batch_id)
            b.retrieves += 1
            if b.ended_at is not None:
                status = "ended"
            elif b.canceled_at is not None:
                if b.cancel_polls_left > 0:
                    b.cancel_polls_left -= 1
                    status = "canceling"
                else:
                    status = "ended"
            elif not f.never_end and b.retrieves > f.polls_until_ended:
                status = "ended"
            else:
                status = "in_progress"
            if status == "ended":
                b.ended_at = b.ended_at or datetime.now(timezone.utc)
                b.observed_ended = True
            return f._batch_payload(b, status)

    def cancel(self, message_batch_id: str, **kwargs: Any) -> MessageBatch:
        f = self._fake
        with f._lock:
            f._record("messages.batches.cancel", {"message_batch_id": message_batch_id, **kwargs})
            b = self._get(message_batch_id)
            if b.ended_at is None and b.canceled_at is None:
                b.canceled_at = datetime.now(timezone.utc)
                for r in b.requests:
                    if prompt_of(r["params"]) not in f.succeed_before_cancel:
                        b.outcomes[r["custom_id"]] = "canceled"
            return f._batch_payload(b, "ended" if b.ended_at else "canceling")

    def results(self, message_batch_id: str, **kwargs: Any) -> JSONLDecoder[MessageBatchIndividualResponse]:
        f = self._fake
        with f._lock:
            f._record("messages.batches.results", {"message_batch_id": message_batch_id, **kwargs})
            b = self._get(message_batch_id)
            if b.ended_at is None:
                raise api_status_error(
                    400,
                    message="results are not available until processing_status is 'ended'",
                    path=f"/v1/messages/batches/{message_batch_id}/results",
                )
            lines = []
            order = list(reversed(b.requests)) if f.results_reversed else list(b.requests)
            for r in order:
                cid, outcome = r["custom_id"], b.outcomes[r["custom_id"]]
                if outcome == "succeeded":
                    payload, msg = f._message(r["params"], service_tier="batch")
                    b.message_ids[cid] = msg.id
                    result = {"type": "succeeded", "message": payload}
                elif outcome == "errored_invalid":
                    result = {
                        "type": "errored",
                        "error": _error_response("invalid_request_error", "messages.0.content: field required"),
                    }
                elif outcome == "errored_api":
                    result = {"type": "errored", "error": _error_response("api_error", "Internal server error")}
                elif outcome == "errored_overloaded":
                    result = {"type": "errored", "error": _error_response("overloaded_error", "Overloaded")}
                else:  # expired / canceled
                    result = {"type": outcome}
                line = {"custom_id": cid, "result": result}
                MessageBatchIndividualResponse.model_validate(line)  # payload must be schema-complete
                lines.append((json.dumps(line) + "\n").encode())
            url = f"{API}/v1/messages/batches/{message_batch_id}/results"
            http_response = httpx2.Response(
                200, headers={"content-type": "application/binary"}, request=httpx2.Request("GET", url)
            )
            return JSONLDecoder(
                raw_iterator=iter(lines), line_type=MessageBatchIndividualResponse, http_response=http_response
            )
