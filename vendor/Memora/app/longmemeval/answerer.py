# The one fixed answering agent for every LongMemEval system (Memora S/P/noabs, full context, RAG).
#
# Prompt: the repo-default `longmemeval` template (prompts.ANSWER_PROMPT_LONGMEMEVAL) verbatim, rendered exactly as
# providers/memora/search.py renders it (jinja2 Template(...).render(memories=..., question=..., question_date=...)).
# Only the request plumbing is ours: the rendered prompt is split into consecutive text blocks
# (static prefix | memories | question tail) whose concatenation is byte-identical to the rendered prompt, so the
# reusable parts can carry cache_control.

import re

from jinja2 import Template

from memobench.context import get_meta
from memobench.runtime import MODEL
from memobench.llm import Req
from prompts import ANSWER_PROMPT_EVERMEMOS_LONGMEMEVAL, ANSWER_PROMPT_LONGMEMEVAL, ANSWER_PROMPT_NEMORI_LONGMEMEVAL

EFFORT = "low"
MAX_TOKENS = 8192  # the default template asks for a 6-step chain of thought; thinking counts toward it

_MEMORIES_SLOT = re.compile(r"\{\{\s*memories\s*\}\}")


def answer_template(cfg) -> str:
    """eval.answer_template, selected as providers/memora/search.py always did ("longmemeval" is the default)."""
    answer_template = cfg.eval.get("answer_template", "longmemeval")
    if answer_template == "nemori":
        return ANSWER_PROMPT_NEMORI_LONGMEMEVAL
    elif answer_template == "evermemos":
        return ANSWER_PROMPT_EVERMEMOS_LONGMEMEVAL
    else:  # default to longmemeval
        return ANSWER_PROMPT_LONGMEMEVAL


def _static_prefix(template: str) -> str:
    prefix = template[: _MEMORIES_SLOT.search(template).start()]
    assert "{" not in prefix and "}" not in prefix, "static prefix must be plain text"
    return prefix


def build_answer_req(memories_text: str, question: str, question_date: str, *, template: str = ANSWER_PROMPT_LONGMEMEVAL,
                     cache_memories: bool = False, batch: bool = False, meta: dict | None = None) -> Req:
    """Request for one question. `memories_text` fills the {{ memories }} slot as-is (Memora: "\\n".join of its
    formatted memories; full context: the whole haystack history; RAG: the retrieved chunks).

    cache_control: always on the static prefix; on the memories block too when `cache_memories` (1h TTL).
    TTL is 1h for batch runs and whenever the memories block is cached (a 1h breakpoint may not follow a 5m one).
    """
    prompt = Template(template).render(memories=memories_text, question=question, question_date=question_date)
    prefix = _static_prefix(template)
    head = prefix + memories_text
    assert prompt.startswith(head), "rendered prompt does not start with prefix + memories"

    cache = {"type": "ephemeral"}
    if batch or cache_memories:
        cache["ttl"] = "1h"
    blocks = [
        {"type": "text", "text": prefix, "cache_control": dict(cache)},
        {"type": "text", "text": memories_text},
        {"type": "text", "text": prompt[len(head):]},
    ]
    if cache_memories:
        blocks[1]["cache_control"] = dict(cache)
    blocks = [b for b in blocks if b["text"]]  # the API rejects empty text blocks; concatenation is unchanged
    assert "".join(b["text"] for b in blocks) == prompt

    params = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "output_config": {"effort": EFFORT},
        "messages": [{"role": "user", "content": blocks}],
    }
    return Req(params=params, meta={**get_meta(), **(meta or {}), "stage": "answer"})


def parse_answer(text: str) -> str:
    """Answer extraction from providers/memora/search.py, shared by every system."""
    raw_result = text or ""
    # Handle FINAL ANSWER extraction (used by evermemos template)
    if "FINAL ANSWER:" in raw_result:
        parts = raw_result.split("FINAL ANSWER:")
        return parts[1].strip() if len(parts) > 1 else raw_result.strip()
    return raw_result.strip()
