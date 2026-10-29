# The one fixed answering agent for every LoCoMo system (Memora S/P/noabs, full context, RAG).
#
# Prompt: ANSWER_PROMPT_COMBINED verbatim, rendered exactly as providers/memora/search.py renders it
# (jinja2 Template(...).render(memories=..., question=...)). Only the request plumbing is ours: the rendered
# prompt is split into consecutive text blocks (static prefix | memories | question tail) whose concatenation
# is byte-identical to the rendered prompt, so the reusable parts can carry cache_control.

from jinja2 import Template

from memobench.context import get_meta
from memobench.runtime import MODEL
from memobench.llm import Req
from prompts import ANSWER_PROMPT_COMBINED

EFFORT = "low"
MAX_TOKENS = 4096  # thinking counts toward it

_MEMORIES_SLOT = "{{memories}}"
_PREFIX = ANSWER_PROMPT_COMBINED.split(_MEMORIES_SLOT, 1)[0]
assert "{" not in _PREFIX and "}" not in _PREFIX, "static prefix must be plain text"


def build_answer_req(memories_text: str, question: str, *, cache_memories: bool = False, batch: bool = False,
                     meta: dict | None = None) -> Req:
    """Request for one question. `memories_text` fills the {{memories}} slot as-is (Memora: json.dumps(formatted,
    indent=4); full context: the rendered conversation; RAG: the retrieved chunks).

    cache_control: always on the static prefix; on the memories block too when `cache_memories` (1h TTL).
    TTL is 1h for batch runs and whenever the memories block is cached (a 1h breakpoint may not follow a 5m one).
    """
    prompt = Template(ANSWER_PROMPT_COMBINED).render(memories=memories_text, question=question)
    head = _PREFIX + memories_text
    assert prompt.startswith(head), "rendered prompt does not start with prefix + memories"

    cache = {"type": "ephemeral"}
    if batch or cache_memories:
        cache["ttl"] = "1h"
    blocks = [
        {"type": "text", "text": _PREFIX, "cache_control": dict(cache)},
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
    """Answer post-processing from providers/memora/search.py, shared by every system."""
    result = (text or "").strip()
    if not result:
        return "ERROR: Empty response from LLM."
    if "FINAL ANSWER:" in result:
        parts = result.split("FINAL ANSWER:")
        if len(parts) > 1:
            return parts[1].strip()
    return result
