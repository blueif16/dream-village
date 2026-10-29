"""scorer.batch.upper_usd: the stage estimate prices a prefix shared by several requests as one cache write plus reads."""

from __future__ import annotations

from memobench.llm import Req
from memobench.pricing import cost_usd
from scorer.batch import upper_usd
from tests.scorer_fakes import make_llm

PREFIX = "p" * 30_000  # ~10k tokens, as a judge's rubric + probe block or an answer's recent window


def req(prefix: str, tail: str) -> Req:
    return Req(params={"model": "claude-sonnet-5-5", "max_tokens": 100, "system": [{"type": "text", "text": "s"}],
                       "messages": [{"role": "user", "content": [
                           {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
                           {"type": "text", "text": tail}]}]})  # fmt: skip


def test_requests_sharing_a_cached_prefix_pay_one_cache_write_and_then_cache_reads(tmp_path):
    llm, _ = make_llm(tmp_path / "runs", lambda p: {"text": "x"})
    shared = upper_usd(llm, [req(PREFIX, f"answer {i}") for i in range(5)], 100)
    distinct = upper_usd(llm, [req(PREFIX + str(i), f"answer {i}") for i in range(5)], 100)
    prefix_tokens = 10_000
    write = cost_usd({"cache_creation_input_tokens": prefix_tokens}, batch=True, model="claude-sonnet-5-5")
    read = cost_usd({"cache_read_input_tokens": prefix_tokens}, batch=True, model="claude-sonnet-5-5")
    assert distinct > 4 * shared / 2  # the shared prefix is not paid in full five times
    assert abs((distinct - shared) - (5 * write - write - 4 * read)) < 0.15 * distinct  # one write, four reads
