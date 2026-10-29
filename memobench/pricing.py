"""Pricing for Claude responses (spec: docs/llm-layer-spec.md, "Pricing"; addendum §4, verified Oct 20 2026)."""

from __future__ import annotations

LONG_CONTEXT_TOKENS = 100_000  # prompt_len above this -> the whole request is priced at the long tier
BATCH_DISCOUNT = 0.5

# Gemini, $/MTok, flat (no long-context tier); always real-time (no batch discount). gemini-4-argon is the
# introductory price through 2026-11-30 (ai.google.dev "What's new in Gemini 4 Argon", checked 2026-10-25).
GEMINI_RATES = {"gemini-4-argon": {"input": 2.00, "output": 10.00, "cache_read": 0.10}}


def _flat(inp: float, out: float, read: float) -> dict[str, tuple[float, float]]:
    """A model without a long-context tier: the same rate in both tiers. Cache writes: 1.25x input (5m), 2x (1h)."""
    rates = {"input": inp, "output": out, "cache_write_5m": 1.25 * inp, "cache_write_1h": 2 * inp, "cache_read": read}
    return {k: (v, v) for k, v in rates.items()}


# Claude, $/MTok: (prompt <= 100k, prompt > 100k). Opus/Sonnet 5.5: claude-api model table, cached 2026-10-06 (D25).
CLAUDE_RATES = {
    "claude-haiku-5-5": {
        "input": (0.10, 0.50),
        "output": (0.50, 2.50),
        "cache_write_5m": (0.125, 0.625),
        "cache_write_1h": (0.20, 1.00),
        "cache_read": (0.01, 0.05),
    },
    "claude-opus-5-5": _flat(4.00, 20.00, 0.20),
    "claude-sonnet-5-5": _flat(2.00, 10.00, 0.20),
}


def cost_usd(usage: dict, *, batch: bool, model: str = "claude-haiku-5-5") -> float:
    """USD cost of one response's `usage` (a dict, e.g. `Usage.model_dump()`); missing/None fields count as 0."""

    def n(d: dict | None, k: str) -> int:
        return (d or {}).get(k) or 0

    if model in GEMINI_RATES:
        r = GEMINI_RATES[model]
        return (n(usage, "input_tokens") * r["input"] + n(usage, "cache_read_input_tokens") * r["cache_read"]
                + n(usage, "output_tokens") * r["output"]) / 1e6  # fmt: skip

    if model not in CLAUDE_RATES:
        raise ValueError(f"no pricing for model {model!r} in memobench.pricing")
    rates = CLAUDE_RATES[model]
    inp, out = n(usage, "input_tokens"), n(usage, "output_tokens")
    write, read = n(usage, "cache_creation_input_tokens"), n(usage, "cache_read_input_tokens")
    tier = 1 if inp + write + read > LONG_CONTEXT_TOKENS else 0

    split = usage.get("cache_creation")
    if split and ("ephemeral_5m_input_tokens" in split or "ephemeral_1h_input_tokens" in split):
        write_5m, write_1h = n(split, "ephemeral_5m_input_tokens"), n(split, "ephemeral_1h_input_tokens")
    else:  # no TTL breakdown -> all writes are 5m
        write_5m, write_1h = write, 0

    tokens = {"input": inp, "output": out, "cache_write_5m": write_5m, "cache_write_1h": write_1h, "cache_read": read}
    usd = sum(tokens[k] * rates[k][tier] for k in rates) / 1e6
    return usd * BATCH_DISCOUNT if batch else usd
