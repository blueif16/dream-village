"""Pricing rules for `cost_usd` (spec: "Pricing").

Every expected value is hand-computed from the spec's rate table, $/MTok (<=100k | >100k prompt):
  input 0.10 | 0.50 · output 0.50 | 2.50 · 5m cache write 0.125 | 0.625 · 1h cache write 0.20 | 1.00
  · cache read 0.01 | 0.05.   prompt_len = input + cache_creation + cache_read.   batch -> x 0.5.
"""

from __future__ import annotations

import pytest
from anthropic.types import Usage

from memobench.pricing import cost_usd
from tests.fakes import usage_payload


def approx(x: float):
    return pytest.approx(x, rel=1e-9, abs=1e-15)


def test_pricing_short_tier_input_and_output():
    usage = {"input_tokens": 1000, "output_tokens": 1000}
    assert cost_usd(usage, batch=False) == approx(1000 * 0.10 / 1e6 + 1000 * 0.50 / 1e6)


def test_pricing_long_tier_input_and_output_when_prompt_over_100k():
    usage = {"input_tokens": 150_000, "output_tokens": 1000}
    assert cost_usd(usage, batch=False) == approx(150_000 * 0.50 / 1e6 + 1000 * 2.50 / 1e6)


def test_pricing_prompt_of_exactly_100_000_stays_short_tier():
    usage = {"input_tokens": 100_000, "output_tokens": 1000}
    assert cost_usd(usage, batch=False) == approx(100_000 * 0.10 / 1e6 + 1000 * 0.50 / 1e6)


def test_pricing_prompt_of_100_001_is_long_tier():
    usage = {"input_tokens": 100_001, "output_tokens": 1000}
    assert cost_usd(usage, batch=False) == approx(100_001 * 0.50 / 1e6 + 1000 * 2.50 / 1e6)


def test_pricing_cache_read_short_tier():
    usage = {"input_tokens": 1000, "output_tokens": 100, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 20_000}
    assert cost_usd(usage, batch=False) == approx(1000 * 0.10 / 1e6 + 20_000 * 0.01 / 1e6 + 100 * 0.50 / 1e6)


def test_pricing_prompt_len_includes_cache_read_tokens():
    """input 60k alone is short, but 60k + 50k cache_read = 110k > 100k -> the whole request is long tier."""
    usage = {"input_tokens": 60_000, "output_tokens": 1000, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 50_000}
    expected = 60_000 * 0.50 / 1e6 + 50_000 * 0.05 / 1e6 + 1000 * 2.50 / 1e6
    assert cost_usd(usage, batch=False) == approx(expected)


def test_pricing_prompt_len_includes_cache_creation_tokens():
    """input 60k + 50k cache write = 110k > 100k -> long tier; the write is 5m (breakdown says so)."""
    usage = {
        "input_tokens": 60_000,
        "output_tokens": 1000,
        "cache_creation_input_tokens": 50_000,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 50_000, "ephemeral_1h_input_tokens": 0},
    }
    expected = 60_000 * 0.50 / 1e6 + 50_000 * 0.625 / 1e6 + 1000 * 2.50 / 1e6
    assert cost_usd(usage, batch=False) == approx(expected)


def test_pricing_5m_and_1h_cache_writes_priced_at_their_own_ttl():
    usage = {
        "input_tokens": 1000,
        "output_tokens": 100,
        "cache_creation_input_tokens": 3000,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 2000},
    }
    expected = 1000 * 0.10 / 1e6 + 1000 * 0.125 / 1e6 + 2000 * 0.20 / 1e6 + 100 * 0.50 / 1e6
    assert cost_usd(usage, batch=False) == approx(expected)


def test_pricing_1h_cache_write_long_tier():
    """prompt_len = 100_000 + 4_000 = 104_000 > 100k -> 1h write at 1.00."""
    usage = {
        "input_tokens": 100_000,
        "output_tokens": 100,
        "cache_creation_input_tokens": 4000,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 4000},
    }
    expected = 100_000 * 0.50 / 1e6 + 4000 * 1.00 / 1e6 + 100 * 2.50 / 1e6
    assert cost_usd(usage, batch=False) == approx(expected)


def test_pricing_missing_cache_creation_breakdown_treats_writes_as_5m():
    usage = {"input_tokens": 1000, "output_tokens": 100, "cache_creation_input_tokens": 2000, "cache_read_input_tokens": 0}
    expected = 1000 * 0.10 / 1e6 + 2000 * 0.125 / 1e6 + 100 * 0.50 / 1e6
    assert cost_usd(usage, batch=False) == approx(expected)


def test_pricing_none_cache_creation_breakdown_treats_writes_as_5m():
    usage = {
        "input_tokens": 1000,
        "output_tokens": 100,
        "cache_creation_input_tokens": 2000,
        "cache_read_input_tokens": 0,
        "cache_creation": None,
    }
    expected = 1000 * 0.10 / 1e6 + 2000 * 0.125 / 1e6 + 100 * 0.50 / 1e6
    assert cost_usd(usage, batch=False) == approx(expected)


def test_pricing_batch_halves_short_tier_cost():
    usage = {
        "input_tokens": 1000,
        "output_tokens": 1000,
        "cache_creation_input_tokens": 2000,
        "cache_read_input_tokens": 4000,
    }
    expected = (1000 * 0.10 / 1e6 + 1000 * 0.50 / 1e6 + 2000 * 0.125 / 1e6 + 4000 * 0.01 / 1e6) * 0.5
    assert cost_usd(usage, batch=True) == approx(expected)


def test_pricing_batch_halves_long_tier_cost():
    usage = {"input_tokens": 150_000, "output_tokens": 1000}
    assert cost_usd(usage, batch=True) == approx((150_000 * 0.50 / 1e6 + 1000 * 2.50 / 1e6) * 0.5)


def test_pricing_none_usage_fields_count_as_zero():
    usage = {
        "input_tokens": 1000,
        "output_tokens": 100,
        "cache_creation_input_tokens": None,
        "cache_read_input_tokens": None,
        "cache_creation": None,
    }
    assert cost_usd(usage, batch=False) == approx(1000 * 0.10 / 1e6 + 100 * 0.50 / 1e6)


def test_pricing_empty_usage_costs_zero():
    assert cost_usd({}, batch=False) == 0.0


def test_pricing_accepts_full_sdk_usage_dump():
    """A real `Usage.model_dump()` (extra fields: service_tier, inference_geo, ...) prices the same.
    1000 in + 300 out + 500 cache read + 800 cache write split 200 5m / 600 1h; prompt_len 2300 -> short."""
    usage = Usage.model_validate(
        usage_payload(1000, 300, 800, 500, ephemeral_5m_input_tokens=200, ephemeral_1h_input_tokens=600)
    ).model_dump()
    expected = 1000 * 0.10 / 1e6 + 300 * 0.50 / 1e6 + 500 * 0.01 / 1e6 + 200 * 0.125 / 1e6 + 600 * 0.20 / 1e6
    assert cost_usd(usage, batch=False) == approx(expected)


# --------------------------------------------------------------------------- Opus 5.5 / Sonnet 5.5
# Hand-computed from the claude-api model table (cached 2026-10-06), $/MTok, flat (no long-context tier):
#   claude-opus-5-5:   input 4.00 · output 20.00 · cache read 0.20 · 5m write 5.00 (1.25x) · 1h write 8.00 (2x)
#   claude-sonnet-5-5: input 2.00 · output 10.00 · cache read 0.20 · 5m write 2.50 (1.25x) · 1h write 4.00 (2x)
#   Batch -> x 0.5.


def test_pricing_opus_input_and_output():
    usage = {"input_tokens": 1000, "output_tokens": 1000}
    assert cost_usd(usage, batch=False, model="claude-opus-5-5") == approx(1000 * 4.00 / 1e6 + 1000 * 20.00 / 1e6)


def test_pricing_sonnet_input_and_output():
    usage = {"input_tokens": 1000, "output_tokens": 1000}
    assert cost_usd(usage, batch=False, model="claude-sonnet-5-5") == approx(1000 * 2.00 / 1e6 + 1000 * 10.00 / 1e6)


def test_pricing_opus_cache_read_and_5m_write():
    usage = {"input_tokens": 100, "output_tokens": 0, "cache_creation_input_tokens": 3000, "cache_read_input_tokens": 7000}
    expected = 100 * 4.00 / 1e6 + 3000 * 5.00 / 1e6 + 7000 * 0.20 / 1e6
    assert cost_usd(usage, batch=False, model="claude-opus-5-5") == approx(expected)


def test_pricing_sonnet_cache_read_and_5m_write():
    usage = {"input_tokens": 100, "output_tokens": 0, "cache_creation_input_tokens": 3000, "cache_read_input_tokens": 7000}
    expected = 100 * 2.00 / 1e6 + 3000 * 2.50 / 1e6 + 7000 * 0.20 / 1e6
    assert cost_usd(usage, batch=False, model="claude-sonnet-5-5") == approx(expected)


def test_pricing_sonnet_1h_cache_write_is_twice_input():
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 2000,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 2000},
    }
    assert cost_usd(usage, batch=False, model="claude-sonnet-5-5") == approx(2000 * 4.00 / 1e6)


def test_pricing_opus_has_no_long_context_tier():
    """150k input stays at $4/MTok (Haiku's >100k tier must not leak into the flat-rate models)."""
    usage = {"input_tokens": 150_000, "output_tokens": 1000}
    assert cost_usd(usage, batch=False, model="claude-opus-5-5") == approx(150_000 * 4.00 / 1e6 + 1000 * 20.00 / 1e6)


def test_pricing_batch_halves_opus_and_sonnet():
    usage = {"input_tokens": 1000, "output_tokens": 1000, "cache_read_input_tokens": 1000}
    opus = (1000 * 4.00 + 1000 * 20.00 + 1000 * 0.20) / 1e6
    sonnet = (1000 * 2.00 + 1000 * 10.00 + 1000 * 0.20) / 1e6
    assert cost_usd(usage, batch=True, model="claude-opus-5-5") == approx(opus * 0.5)
    assert cost_usd(usage, batch=True, model="claude-sonnet-5-5") == approx(sonnet * 0.5)


def test_pricing_haiku_by_name_matches_default():
    usage = {"input_tokens": 150_000, "output_tokens": 1000}
    assert cost_usd(usage, batch=False, model="claude-haiku-5-5") == approx(150_000 * 0.50 / 1e6 + 1000 * 2.50 / 1e6)


def test_pricing_unknown_model_raises_instead_of_haiku_price():
    with pytest.raises(ValueError, match="claude-opus-9"):
        cost_usd({"input_tokens": 1000, "output_tokens": 1000}, batch=False, model="claude-opus-9")
