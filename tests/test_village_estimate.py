"""village.estimate: the T2 dry run builds the requests it can without spending, prices them with LLM.estimate, and
extrapolates every later stage from measured pilot token profiles; it never calls the API."""

from __future__ import annotations

import json

from tests.fakes import FakeAnthropic
from village import estimate as est
from village import generate as g
from village.spec import FULL, PILOT


def write_profile(root, stage, model, n, usage, cost):
    (root / "logs").mkdir(parents=True, exist_ok=True)
    with open(root / "logs" / "requests.jsonl", "a") as f:
        for _ in range(n):
            f.write(json.dumps({"stage": stage, "model": model, "cost_usd": cost, "response": {"usage": usage}}) + "\n")


def test_profiles_average_tokens_per_request_by_stage(tmp_path):
    u = {"input_tokens": 100, "cache_creation_input_tokens": 1000, "cache_read_input_tokens": 3000, "output_tokens": 500}
    write_profile(tmp_path, "r6_render", "claude-sonnet-5-5", 4, u, 0.01)
    p = est.profiles([tmp_path])["r6_render"]
    assert (p.model, p.requests, p.input, p.cache_write, p.cache_read, p.output) == (
        "claude-sonnet-5-5", 4, 100, 1000, 3000, 500)  # fmt: skip


def test_later_roots_override_earlier_ones_per_stage(tmp_path):
    a, b = tmp_path / "v1", tmp_path / "v2"
    write_profile(a, "r7_verify", "claude-haiku-5-5", 2, {"input_tokens": 10, "output_tokens": 10}, 0.001)
    write_profile(b, "r7_verify", "claude-sonnet-5-5", 2, {"input_tokens": 20, "output_tokens": 20}, 0.01)
    write_profile(a, "r1_world", "claude-opus-5-5", 1, {"input_tokens": 5, "output_tokens": 5}, 0.05)
    p = est.profiles([a, b])
    assert p["r7_verify"].model == "claude-sonnet-5-5" and p["r1_world"].requests == 1


def test_counts_follow_the_spec(tmp_path):
    c = est.counts(FULL)
    assert c["r3_bible"] == 12 and c["r4_life_log"] == 12 * 52 and c["r5_plan"] == 12 * 52
    assert c["r6_render"] == round(52 * (4 * 6.5 + 5 * 3.5 + 3 * 1.0))  # daily 6-7, several 3-4, weekly 1 per week


def test_dry_run_prices_the_buildable_request_and_never_calls_the_api(tmp_path):
    u = {"input_tokens": 1000, "cache_creation_input_tokens": 5000, "cache_read_input_tokens": 10000, "output_tokens": 1000}
    for stage, model in [(s, g.STAGES[s].model) for s in g.STAGES]:
        write_profile(tmp_path / "pilot", stage, model, 3, u, 0.01)
    fake = FakeAnthropic()
    out = est.dry_run(FULL, [tmp_path / "pilot"], root=tmp_path / "llm", client=fake, regen=(0.3, 0.6))
    assert fake.calls == []
    assert out["buildable"]["r1_world"]["layer_estimate_usd"] > 0
    stages = {r["stage"]: r for r in out["stages"]}
    assert set(stages) == set(g.STAGES) | {"r6_render (regeneration)", "r7_verify (re-verify)"}
    assert all(0 < r["low_usd"] <= r["high_usd"] for r in stages.values())
    assert out["total"]["low_usd"] <= out["total"]["high_usd"]


def test_pilot_dry_run_scales_nothing(tmp_path):
    """On the pilot spec every growth factor is 1, so the estimate is the measured per-request cost times the counts."""
    u = {"input_tokens": 1000, "output_tokens": 1000}
    write_profile(tmp_path / "pilot", "r4_life_log", "claude-sonnet-5-5", 8, u, 0.0)
    out = est.dry_run(PILOT, [tmp_path / "pilot"], root=tmp_path / "llm", client=FakeAnthropic(), regen=(0.0, 0.0),
                      stages_only=["r4_life_log"])  # fmt: skip
    (row,) = out["stages"]
    per_request = (1000 * 2.00 + 1000 * 10.00) / 1e6 * 0.5
    assert row["requests"] == 8 and abs(row["low_usd"] - 8 * per_request) < 1e-9 and row["low_usd"] == row["high_usd"]


def test_profiles_can_ignore_requests_before_a_cutoff(tmp_path):
    (tmp_path / "logs").mkdir()
    rows = [{"stage": "r7_verify", "model": "claude-haiku-5-5", "cost_usd": 0, "ts": "2026-10-28T13:00:00+00:00",
             "response": {"usage": {"input_tokens": 999, "output_tokens": 1}}},
            {"stage": "r7_verify", "model": "claude-sonnet-5-5", "cost_usd": 0, "ts": "2026-10-28T15:00:00+00:00",
             "response": {"usage": {"input_tokens": 100, "output_tokens": 1}}}]  # fmt: skip
    (tmp_path / "logs" / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    p = est.profiles([tmp_path], since="2026-10-28T14:00:00+00:00")["r7_verify"]
    assert (p.model, p.requests, p.input) == ("claude-sonnet-5-5", 1, 100)


def test_cutoff_applies_only_to_the_latest_root(tmp_path):
    """Earlier roots (another village run) keep every request; the cutoff drops a superseded checkpoint on the last."""
    a, b = tmp_path / "v1", tmp_path / "v2"
    for root, stage in ((a, "r1_world"), (b, "r7_verify")):
        (root / "logs").mkdir(parents=True)
        (root / "logs" / "requests.jsonl").write_text(json.dumps({"stage": stage, "model": "claude-sonnet-5-5", "cost_usd": 0,
            "ts": "2026-10-28T10:00:00+00:00", "response": {"usage": {"input_tokens": 1, "output_tokens": 1}}}) + "\n")  # fmt: skip
    assert set(est.profiles([a, b], since="2026-10-28T12:00:00+00:00")) == {"r1_world"}


def test_every_named_village_but_the_pilot_has_growth_for_every_stage():
    """A missing stage would silently be priced at the pilot's size."""
    from village.spec import VILLAGES

    for name in VILLAGES:
        if name != "pilot":
            assert set(est.GROWTH[name]) == set(g.STAGES), name
