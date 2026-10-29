"""Request log, text extraction, ledger, stop reasons: L1–L4."""

from __future__ import annotations

import pytest

from tests.fakes import text_block, thinking_block, usage_payload
from tests.helpers import BATCH_COST, MODEL, SYNC_COST, make_req, read_jsonl, run_batch


def approx(x: float):
    return pytest.approx(x, rel=1e-9, abs=1e-15)


def log_records(tmp_path) -> list[dict]:
    return read_jsonl(tmp_path / "logs" / "requests.jsonl")


def ledger_rows(tmp_path) -> list[dict]:
    return read_jsonl(tmp_path / "ledger.jsonl")


L1_FIELDS = [
    "run_id", "system", "conv_id", "question_id", "stage", "model", "effort", "mode", "batch_id", "key",
    "request", "response", "usage", "visible_text_chars", "stop_reason", "latency_ms", "cost_usd", "ts",
]  # fmt: skip
USAGE_FIELDS = ["input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"]

THINK_THEN_TEXT = [thinking_block("Let me reason about this at length first."), text_block("Hello "), text_block("world")]


# ---------------------------------------------------------------- L1 request log


def test_L1_one_record_per_api_response(llm, tmp_path):
    llm.call(make_req("a"))
    llm.call(make_req("b"))
    assert len(log_records(tmp_path)) == 2


def test_L1_cache_hit_is_not_an_api_response_and_adds_no_record(llm, tmp_path):
    req = make_req("a")
    llm.call(req)
    llm.call(req)
    assert len(log_records(tmp_path)) == 1


def test_L1_record_has_every_spec_field(llm, tmp_path):
    llm.call(make_req("a", output_config={"effort": "low"}))
    [rec] = log_records(tmp_path)
    assert set(L1_FIELDS) <= set(rec)
    assert set(USAGE_FIELDS) <= set(rec["usage"])


def test_L1_record_takes_identity_fields_from_meta(llm, tmp_path):
    """`system` is meta's system name, not the params' system prompt."""
    req = make_req(
        "a",
        system="You are terse.",
        meta={"system": "memora", "conv_id": "conv-7", "question_id": "q-42", "stage": "judge"},
    )
    llm.call(req)
    [rec] = log_records(tmp_path)
    got = {k: rec[k] for k in ("run_id", "system", "conv_id", "question_id", "stage")}
    assert got == {"run_id": "r1", "system": "memora", "conv_id": "conv-7", "question_id": "q-42", "stage": "judge"}


def test_L1_record_has_key_model_full_request_and_full_response(llm, fake, tmp_path):
    fake.reply("a", content=THINK_THEN_TEXT)
    req = make_req("a", system="You are terse.")
    llm.call(req)
    [rec] = log_records(tmp_path)
    assert (rec["key"], rec["model"]) == (req.key, MODEL)
    assert rec["request"] == req.params
    assert rec["response"]["id"] == fake.responses[-1].id
    assert [b["type"] for b in rec["response"]["content"]] == ["thinking", "text", "text"]


def test_L1_record_effort_comes_from_output_config(llm, tmp_path):
    llm.call(make_req("a", output_config={"effort": "low"}))
    [rec] = log_records(tmp_path)
    assert rec["effort"] == "low"


def test_L1_record_usage_cost_stop_reason_and_latency_match_response(llm, fake, tmp_path):
    fake.reply("a", text="hi", usage=usage_payload(1200, 300, 0, 500))
    llm.call(make_req("a"))
    [rec] = log_records(tmp_path)
    assert {k: rec["usage"][k] for k in USAGE_FIELDS} == {
        "input_tokens": 1200,
        "output_tokens": 300,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 500,
    }
    assert rec["cost_usd"] == approx(1200 * 0.10 / 1e6 + 500 * 0.01 / 1e6 + 300 * 0.50 / 1e6)
    assert rec["stop_reason"] == "end_turn"
    assert isinstance(rec["latency_ms"], (int, float)) and rec["latency_ms"] >= 0
    assert rec["ts"]


def test_L1_visible_text_chars_counts_text_blocks_only(llm, fake, tmp_path):
    fake.reply("a", content=THINK_THEN_TEXT)
    llm.call(make_req("a"))
    [rec] = log_records(tmp_path)
    assert rec["visible_text_chars"] == len("Hello world")


def test_L1_sync_record_has_sync_mode_and_null_batch_id(llm, tmp_path):
    llm.call(make_req("a"))
    [rec] = log_records(tmp_path)
    assert (rec["mode"], rec["batch_id"]) == ("sync", None)


def test_L1_batch_record_has_batch_mode_batch_id_and_batch_cost(llm, fake, tmp_path):
    req = make_req("a")
    run_batch(llm, [req])
    [rec] = log_records(tmp_path)
    assert (rec["mode"], rec["batch_id"], rec["key"]) == ("batch", fake.batch_ids[0], req.key)
    assert rec["cost_usd"] == approx(BATCH_COST)


# ---------------------------------------------------------------- L2 text


def test_L2_text_concatenates_text_blocks_and_skips_thinking(llm, fake):
    fake.reply("a", content=THINK_THEN_TEXT)
    assert llm.call(make_req("a")).text == "Hello world"


def test_L2_result_message_is_the_full_response(llm, fake):
    fake.reply("a", content=THINK_THEN_TEXT)
    res = llm.call(make_req("a"))
    assert res.message["id"] == fake.responses[-1].id
    assert [b["type"] for b in res.message["content"]] == ["thinking", "text", "text"]


def test_L2_batch_text_concatenates_text_blocks_and_skips_thinking(llm, fake):
    fake.reply("a", content=THINK_THEN_TEXT)
    req = make_req("a")
    assert run_batch(llm, [req])[req.key].text == "Hello world"


# ---------------------------------------------------------------- L3 ledger


def test_L3_ledger_appends_one_row_per_paid_response(llm, tmp_path):
    req = make_req("a", meta={"stage": "judge"})
    llm.call(req)
    [row] = ledger_rows(tmp_path)
    assert (row["run_id"], row["stage"], row["key"], row["mode"]) == ("r1", "judge", req.key, "sync")
    assert row["cost_usd"] == approx(SYNC_COST)
    assert row["ts"]


def test_L3_cache_hit_adds_no_ledger_row(llm, tmp_path):
    req = make_req("a")
    llm.call(req)
    llm.call(req)
    assert len(ledger_rows(tmp_path)) == 1


def test_L3_total_usd_sums_the_ledger_across_runs(make_llm):
    make_llm(run_id="r1").call(make_req("a"))
    llm = make_llm(run_id="r2")
    llm.call(make_req("b", run_id="r2"))
    llm.call(make_req("c", run_id="r2"))
    assert llm.total_usd == approx(3 * SYNC_COST)


def test_L3_spent_usd_counts_only_this_instance(make_llm):
    first = make_llm(run_id="r1")
    first.call(make_req("a"))
    second = make_llm(run_id="r2")
    second.call(make_req("b", run_id="r2"))
    second.call(make_req("c", run_id="r2"))
    assert second.spent_usd == approx(2 * SYNC_COST)
    assert first.spent_usd == approx(SYNC_COST)


def test_L3_fresh_root_has_zero_total_and_spent(llm):
    assert llm.total_usd == 0
    assert llm.spent_usd == 0


# ---------------------------------------------------------------- L4 stop reasons


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_L4_stop_reason_is_recorded_on_result_not_raised(llm, fake, stop_reason):
    fake.reply("a", text="partial", stop_reason=stop_reason)
    res = llm.call(make_req("a"))
    assert res.stop_reason == stop_reason
    assert res.text == "partial"


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_L4_stop_reason_is_recorded_in_log(llm, fake, tmp_path, stop_reason):
    fake.reply("a", text="partial", stop_reason=stop_reason)
    llm.call(make_req("a"))
    [rec] = log_records(tmp_path)
    assert rec["stop_reason"] == stop_reason
