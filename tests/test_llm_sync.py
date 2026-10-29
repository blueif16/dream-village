"""Real-time path: call() / run(mode="sync"), cache C1–C4, S1–S3, thread safety T1."""

from __future__ import annotations

import hashlib
import json
import threading

import anthropic
import pytest

from memobench.llm import CostOverrun, Req
from tests.fakes import api_status_error, connection_error, timeout_error, usage_payload
from tests.helpers import MODEL, SYNC_COST, make_req, read_jsonl, run_sync


def approx(x: float):
    return pytest.approx(x, rel=1e-9, abs=1e-15)


# ---------------------------------------------------------------- call() result


def test_call_result_reflects_the_response(llm, fake):
    req = make_req("hello")
    res = llm.call(req)
    assert res.key == req.key
    assert res.text == "answer:hello"
    assert res.stop_reason == "end_turn"
    assert res.message["id"] == fake.responses[-1].id
    assert (res.usage["input_tokens"], res.usage["output_tokens"]) == (1000, 200)
    assert (res.cached, res.mode, res.batch_id, res.error) == (False, "sync", None, None)


def test_call_prices_result_at_real_time_rate(llm, fake):
    res = llm.call(make_req("hello"))
    assert res.cost_usd == approx(1000 * 0.10 / 1e6 + 200 * 0.50 / 1e6)


def test_call_sends_the_request_params_to_messages_create(llm, fake):
    req = make_req("hello", system="Be terse.", output_config={"effort": "low"})
    llm.call(req)
    [sent] = fake.create_calls
    assert {k: sent.get(k) for k in req.params} == req.params


# ---------------------------------------------------------------- C1 cache


def test_C1_repeated_call_is_served_from_cache_without_client_call(llm, fake):
    req = make_req("hello")
    first = llm.call(req)
    second = llm.call(req)
    assert len(fake.create_calls) == 1
    assert second.cached is True
    assert second.text == first.text


def test_C1_cache_hit_adds_nothing_to_spend(llm, fake):
    req = make_req("hello")
    llm.call(req)
    llm.call(req)
    assert llm.spent_usd == approx(SYNC_COST)
    assert llm.total_usd == approx(SYNC_COST)


def test_C1_cache_persists_in_root_cache_sqlite_across_instances(make_llm, fake, tmp_path):
    req = make_req("hello")
    make_llm(run_id="r1").call(req)
    assert (tmp_path / "cache.sqlite").is_file()
    res = make_llm(run_id="r2").call(req)
    assert res.cached is True
    assert len(fake.create_calls) == 1


def test_C1_sync_run_serves_cached_keys_and_sends_only_misses(llm, fake):
    a, b = make_req("a"), make_req("b")
    llm.call(a)
    out = run_sync(llm, [a, b])
    assert fake.created_prompts == ["a", "b"]
    assert out[a.key].cached is True
    assert out[b.key].cached is False


# ---------------------------------------------------------------- C2 key

BASE = {
    "model": MODEL,
    "max_tokens": 64,
    "system": "You are terse.",
    "messages": [{"role": "user", "content": "hi"}],
    "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": {"type": "object"}}},
}


def test_C2_key_is_sha256_hex_of_canonical_params_json():
    canonical = json.dumps(BASE, sort_keys=True, separators=(",", ":"))
    assert Req(params=BASE).key == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_C2_key_ignores_dict_ordering_at_every_level():
    reordered = {
        "output_config": {"format": {"schema": {"type": "object"}, "type": "json_schema"}, "effort": "low"},
        "messages": [{"content": "hi", "role": "user"}],
        "system": "You are terse.",
        "max_tokens": 64,
        "model": MODEL,
    }
    assert Req(params=reordered).key == Req(params=BASE).key


def test_C2_key_ignores_meta():
    a = Req(params=BASE, meta={"run_id": "r1", "stage": "answer", "conv_id": "c1"})
    b = Req(params=BASE, meta={"run_id": "r9", "stage": "judge", "question_id": "q7", "prefix_group": "g"})
    assert a.key == b.key == Req(params=BASE).key


CHANGES = {
    "model": lambda p: p.update(model="claude-sonnet-5-5"),
    "max_tokens": lambda p: p.update(max_tokens=65),
    "system": lambda p: p.update(system="You are verbose."),
    "message_text": lambda p: p["messages"][0].update(content="hi!"),
    "effort": lambda p: p["output_config"].update(effort="high"),
    "extra_param": lambda p: p.update(stop_sequences=["END"]),
}


@pytest.mark.parametrize("change", list(CHANGES), ids=list(CHANGES))
def test_C2_key_changes_when_any_param_changes(change):
    changed = json.loads(json.dumps(BASE))
    CHANGES[change](changed)
    assert Req(params=changed).key != Req(params=BASE).key


# ---------------------------------------------------------------- C3 dedup within run (sync)


def test_C3_sync_run_executes_duplicate_keys_once_and_answers_every_req(llm, fake):
    fake.hold_s = 0.05  # duplicates would overlap in flight, so the cache alone cannot hide a double call
    a = make_req("a")
    a_dup = make_req("a", meta={"stage": "judge"})  # same params, different meta -> same key
    b = make_req("b")
    out = run_sync(llm, [a, b, a_dup])
    assert sorted(fake.created_prompts) == ["a", "b"]
    assert set(out) == {a.key, b.key}
    assert out[a_dup.key].text == "answer:a"


# ---------------------------------------------------------------- C4 errors not cached (sync)


def test_C4_failed_call_is_not_cached(llm, fake):
    req = make_req("hello")
    fake.fail_next(api_status_error(400))
    with pytest.raises(anthropic.APIStatusError):
        llm.call(req)
    res = llm.call(req)
    assert len(fake.create_calls) == 2
    assert res.cached is False
    assert res.text == "answer:hello"


# ---------------------------------------------------------------- S1 thread pool


def test_S1_sync_run_runs_exactly_concurrency_requests_in_parallel(llm, fake):
    fake.hold_s = 0.1
    reqs = [make_req(f"q{i}") for i in range(8)]
    out = run_sync(llm, reqs, concurrency=4)
    assert fake.max_in_flight == 4
    assert len(out) == 8


# ---------------------------------------------------------------- S2 retries

RETRYABLE = {
    "429": lambda: api_status_error(429),
    "500": lambda: api_status_error(500),
    "503": lambda: api_status_error(503),
    "529_overloaded": lambda: api_status_error(529),
    "connection": connection_error,
    "timeout": timeout_error,
}


@pytest.mark.parametrize("kind", list(RETRYABLE), ids=list(RETRYABLE))
def test_S2_retryable_error_is_retried_then_succeeds(llm, fake, kind):
    fake.fail_next(RETRYABLE[kind]())
    res = llm.call(make_req("hello"))
    assert res.text == "answer:hello"
    assert len(fake.create_calls) == 2


def test_S2_each_retry_appends_a_record_to_retries_jsonl(llm, fake, tmp_path):
    fake.fail_next(api_status_error(429), api_status_error(529))
    llm.call(make_req("hello"))
    records = read_jsonl(tmp_path / "logs" / "retries.jsonl")
    assert len(records) == 2
    assert all(isinstance(r, dict) and r for r in records)


def test_S2_retry_honors_retry_after_header_via_injected_sleep(llm, fake, sleeps):
    fake.fail_next(api_status_error(429, headers={"retry-after": "37"}))
    llm.call(make_req("hello"))
    assert sleeps, "the retry wait must go through the injected sleep"
    assert max(sleeps) >= 37


def test_S2_gives_up_after_max_attempts_total_calls(make_llm, fake):
    llm = make_llm(max_attempts=3)
    fake.fail_next(*[api_status_error(429) for _ in range(10)])
    with pytest.raises(Exception):
        llm.call(make_req("hello"))
    assert len(fake.create_calls) == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413])
def test_S2_non_retryable_4xx_raises_immediately_after_one_call(llm, fake, tmp_path, status):
    fake.fail_next(api_status_error(status))  # a retry would succeed, so retrying is visible
    with pytest.raises(anthropic.APIStatusError) as ei:
        llm.call(make_req("hello"))
    assert ei.value.status_code == status
    assert len(fake.create_calls) == 1
    assert read_jsonl(tmp_path / "logs" / "retries.jsonl") == []


# ---------------------------------------------------------------- S3 overrun stop
# Each default response costs 1000*0.10/1e6 + 200*0.50/1e6 = 0.0002 USD.


def test_S3_overrun_stops_dispatch_and_raises_cost_overrun(llm, fake):
    """estimate 0.0002 -> limit 1.5*0.0002 = 0.0003; after 1 response 0.0002 (ok), after 2 0.0004 > 0.0003."""
    reqs = [make_req(f"q{i}") for i in range(4)]
    with pytest.raises(CostOverrun):
        llm.run(reqs, mode="sync", estimate_usd=0.0002, concurrency=1)
    assert len(fake.create_calls) == 2


def test_S3_requests_finished_before_overrun_stay_cached(llm, fake):
    reqs = {f"q{i}": make_req(f"q{i}") for i in range(4)}
    with pytest.raises(CostOverrun):
        llm.run(list(reqs.values()), mode="sync", estimate_usd=0.0002, concurrency=1)
    finished = list(fake.created_prompts)
    assert len(finished) == 2
    for prompt in finished:
        assert llm.call(reqs[prompt]).cached is True
    assert len(fake.create_calls) == 2


def test_S3_spend_between_1x_and_1_5x_estimate_does_not_raise(llm, fake):
    """3 responses = 0.0006 > estimate 0.0005 but <= 1.5*0.0005 = 0.00075 -> completes."""
    reqs = [make_req(f"q{i}") for i in range(3)]
    out = llm.run(reqs, mode="sync", estimate_usd=0.0005, concurrency=1)
    assert len(out) == 3
    assert len(fake.create_calls) == 3


# ---------------------------------------------------------------- T1 thread safety
# 16 threads x 4 distinct requests each, through ONE shared LLM instance, released together.
# Request i returns 1000 input + (100+i) output tokens, so every cost is distinct:
#   cost_i = 1000*0.10/1e6 + (100+i)*0.50/1e6

N_THREADS, PER_THREAD = 16, 4
N_REQS = N_THREADS * PER_THREAD


def _t1_reqs(fake) -> list[Req]:
    fake.hold_s = 0.002
    reqs = []
    for i in range(N_REQS):
        prompt = f"t1-{i}"
        fake.reply(prompt, text=f"answer:{prompt}", usage=usage_payload(1000, 100 + i))
        reqs.append(make_req(prompt))
    return reqs


def _hammer(llm, reqs: list[Req]) -> dict[str, object]:
    """call() every req from N_THREADS threads at once; return {key: Result}; re-raise any thread error."""
    barrier = threading.Barrier(N_THREADS)
    results, errors, lock = {}, [], threading.Lock()

    def worker(chunk):
        try:
            barrier.wait(timeout=10)
            for r in chunk:
                res = llm.call(r)
                with lock:
                    results[r.params["messages"][0]["content"]] = res
        except BaseException as e:  # noqa: BLE001 - surfaced below
            with lock:
                errors.append(e)

    chunks = [reqs[i::N_THREADS] for i in range(N_THREADS)]
    threads = [threading.Thread(target=worker, args=(c,)) for c in chunks]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    if errors:
        raise errors[0]
    return results


def test_T1_concurrent_calls_write_exactly_one_intact_log_line_each(llm, fake, tmp_path):
    reqs = _t1_reqs(fake)
    _hammer(llm, reqs)
    records = read_jsonl(tmp_path / "logs" / "requests.jsonl")  # json.loads fails on a torn line
    assert len(records) == N_REQS
    assert sorted(r["key"] for r in records) == sorted(r.key for r in reqs)


def test_T1_concurrent_calls_write_exactly_one_intact_ledger_line_each(llm, fake, tmp_path):
    reqs = _t1_reqs(fake)
    _hammer(llm, reqs)
    rows = read_jsonl(tmp_path / "ledger.jsonl")
    assert len(rows) == N_REQS
    assert sorted(r["key"] for r in rows) == sorted(r.key for r in reqs)


def test_T1_concurrent_calls_lose_no_cache_entries(llm, fake):
    reqs = _t1_reqs(fake)
    _hammer(llm, reqs)
    second = _hammer(llm, reqs)
    assert len(fake.create_calls) == N_REQS  # nothing paid twice
    assert all(res.cached for res in second.values())
    assert {p: res.text for p, res in second.items()} == {f"t1-{i}": f"answer:t1-{i}" for i in range(N_REQS)}


def test_T1_spent_usd_equals_sum_of_concurrent_call_costs(llm, fake):
    reqs = _t1_reqs(fake)
    _hammer(llm, reqs)
    expected = sum(1000 * 0.10 / 1e6 + (100 + i) * 0.50 / 1e6 for i in range(N_REQS))
    assert llm.spent_usd == approx(expected)
