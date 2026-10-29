"""Batch path: run(mode="batch") — B1–B8, plus the batch side of C3/C4."""

from __future__ import annotations

import json

import pytest

from tests.fakes import SimulatedCrash, prompt_of
from tests.helpers import BATCH_COST, make_req, read_jsonl, run_batch


def approx(x: float):
    return pytest.approx(x, rel=1e-9, abs=1e-15)


def submitted_custom_ids(fake) -> list[str]:
    return [r["custom_id"] for chunk in fake.batch_creates for r in chunk]


def submitted_prompts(fake) -> list[str]:
    return [prompt_of(r["params"]) for chunk in fake.batch_creates for r in chunk]


# ---------------------------------------------------------------- B1 submit misses only


def test_B1_submits_only_cache_misses(llm, fake):
    a, b, c = make_req("a"), make_req("b"), make_req("c")
    llm.call(a)
    out = run_batch(llm, [a, b, c])
    assert len(fake.batch_creates) == 1
    assert sorted(submitted_prompts(fake)) == ["b", "c"]
    assert out[a.key].cached is True


def test_B1_custom_id_is_req_key_and_params_are_req_params(llm, fake):
    a, b = make_req("a"), make_req("b", system="Be terse.")
    run_batch(llm, [a, b])
    [chunk] = fake.batch_creates
    assert {r["custom_id"]: r["params"] for r in chunk} == {a.key: a.params, b.key: b.params}


def test_C3_batch_submits_duplicate_key_once_and_answers_every_req(llm, fake):
    a = make_req("a")
    a_dup = make_req("a", meta={"stage": "judge"})
    b = make_req("b")
    out = run_batch(llm, [a, a_dup, b])
    assert sorted(submitted_custom_ids(fake)) == sorted([a.key, b.key])
    assert set(out) == {a.key, b.key}
    assert out[a_dup.key].text == "answer:a"


# ---------------------------------------------------------------- B2 chunking


def test_B2_chunks_by_max_batch_requests(make_llm, fake):
    llm = make_llm(max_batch_requests=2)
    reqs = [make_req(f"q{i}") for i in range(5)]
    out = run_batch(llm, reqs)
    assert len(fake.batch_creates) == 3  # ceil(5 / 2)
    assert all(len(chunk) <= 2 for chunk in fake.batch_creates)
    assert sorted(submitted_custom_ids(fake)) == sorted(r.key for r in reqs)
    assert all(out[r.key].error is None for r in reqs)


def test_B2_chunks_by_max_batch_bytes(make_llm, fake):
    reqs = [make_req(f"q{i}-" + "x" * 1000) for i in range(5)]
    compact = [len(json.dumps(r.params, separators=(",", ":"))) for r in reqs]
    spaced = [len(json.dumps(r.params)) for r in reqs]
    limit = 2600
    # Premise, for any reasonable serialization (compact or spaced, with or without the
    # {"custom_id": <64 chars>, "params": ...} wrapper): two requests fit, three never do.
    assert 2 * (max(spaced) + 120) <= limit < 3 * min(compact)
    llm = make_llm(max_batch_bytes=limit)
    run_batch(llm, reqs)
    assert all(len(chunk) <= 2 for chunk in fake.batch_creates)
    assert sorted(submitted_custom_ids(fake)) == sorted(r.key for r in reqs)


# ---------------------------------------------------------------- B3 manifest before polling


def _metas_in(manifest_metas) -> list[dict]:
    if isinstance(manifest_metas, dict):
        return list(manifest_metas.values())
    return list(manifest_metas)


def test_B3_manifest_is_written_after_create_and_before_polling(llm, fake, tmp_path):
    """The process 'dies' on the very first retrieve, so the manifest must already be on disk."""
    a = make_req("a", meta={"conv_id": "conv-a", "question_id": "qa"})
    b = make_req("b", meta={"conv_id": "conv-b", "question_id": "qb"})
    fake.crash_on_next_retrieve = True
    with pytest.raises(SimulatedCrash):
        run_batch(llm, [a, b])
    [batch_id] = fake.batch_ids
    manifest = json.loads((tmp_path / "batches" / f"{batch_id}.json").read_text())
    assert manifest["batch_id"] == batch_id
    assert manifest["run_id"] == "r1"
    assert sorted(manifest["custom_ids"]) == sorted([a.key, b.key])
    assert manifest["created_at"]
    metas = _metas_in(manifest["metas"])
    assert a.meta in metas and b.meta in metas


# ---------------------------------------------------------------- B4 resume


def test_B4_resumes_inflight_manifest_without_resubmitting(make_llm, fake):
    reqs = [make_req("a"), make_req("b")]
    fake.crash_on_next_retrieve = True
    with pytest.raises(SimulatedCrash):
        run_batch(make_llm(), reqs)
    [batch_id] = fake.batch_ids
    polls_before = len(fake.retrieved_ids)

    out = run_batch(make_llm(), reqs)  # a NEW LLM instance on the same root = restarted process

    assert len(fake.batch_creates) == 1, "the crashed batch must never be submitted a second time"
    assert batch_id in fake.retrieved_ids[polls_before:]
    for r in reqs:
        assert out[r.key].text == f"answer:{prompt_of(r.params)}"
        assert out[r.key].batch_id == batch_id


# ---------------------------------------------------------------- B5 poll + collect


def test_B5_polls_every_poll_s_until_ended(llm, fake, sleeps):
    fake.polls_until_ended = 3  # three "in_progress" answers, then "ended"
    run_batch(llm, [make_req("a")], poll_s=7)  # not the default 30, so a hard-coded interval fails
    [batch_id] = fake.batch_ids
    assert fake.retrieved_ids == [batch_id] * 4
    assert len(sleeps) >= 3
    assert all(s == 7 for s in sleeps)


def test_B5_collects_results_once_after_ended(llm, fake):
    fake.polls_until_ended = 2
    run_batch(llm, [make_req("a"), make_req("b")])
    [batch_id] = fake.batch_ids
    assert fake.results_ids == [batch_id]
    assert fake.batch_observed_ended(batch_id)
    assert fake.methods.index("messages.batches.results") > max(
        i for i, m in enumerate(fake.methods) if m == "messages.batches.retrieve"
    )


def test_B5_matches_out_of_order_results_by_custom_id(llm, fake):
    fake.results_reversed = True
    for prompt, text in [("a", "alpha"), ("b", "beta"), ("c", "gamma")]:
        fake.reply(prompt, text=text)
    reqs = {p: make_req(p) for p in "abc"}
    out = run_batch(llm, list(reqs.values()))
    [batch_id] = fake.batch_ids
    assert {p: out[r.key].text for p, r in reqs.items()} == {"a": "alpha", "b": "beta", "c": "gamma"}
    for r in reqs.values():
        assert out[r.key].message["id"] == fake.message_id_for(batch_id, r.key)


def test_B5_collected_manifest_is_marked_done_and_not_polled_again(make_llm, fake):
    """b errors (invalid request, so uncached); a later run must submit b fresh, not re-poll the done batch."""
    a, b = make_req("a"), make_req("b")
    fake.batch_outcomes("b", "errored_invalid")
    run_batch(make_llm(), [a, b])
    [first_batch] = fake.batch_ids
    polls_before = len(fake.retrieved_ids)

    out = run_batch(make_llm(), [a, b])

    assert first_batch not in fake.retrieved_ids[polls_before:]
    assert out[b.key].error is None
    assert out[b.key].text == "answer:b"


# ---------------------------------------------------------------- B6 result handling


def test_B6_succeeded_result_priced_at_batch_rate(llm, fake):
    a = make_req("a")
    out = run_batch(llm, [a])
    assert out[a.key].cost_usd == approx((1000 * 0.10 / 1e6 + 200 * 0.50 / 1e6) * 0.5)
    assert (out[a.key].mode, out[a.key].batch_id, out[a.key].error) == ("batch", fake.batch_ids[0], None)


def test_B6_succeeded_result_is_cached(llm, fake):
    a = make_req("a")
    run_batch(llm, [a])
    res = llm.call(a)
    assert res.cached is True
    assert fake.create_calls == []


def test_B6_succeeded_result_is_written_to_ledger_as_batch(llm, fake, tmp_path):
    a = make_req("a", meta={"stage": "judge"})
    run_batch(llm, [a])
    [row] = read_jsonl(tmp_path / "ledger.jsonl")
    assert (row["key"], row["mode"], row["stage"], row["run_id"]) == (a.key, "batch", "judge", "r1")
    assert row["cost_usd"] == approx(BATCH_COST)
    assert llm.spent_usd == approx(BATCH_COST)


def test_B6_invalid_request_error_sets_result_error_and_is_not_retried(llm, fake):
    a, b = make_req("a"), make_req("b")
    fake.batch_outcomes("b", "errored_invalid")
    out = run_batch(llm, [a, b])
    assert isinstance(out[b.key].error, str) and out[b.key].error
    assert out[a.key].error is None
    assert len(fake.batch_creates) == 1
    assert fake.create_calls == []


TRANSIENT = ["errored_api", "errored_overloaded", "expired", "canceled"]


@pytest.mark.parametrize("outcome", TRANSIENT)
def test_B6_transient_failure_is_resubmitted_once_in_a_new_batch(llm, fake, outcome):
    a, b = make_req("a"), make_req("b")
    fake.batch_outcomes("b", outcome)
    out = run_batch(llm, [a, b])
    assert len(fake.batch_creates) == 2
    assert [r["custom_id"] for r in fake.batch_creates[1]] == [b.key]
    assert out[b.key].error is None
    assert out[b.key].text == "answer:b"
    assert out[b.key].batch_id == fake.batch_ids[1]


@pytest.mark.parametrize("outcome", TRANSIENT)
def test_B6_failure_after_one_resubmission_sets_result_error(llm, fake, outcome):
    b = make_req("b")
    fake.batch_outcomes("b", outcome, outcome)
    out = run_batch(llm, [b])
    assert len(fake.batch_creates) == 2
    assert isinstance(out[b.key].error, str) and out[b.key].error
    assert fake.create_calls == []


def test_C4_batch_errored_result_is_not_cached(llm, fake):
    b = make_req("b")
    fake.batch_outcomes("b", "errored_invalid")
    run_batch(llm, [b])
    out = run_batch(llm, [b])
    assert submitted_custom_ids(fake) == [b.key, b.key]
    assert out[b.key].cached is False
    assert out[b.key].error is None


# ---------------------------------------------------------------- B7 timeout
# Uses a sleep that really waits a few ms, so the timeout elapses whether it is measured by clock or by sleeps.


def _timeout_run(make_llm, fake, tiny_real_sleep):
    fake.never_end = True
    fake.succeed_before_cancel = {"a"}
    fake.max_retrieves = 200  # an implementation that never cancels fails in ~2 s instead of hanging
    reqs = {p: make_req(p) for p in "abc"}
    llm = make_llm(sleep=tiny_real_sleep)
    out = run_batch(llm, list(reqs.values()), poll_s=0.01, batch_timeout_s=0.05)
    return reqs, out


def test_B7_timeout_cancels_batch_waits_for_ended_and_keeps_succeeded(make_llm, fake, tiny_real_sleep):
    reqs, out = _timeout_run(make_llm, fake, tiny_real_sleep)
    batch_id = fake.batch_ids[0]
    assert fake.canceled_ids == [batch_id]
    assert fake.batch_observed_ended(batch_id)
    a = out[reqs["a"].key]
    assert (a.mode, a.batch_id, a.text, a.error) == ("batch", batch_id, "answer:a", None)
    assert "a" not in fake.created_prompts


def test_B7_timeout_runs_remaining_requests_through_real_time_path(make_llm, fake, tiny_real_sleep):
    reqs, out = _timeout_run(make_llm, fake, tiny_real_sleep)
    assert sorted(fake.created_prompts) == ["b", "c"]
    assert len(fake.batch_creates) == 1
    for p in "bc":
        res = out[reqs[p].key]
        assert (res.mode, res.text, res.error) == ("sync", f"answer:{p}", None)


# ---------------------------------------------------------------- B8 prefix-group warm-up


def _grouped():
    g1 = [make_req(f"g1-{i}", meta={"prefix_group": "g1"}) for i in range(3)]
    g2 = [make_req(f"g2-{i}", meta={"prefix_group": "g2"}) for i in range(2)]
    solo = make_req("solo")
    return [g1[0], g2[0], g1[1], solo, g2[1], g1[2]]


def test_B8_first_request_of_each_group_completes_real_time_before_batch_submit(llm, fake):
    fake.hold_s = 0.05  # a warm-up that is merely started (not awaited) would end after the submit
    out = run_batch(llm, _grouped())
    assert sorted(fake.created_prompts) == ["g1-0", "g2-0"]
    first_submit = next(i for i, e in enumerate(fake.events) if e.startswith("messages.batches.create"))
    for p in ["g1-0", "g2-0"]:
        assert fake.events.index(f"messages.create:end:{p}") < first_submit
    assert all(r.error is None for r in out.values())


def test_B8_batch_contains_the_rest_of_each_group_and_ungrouped_reqs(llm, fake):
    reqs = _grouped()
    out = run_batch(llm, reqs)
    assert sorted(submitted_prompts(fake)) == ["g1-1", "g1-2", "g2-1", "solo"]
    assert out[reqs[0].key].mode == "sync" and out[reqs[1].key].mode == "sync"
    assert len(out) == 6
