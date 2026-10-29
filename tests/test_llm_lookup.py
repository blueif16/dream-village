"""C6: lookup(req) is a cache read only — hit -> the cached Result, miss -> None, never an API call."""

from __future__ import annotations

from tests.helpers import make_req, read_jsonl, run_batch


def test_C6_lookup_hit_returns_the_cached_result_without_a_client_call(llm, fake):
    req = make_req("hello")
    paid = llm.call(req)
    calls_before = len(fake.calls)

    hit = llm.lookup(req)

    assert hit is not None
    assert (hit.key, hit.text, hit.cached, hit.cost_usd, hit.error) == (req.key, paid.text, True, 0.0, None)
    assert len(fake.calls) == calls_before


def test_C6_lookup_miss_returns_none_and_never_reaches_the_api(llm, fake, tmp_path):
    assert llm.lookup(make_req("never sent")) is None
    assert fake.calls == []
    assert read_jsonl(tmp_path / "logs" / "requests.jsonl") == []
    assert read_jsonl(tmp_path / "ledger.jsonl") == []
    assert llm.spent_usd == 0.0


def test_C6_lookup_sees_batch_results_cached_by_another_instance_on_the_same_root(make_llm, fake):
    req = make_req("from a batch")
    run_batch(make_llm(run_id="prefetch"), [req])

    hit = make_llm(run_id="build").lookup(req)

    assert hit is not None
    assert (hit.text, hit.mode, hit.cached) == ("answer:from a batch", "batch", True)
