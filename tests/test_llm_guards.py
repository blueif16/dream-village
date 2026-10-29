"""Guards G1–G4: checked before any network call (spec: "Guards")."""

from __future__ import annotations

import pytest

from memobench.llm import BudgetExceeded, NeedsApproval
from tests.helpers import make_req, run_batch, run_sync

FROZEN = [("temperature", 0.0), ("top_p", 0.9), ("top_k", 5), ("seed", 1)]
FROZEN_IDS = [name for name, _ in FROZEN]


# ---------------------------------------------------------------- G1 frozen sampling params


@pytest.mark.parametrize(("param", "value"), FROZEN, ids=FROZEN_IDS)
def test_G1_call_rejects_frozen_sampling_param_without_network(llm, fake, param, value):
    with pytest.raises(ValueError):
        llm.call(make_req("hi", **{param: value}))
    assert fake.calls == []


@pytest.mark.parametrize("mode", ["sync", "batch"])
@pytest.mark.parametrize(("param", "value"), FROZEN, ids=FROZEN_IDS)
def test_G1_run_rejects_frozen_param_before_any_request_is_sent(llm, fake, param, value, mode):
    """One bad request in the list -> ValueError and NOTHING is sent, not even the valid ones."""
    reqs = [make_req("ok-1"), make_req("bad", **{param: value}), make_req("ok-2")]
    with pytest.raises(ValueError):
        llm.run(reqs, mode=mode, estimate_usd=1.0)
    assert fake.calls == []


# ---------------------------------------------------------------- G2 allowed models


def test_G2_call_rejects_model_not_in_allowed_models_without_network(llm, fake):
    with pytest.raises(ValueError):
        llm.call(make_req("hi", model="claude-opus-5-5"))
    assert fake.calls == []


@pytest.mark.parametrize("mode", ["sync", "batch"])
def test_G2_run_rejects_disallowed_model_before_any_request_is_sent(llm, fake, mode):
    reqs = [make_req("ok"), make_req("bad", model="claude-sonnet-5-5")]
    with pytest.raises(ValueError):
        llm.run(reqs, mode=mode, estimate_usd=1.0)
    assert fake.calls == []


def test_G2_model_in_configured_allowed_models_is_accepted(make_llm, fake):
    llm = make_llm(allowed_models=("claude-sonnet-5-5",))
    res = llm.call(make_req("hi", model="claude-sonnet-5-5"))
    assert res.text == "answer:hi"
    assert len(fake.create_calls) == 1


def test_G2_default_model_rejected_when_not_in_configured_allowed_models(make_llm, fake):
    llm = make_llm(allowed_models=("claude-sonnet-5-5",))
    with pytest.raises(ValueError):
        llm.call(make_req("hi"))  # claude-haiku-5-5
    assert fake.calls == []


# ---------------------------------------------------------------- G3 approval


@pytest.mark.parametrize("mode", ["sync", "batch"])
def test_G3_estimate_above_approval_raises_needs_approval_and_sends_nothing(llm, fake, mode):
    with pytest.raises(NeedsApproval):
        llm.run([make_req("a"), make_req("b")], mode=mode, estimate_usd=50.01)  # approval_usd default 50.0
    assert fake.calls == []


def test_G3_approved_run_above_threshold_proceeds(make_llm, fake):
    llm = make_llm(budget_usd=100.0)  # keep G4 out of the picture
    req = make_req("a")
    out = llm.run([req], mode="sync", estimate_usd=50.01, approved=True)
    assert out[req.key].text == "answer:a"
    assert len(fake.create_calls) == 1


def test_G3_estimate_equal_to_approval_does_not_need_approval(make_llm, fake):
    """'exceeds' is strict: 50.0 is not above approval_usd=50.0."""
    llm = make_llm(budget_usd=100.0)  # keep G4 out of the picture
    req = make_req("a")
    out = llm.run([req], mode="sync", estimate_usd=50.0)
    assert out[req.key].text == "answer:a"
    assert len(fake.create_calls) == 1


def test_G3_uses_llm_estimate_when_estimate_usd_is_none(make_llm, fake):
    """approval_usd is far below any real request's cost, so llm.estimate(...) must exceed it."""
    llm = make_llm(approval_usd=1e-12)
    with pytest.raises(NeedsApproval):
        llm.run([make_req("a")], mode="sync")
    assert fake.calls == []


# ---------------------------------------------------------------- G4 budget
# Each default fake response costs 1000*0.10/1e6 + 200*0.50/1e6 = 0.0002 USD (> budget 0.0001 below).


def test_G4_call_raises_budget_exceeded_once_total_reaches_budget(make_llm, fake):
    llm = make_llm(budget_usd=0.0001)
    llm.call(make_req("a"))  # total 0.0 < budget -> dispatched; total becomes 0.0002
    with pytest.raises(BudgetExceeded):
        llm.call(make_req("b"))
    assert fake.created_prompts == ["a"]


def test_G4_total_exactly_equal_to_budget_blocks_next_request(make_llm, fake):
    first = make_llm()
    first.call(make_req("a"))
    llm = make_llm(budget_usd=first.total_usd)  # total_usd >= budget_usd (equal)
    with pytest.raises(BudgetExceeded):
        llm.call(make_req("b"))
    assert fake.created_prompts == ["a"]


def test_G4_budget_counts_spend_of_previous_runs_from_ledger(make_llm, fake):
    make_llm(run_id="r1").call(make_req("a"))
    llm = make_llm(run_id="r2", budget_usd=0.0001)
    with pytest.raises(BudgetExceeded):
        llm.call(make_req("b", run_id="r2"))
    assert fake.created_prompts == ["a"]


def test_G4_sync_run_stops_dispatching_when_budget_reached(make_llm, fake):
    llm = make_llm(budget_usd=0.0001)
    with pytest.raises(BudgetExceeded):
        run_sync(llm, [make_req("a"), make_req("b"), make_req("c")], concurrency=1)
    assert len(fake.create_calls) == 1


def test_G4_batch_not_submitted_when_budget_already_reached(make_llm, fake):
    llm = make_llm(budget_usd=0.0001)
    llm.call(make_req("a"))
    with pytest.raises(BudgetExceeded):
        run_batch(llm, [make_req("b"), make_req("c")])
    assert fake.batch_creates == []
