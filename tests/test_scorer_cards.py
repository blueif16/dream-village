"""scorer.cards: click-only context cards for human labels (design §7), verdict hidden, stratified pass/fail."""

from __future__ import annotations

import csv
import json
from collections import Counter

import pytest

from scorer import cards as C
from scorer.gold import Gold
from scorer.probes import Probe
from tests import scorer_village as V

T = "2025-09-12T12:00:00"


@pytest.fixture
def gold(tmp_path):
    return Gold(V.build(tmp_path / "village"))


def card(i, typ, passed):
    return {"card_id": f"{typ}-{i}", "type": typ, "verdict": {"pass": passed}}


@pytest.mark.parametrize(
    ("n_pass", "n_fail", "want_pass", "want_fail"),
    [(200, 200, 50, 50), (20, 70, 20, 70), (20, 200, 20, 80), (0, 30, 0, 30), (120, 10, 90, 10)],
)
def test_stratify_takes_up_to_100_per_type_with_50_each_side_where_available(n_pass, n_fail, want_pass, want_fail):
    cards = [card(i, "stale", True) for i in range(n_pass)] + [card(i + 1000, "stale", False) for i in range(n_fail)]
    picked = C.stratify(cards)
    assert Counter(c["verdict"]["pass"] for c in picked) == {k: v for k, v in ((True, want_pass), (False, want_fail)) if v}


def test_stratify_is_per_type_and_deterministic():
    cards = [card(i, t, i % 2 == 0) for t in ("must_not", "stale", "fabricated") for i in range(300)]
    picked = C.stratify(cards)
    assert Counter(c["type"] for c in picked) == {"must_not": 100, "stale": 100, "fabricated": 100}
    assert [c["card_id"] for c in picked] == [c["card_id"] for c in C.stratify(list(reversed(cards)))]


def test_an_answer_card_shows_the_request_response_and_the_one_entry(gold):
    p = Probe("p1", V.P, "cp", T, "fact", "dinner ideas?", ["kai.f02"], [], [], ["kai.f01"], [], {}, None, {})
    verdicts = [{"probe_id": "p1", "system": "v0", "check": "stale:kai.f01", "type": "stale", "item_id": "kai.f01",
                 "pass": False, "decided_by": "judges", "primary": {"pass": False}, "second": {"pass": True}}]  # fmt: skip
    answers = [{"probe_id": "p1", "system": "v0", "text": "Carnitas tacos!"}]
    [c] = C.answer_cards(gold, [p], verdicts, answers)
    assert c["type"] == "stale" and c["request"] == "dinner ideas?" and c["response"] == "Carnitas tacos!"
    assert c["entry"]["value"].startswith("eats everything") and c["entry"]["valid_to"] == "2025-09-05"
    assert "carnitas" in c["entry"]["disclosure_excerpt"]
    assert c["verdict"]["pass"] is False


def test_the_previous_value_is_shown_for_a_changed_entry(gold):
    p = Probe("p1", V.P, "cp", T, "fact", "dinner ideas?", ["kai.f02"], [], ["kai.f02"], [], [], {}, None, {})
    verdicts = [{"probe_id": "p1", "system": "v0", "check": "absent:kai.f02", "type": "absent", "item_id": "kai.f02",
                 "pass": True, "decided_by": "judges", "primary": None, "second": None}]  # fmt: skip
    [c] = C.answer_cards(gold, [p], verdicts, [{"probe_id": "p1", "system": "v0", "text": "Pasta."}])
    assert c["type"] == "must_not" and c["entry"]["previous_value"].startswith("eats everything")


def test_fabricated_cards_come_from_audited_lines_judged_when_first_seen(gold):
    rel = [{"id": "kai.f02", "relation": "consistent", "current": True}]
    audit = {"principals": {V.P: {"series": [], "mapping": [
        {"file": "/topics/food.md", "line": "- vegetarian", "first_seen": "2025-09-06T00:00:00", "kind": "fact",
         "entries": rel},
        {"file": "/topics/food.md", "line": "- is vegetarian", "first_seen": "2025-09-02T00:00:00", "kind": "fact",
         "entries": rel},  # written before the change was told: fabricated
        {"file": "/topics/x.md", "line": "- owns a boat", "first_seen": "2025-09-06T00:00:00", "kind": "fact",
         "entries": []},
        {"file": "/topics/x.md", "line": "# heading-like note", "first_seen": "2025-09-06T00:00:00",
         "kind": "not_a_claim", "entries": []},
    ]}}}  # fmt: skip
    cards = C.fabricated_cards(gold, audit)
    assert [(c["response"], c["verdict"]["pass"]) for c in cards] == [
        ("/topics/food.md: - vegetarian", True), ("/topics/food.md: - is vegetarian", False),
        ("/topics/x.md: - owns a boat", False)]  # fmt: skip
    unmapped = cards[2]["entry"]
    assert len(unmapped["candidates"]) == 3  # no mapped entry: the closest ledger entries give context
    assert unmapped["id"] == "" and unmapped["value"].startswith("(no matching entry)")  # never posed as THE entry


def test_export_hides_the_verdict_from_the_labeling_files(gold, tmp_path):
    cards = [{"card_id": "c1", "type": "must_not", "question": "q?", "request": "r", "response": "x", "t": T,
              "principal": V.P, "entry": {"id": "kai.f03", "value": "v", "valid_from": None, "valid_to": None,
                                          "previous_value": "", "disclosure_excerpt": "e"},
              "verdict": {"pass": False, "decided_by": "code"}}]  # fmt: skip
    C.export(tmp_path / "cards", cards)
    shown = (tmp_path / "cards" / "cards.jsonl").read_text()
    assert '"verdict"' not in shown and "decided_by" not in shown
    assert json.loads((tmp_path / "cards" / "cards_key.jsonl").read_text())["verdict"]["pass"] is False
    rows = list(csv.DictReader(open(tmp_path / "cards" / "cards.csv")))
    assert rows[0]["card_id"] == "c1" and rows[0]["label_yes_no"] == "" and "verdict" not in rows[0]
