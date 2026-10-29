"""scorer.label_agreement: human context-card labels vs the judges' hidden verdicts (design §7, D18, D33)."""

from __future__ import annotations

import json

import pytest

from scorer import label_agreement as LA

P, F = True, False


def judged(passed_primary, passed_second):
    """A key verdict decided by the two judges, combined by the §7 veto (either fails -> fail)."""
    j = lambda m, v: None if v is None else {"model": m, "pass": v, "evidence": "x"}  # noqa: E731
    votes = [v for v in (passed_primary, passed_second) if v is not None]
    return {"pass": all(votes), "decided_by": "judges", "code_rule": None,
            "primary": j("opus", passed_primary), "second": j("sonnet", passed_second), "system": "v0"}  # fmt: skip


def write(tmp_path, labels: list[tuple[str, str]], key: dict[str, dict]):
    lp, kp = tmp_path / "labels.jsonl", tmp_path / "cards_key.jsonl"
    lp.write_text("".join(json.dumps({"card_id": c, "label": lab, "ts": "2026-10-28T12:00:00Z"}) + "\n" for c, lab in labels))
    kp.write_text("".join(json.dumps({"card_id": c, "verdict": v}) + "\n" for c, v in key.items()))
    return lp, kp


def row(rows, typ, judge):
    [r] = [r for r in rows if r["type"] == typ and r["judge"] == judge]
    return r


# Ten stale cards worked by hand. Human: 6 yes, 4 no. Primary: TP=5 FN=1 TN=2 FP=2 (passes 7, fails 3).
#   TPR = 5/6, TNR = 2/4, observed agreement po = 7/10,
#   chance pe = P(h yes)P(j yes) + P(h no)P(j no) = .6*.7 + .4*.3 = .54, kappa = (.7 - .54) / (1 - .54) = .16/.46.
# The second judge passes everything: TPR 1, TNR 0, po = .6, pe = .6*1 + .4*0 = .6, kappa 0.
HUMAN = ["yes"] * 6 + ["no"] * 4
PRIMARY = [P, P, P, P, P, F, F, F, P, P]


def ten_stale():
    labels = [(f"stale|c{i}", h) for i, h in enumerate(HUMAN)]
    key = {f"stale|c{i}": judged(p, P) for i, p in enumerate(PRIMARY)}
    return labels, key


def test_tpr_tnr_and_kappa_match_the_hand_computed_values(tmp_path):
    labels, key = ten_stale()
    r = row(LA.rows(*write(tmp_path, labels, key)), "stale", "primary")
    assert (r["n"], r["yes"], r["no"]) == (10, 6, 4)
    assert r["tpr"] == pytest.approx(5 / 6)
    assert r["tnr"] == pytest.approx(2 / 4)
    assert r["kappa"] == pytest.approx(0.16 / 0.46)
    assert r["accepted"] is False


def test_a_judge_that_passes_everything_has_kappa_zero_and_tnr_zero(tmp_path):
    labels, key = ten_stale()
    r = row(LA.rows(*write(tmp_path, labels, key)), "stale", "second")
    assert (r["tpr"], r["tnr"]) == (1.0, 0.0)
    assert r["kappa"] == pytest.approx(0.0)


def test_skips_and_unlabeled_cards_are_excluded(tmp_path):
    labels, key = ten_stale()
    # three skipped cards the primary passes and two cards nobody labeled: counting a skip as yes would raise TPR,
    # counting it as no would lower TNR; either way n would move off 10
    labels += [(f"stale|skip{i}", "skip") for i in range(3)]
    key |= {f"stale|skip{i}": judged(P, P) for i in range(3)} | {f"stale|open{i}": judged(F, F) for i in range(2)}
    r = row(LA.rows(*write(tmp_path, labels, key)), "stale", "primary")
    assert (r["n"], r["yes"], r["no"]) == (10, 6, 4)
    assert (r["tpr"], r["tnr"]) == (pytest.approx(5 / 6), pytest.approx(2 / 4))
    assert r["kappa"] == pytest.approx(0.16 / 0.46)


def test_the_last_label_for_a_card_wins(tmp_path):
    labels, key = ten_stale()
    labels = [("stale|c0", "no")] + labels  # relabelled later as yes (HUMAN[0])
    r = row(LA.rows(*write(tmp_path, labels, key)), "stale", "primary")
    assert (r["n"], r["yes"]) == (10, 6)


@pytest.mark.parametrize(
    ("n_yes", "n_no", "tp", "tn", "accepted"),
    [
        (10, 10, 9, 9, True),  # TPR .9, TNR .9, kappa .8: all three exactly at the bar
        (10, 10, 10, 8, False),  # TNR .8
        (10, 10, 8, 10, False),  # TPR .8
        (40, 10, 36, 9, False),  # TPR .9, TNR .9, but kappa = (.9 - .644) / .356 = .72
    ],
)
def test_accepted_needs_tpr_and_tnr_at_least_09_and_kappa_at_least_08(tmp_path, n_yes, n_no, tp, tn, accepted):
    human = ["yes"] * n_yes + ["no"] * n_no
    judge = [i < tp for i in range(n_yes)] + [i >= tn for i in range(n_no)]
    labels = [(f"must_not|c{i}", h) for i, h in enumerate(human)]
    key = {f"must_not|c{i}": judged(j, P) for i, j in enumerate(judge)}
    r = row(LA.rows(*write(tmp_path, labels, key)), "must_not", "primary")
    assert r["accepted"] is accepted


def test_veto_fails_on_either_judge_and_d33_fails_only_when_both_do(tmp_path):
    #            human   primary second   veto  both(D33)
    cards = {"a": ("yes", P, F),  # F     P
             "b": ("no", F, P),  # F     P
             "c": ("no", F, F),  # F     F
             "d": ("yes", P, P)}  # P     P   # fmt: skip
    labels = [(f"must_not|{k}", h) for k, (h, _, _) in cards.items()] + [("must_not|code", "no")]
    key = {f"must_not|{k}": judged(p, s) for k, (_, p, s) in cards.items()}
    key["must_not|code"] = {"pass": False, "decided_by": "code", "code_rule": "verbatim", "primary": None,
                            "second": None, "system": "v0"}  # fmt: skip
    rows = LA.rows(*write(tmp_path, labels, key))
    got = {j: (row(rows, "must_not", j)["n"], row(rows, "must_not", j)["tpr"], row(rows, "must_not", j)["tnr"])
           for j in ("primary", "second", "veto", "d33")}  # fmt: skip
    assert got == {"primary": (4, 1.0, 1.0), "second": (4, 0.5, 0.5), "veto": (4, 0.5, 1.0), "d33": (4, 1.0, 0.5)}


def test_fabricated_cards_are_scored_against_the_audit_only(tmp_path):
    labels = [("fabricated|a", "yes"), ("fabricated|b", "no"), ("fabricated|c", "no")]
    key = {"fabricated|a": {"pass": True, "decided_by": "audit", "label": "supported"},
           "fabricated|b": {"pass": False, "decided_by": "audit", "label": "fabricated"},
           "fabricated|c": {"pass": True, "decided_by": "audit", "label": "supported"}}  # fmt: skip
    rows = LA.rows(*write(tmp_path, labels, key))
    assert {r["judge"] for r in rows if r["type"] == "fabricated"} == {"audit"}
    r = row(rows, "fabricated", "audit")
    assert (r["n"], r["tpr"], r["tnr"]) == (3, 1.0, 0.5)


def test_main_prints_a_markdown_row_per_type_and_judge(tmp_path, capsys):
    labels, key = ten_stale()
    lp, kp = write(tmp_path, labels, key)
    assert LA.main([str(lp), str(kp)]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("| stale")]
    cells = {tuple(c.strip() for c in ln.strip("|").split("|"))[1]: ln for ln in lines}
    assert set(cells) == {"primary", "second", "veto", "d33"}
    assert "0.83" in cells["primary"] and "0.50" in cells["primary"] and "0.35" in cells["primary"]
