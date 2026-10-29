"""Human context-card labels vs the judges' hidden verdicts (design §7, D18, D33).

    python -m scorer.label_agreement labels.jsonl runs/scorer/<run>/cards/cards_key.jsonl

"yes" on a card means its check passes, so the human label is the reference: TPR is the share of human-yes cards the
judge passes, TNR the share of human-no cards it fails, and Cohen's kappa their chance-corrected agreement. Skipped
and unlabeled cards are left out. Judges per card: `primary` and `second` (their own votes), `veto` (fails if either
judge fails: the verdict Restraint uses, §7), `d33` (fails only when both judges fail: the no-leak gate's evidence,
D33) and, for `fabricated` cards, `audit` (the integrity audit's label). A judge is accepted for a type when
TPR >= 0.9, TNR >= 0.9 and kappa >= 0.8 (§7). Reads files only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

BAR = {"tpr": 0.9, "tnr": 0.9, "kappa": 0.8}
TYPES = ("must_not", "stale", "fabricated")


def _vote(v: dict, who: str) -> bool | None:
    return v[who]["pass"] if v.get(who) else None


def _veto(v: dict) -> bool | None:
    votes = [x for x in (_vote(v, "primary"), _vote(v, "second")) if x is not None]
    return all(votes) if votes else None


def _d33(v: dict) -> bool | None:
    p, s = _vote(v, "primary"), _vote(v, "second")
    return None if p is None or s is None else (p or s)


JUDGES: dict[str, Callable[[dict], bool | None]] = {
    "primary": lambda v: _vote(v, "primary"),
    "second": lambda v: _vote(v, "second"),
    "veto": _veto,
    "d33": _d33,
    "audit": lambda v: v.get("pass") if v.get("decided_by") == "audit" else None,
}


def _jsonl(path: Path | str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def read_labels(path: Path | str) -> dict[str, bool]:
    """card_id -> human pass (yes) / fail (no); the last label per card wins, skips dropped."""
    last = {r["card_id"]: r["label"] for r in _jsonl(path)}
    return {c: lab == "yes" for c, lab in last.items() if lab in ("yes", "no")}


def agreement(pairs: list[tuple[bool, bool]]) -> dict:
    """(human, judge) pairs -> n, TPR, TNR, Cohen's kappa (None where undefined) and the §7 acceptance."""
    n = len(pairs)
    tp = sum(h and j for h, j in pairs)
    tn = sum(not h and not j for h, j in pairs)
    yes = sum(h for h, _ in pairs)
    no = n - yes
    judge_yes = sum(j for _, j in pairs)
    tpr = tp / yes if yes else None
    tnr = tn / no if no else None
    kappa = None
    if n:
        po = (tp + tn) / n
        pe = (yes / n) * (judge_yes / n) + (no / n) * ((n - judge_yes) / n)
        kappa = (po - pe) / (1 - pe) if pe < 1 else None
    stats = {"n": n, "yes": yes, "no": no, "tpr": tpr, "tnr": tnr, "kappa": kappa}
    stats["accepted"] = all(stats[k] is not None and round(stats[k], 9) >= bar for k, bar in BAR.items())
    return stats


def rows(labels_path: Path | str, key_path: Path | str) -> list[dict]:
    human = read_labels(labels_path)
    key = {r["card_id"]: r["verdict"] for r in _jsonl(key_path)}
    out = []
    for typ in TYPES:
        cards = [(human[c], key[c]) for c in human if c in key and c.split("|", 1)[0] == typ]
        for name, judge in JUDGES.items():
            pairs = [(h, j) for h, v in cards if (j := judge(v)) is not None]
            if pairs:
                out.append({"type": typ, "judge": name, **agreement(pairs)})
    return out


def _f(x: float | None) -> str:
    return "—" if x is None else f"{x:.2f}"


def markdown(table: list[dict]) -> str:
    lines = ["| type | judge | n (yes/no) | TPR | TNR | κ | accepted |", "|---|---|---|---|---|---|---|"]
    for r in table:
        lines.append(f"| {r['type']} | {r['judge']} | {r['n']} ({r['yes']}/{r['no']}) | {_f(r['tpr'])} | "
                     f"{_f(r['tnr'])} | {_f(r['kappa'])} | {'yes' if r['accepted'] else 'no'} |")  # fmt: skip
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("labels", help="labels.jsonl exported from labels/index.html")
    ap.add_argument("key", help="cards_key.jsonl next to the exported cards")
    a = ap.parse_args(argv)
    table = rows(a.labels, a.key)
    print(markdown(table))
    print(f"\nBar (§7): TPR ≥ {BAR['tpr']}, TNR ≥ {BAR['tnr']}, κ ≥ {BAR['kappa']}. Skips and unlabeled cards excluded. "
          "veto = fails if either judge fails (Restraint); d33 = fails only if both fail (leak gate).")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
