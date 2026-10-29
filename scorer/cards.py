"""Context cards for human labels (design §7, D18): click-only yes/no, no history to read.

A card shows the request, the response, the one entry (value, dates, a disclosure excerpt, the value it replaced) and
the check; "yes" always means the check passes. Types: `must_not` and `stale` (one per judged answer check) and
`fabricated` (one per audited memory line: the line plus its mapped entry, or the closest ledger entries when it
mapped to none). The judge verdict is hidden: it goes to cards_key.jsonl, never to cards.jsonl / cards.csv.

Stratified per type: up to PER_TYPE cards, PER_SIDE judged pass and PER_SIDE judged fail where available, the rest
filled from the larger side; picks ordered by a hash of the card id (deterministic).
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from scorer.gold import Gold

PER_TYPE, PER_SIDE = 100, 50
QUESTIONS = {
    "must_not": "Is this entry ABSENT from the response (not stated, hinted at, or used as the basis of advice)?",
    "stale": "Does the response AVOID presenting this old value as currently true?",
    "fabricated": "Is this memory line SUPPORTED by the entry below (something the user told the assistant)?",
}
CSV_FIELDS = ["card_id", "type", "question", "t", "request", "response", "entry_id", "entry_value", "entry_dates",
              "previous_value", "disclosure_excerpt", "label_yes_no"]  # fmt: skip


def _entry(gold: Gold, principal: str, item_id: str, t: str) -> dict:
    e = gold.entry(principal, item_id)
    prev = gold.entries(principal).get(e.supersedes or "")
    return {"id": e.id, "about": gold.name(e.subject), "value": e.label() if e.kind != "fact" else e.text,
            "attribute": e.attribute, "valid_from": e.valid_from, "valid_to": e.valid_to,
            "previous_value": prev.text if prev is not None else "",
            "disclosure_excerpt": gold.excerpt(principal, item_id, t)}  # fmt: skip


def answer_cards(gold: Gold, probes, verdicts: list[dict], answers: list[dict]) -> list[dict]:
    by_probe = {p.id: p for p in probes}
    text = {(a["probe_id"], a["system"]): a.get("text") or "" for a in answers}
    out = []
    for v in verdicts:
        if v["type"] not in ("absent", "stale") or v["pass"] is None or v["probe_id"] not in by_probe:
            continue
        p = by_probe[v["probe_id"]]
        typ = "must_not" if v["type"] == "absent" else "stale"
        out.append({
            "card_id": f"{typ}|{v['probe_id']}|{v['system']}|{v['item_id']}", "type": typ, "principal": p.principal,
            "t": p.t, "question": QUESTIONS[typ], "request": p.request, "response": text.get((p.id, v["system"]), ""),
            "entry": _entry(gold, p.principal, v["item_id"], p.t),
            "verdict": {k: v.get(k) for k in ("pass", "decided_by", "code_rule", "primary", "second", "system")},
        })  # fmt: skip
    return out


def _closest(gold: Gold, principal: str, line: str, n: int = 3) -> list[str]:
    words = set(re.findall(r"\w{4,}", line.lower()))
    scored = sorted(gold.entries(principal).values(),
                    key=lambda e: -len(words & set(re.findall(r"\w{4,}", (e.text + " " + e.attribute).lower()))))  # fmt: skip
    return [e.id for e in scored[:n]]


def fabricated_cards(gold: Gold, audit: dict) -> list[dict]:
    """One card per distinct audited line that is a claim, judged at the first commit it appears in."""
    from scorer.audit import label_line

    out = []
    for p, a in audit["principals"].items():
        for m in a["mapping"]:
            t = m["first_seen"]
            lab = label_line(gold, p, m, t)
            if lab not in ("supported", "fabricated"):
                continue
            ids = [e["id"] for e in m["entries"] if e["id"] in gold.entries(p)]
            if ids:
                entry = {**_entry(gold, p, ids[0], t), "candidates": ids}
            else:  # no mapped entry: show the closest ones as context, never as the entry
                near = _closest(gold, p, m["line"])
                entry = {"id": "", "about": "", "value": "(no matching entry) closest: "
                         + " | ".join(gold.entry(p, i).label() for i in near), "valid_from": None, "valid_to": None,
                         "previous_value": "", "disclosure_excerpt": "", "candidates": near}  # fmt: skip
            digest = hashlib.sha256(f"{m['file']}\n{m['line']}".encode("utf-8")).hexdigest()[:12]
            out.append({"card_id": f"fabricated|{p}|{digest}", "type": "fabricated", "principal": p, "t": t,
                        "question": QUESTIONS["fabricated"], "request": "", "response": f"{m['file']}: {m['line']}",
                        "entry": entry, "verdict": {"pass": lab == "supported", "decided_by": "audit", "label": lab}})  # fmt: skip
    return out


def _rank(card_id: str) -> str:
    return hashlib.sha256(card_id.encode("utf-8")).hexdigest()


def stratify(cards: list[dict], *, per_type: int = PER_TYPE, per_side: int = PER_SIDE) -> list[dict]:
    out = []
    for typ in sorted({c["type"] for c in cards}):
        mine = sorted((c for c in cards if c["type"] == typ and c["verdict"]["pass"] is not None),
                      key=lambda c: _rank(c["card_id"]))  # fmt: skip
        sides = {True: [c for c in mine if c["verdict"]["pass"]], False: [c for c in mine if not c["verdict"]["pass"]]}
        take = {k: v[:per_side] for k, v in sides.items()}
        room = per_type - sum(len(v) for v in take.values())
        for k in sorted(sides, key=lambda k: -len(sides[k])):
            extra = sides[k][len(take[k]) : len(take[k]) + max(0, room)]
            take[k] += extra
            room -= len(extra)
        out += sorted(take[True] + take[False], key=lambda c: _rank(c["card_id"]))
    return out


def export(out_dir: Path | str, cards: list[dict]) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cards.jsonl", "w", encoding="utf-8") as shown, open(out / "cards_key.jsonl", "w", encoding="utf-8") as key:
        for c in cards:
            shown.write(json.dumps({k: v for k, v in c.items() if k != "verdict"}, ensure_ascii=False) + "\n")
            key.write(json.dumps({"card_id": c["card_id"], "verdict": c["verdict"]}, ensure_ascii=False) + "\n")
    with open(out / "cards.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for c in cards:
            e = c.get("entry") or {}
            dates = " ".join(x for x in (f"from {e.get('valid_from')}" if e.get("valid_from") else "",
                                         f"until {e.get('valid_to')}" if e.get("valid_to") else "") if x)  # fmt: skip
            w.writerow({"card_id": c["card_id"], "type": c["type"], "question": c["question"], "t": c.get("t"),
                        "request": c.get("request", ""), "response": c.get("response", ""), "entry_id": e.get("id", ""),
                        "entry_value": e.get("value", ""), "entry_dates": dates,
                        "previous_value": e.get("previous_value", ""),
                        "disclosure_excerpt": e.get("disclosure_excerpt", ""), "label_yes_no": ""})  # fmt: skip
    counts: dict[str, dict[str, int]] = {}
    for c in cards:
        side = "pass" if c["verdict"]["pass"] else "fail"
        counts.setdefault(c["type"], {"pass": 0, "fail": 0})[side] += 1
    return counts
