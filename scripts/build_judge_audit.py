"""Blinded human audit of the Argon-judge vs Haiku-judge disagreements on Argon's answers.

Both judges labeled the SAME answers (runs-argon/crossjudge/*__judge-haiku_eval.json vs the original Argon-judged
*_eval.json); they disagree on ~7% of items, almost always Argon=CORRECT / Haiku=WRONG. A human label on a sample
says which judge follows the rubric. Sample (seed 0), shuffled together so the labeler can't tell strata apart:
  - 25 Argon-only-CORRECT items from each of LoCoMo FC, LoCoMo RAG, LongMemEval FC, LongMemEval RAG (100)
  - every Haiku-only-CORRECT item (5)
  - 4 agreement controls per set (2 both-CORRECT, 2 both-WRONG; 16)
Writes docs/judge-audit/audit_sheet.csv (no judge labels) and runs-argon/crossjudge/audit_key.csv (the labels).

Usage: uv run python scripts/build_judge_audit.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
L, M = REPO / "vendor/Memora/app/locomo/results", REPO / "vendor/Memora/app/longmemeval/results"
X = REPO / "runs-argon/crossjudge"
SETS = {  # set -> (benchmark, Argon-judged eval, Haiku-judged eval)
    "locomo-fc": ("LoCoMo", L / "full_context_20261025_232937/full_context_eval.json", X / "argon-locomo-fc__judge-haiku_eval.json"),
    "locomo-rag": ("LoCoMo", L / "rag_20261025_232937/rag_eval.json", X / "argon-locomo-rag__judge-haiku_eval.json"),
    "lme-fc": ("LongMemEval", M / "full_context_20261025_232936/full_context_eval.json", X / "argon-lme-fc__judge-haiku_eval.json"),
    "lme-rag": ("LongMemEval", M / "rag_20261025_232936/rag_eval.json", X / "argon-lme-rag__judge-haiku_eval.json"),
}
PER_SET, CONTROLS_PER_LABEL = 25, 2


def records(path: Path) -> dict[str, dict]:
    """{item key: record with question / gold / response / type / llm_score} for either benchmark's eval format."""
    d = json.loads(path.read_text())
    out = {}
    if isinstance(d, dict):  # LoCoMo: {conv: [records]}; key = conv + question + occurrence
        for conv, recs in d.items():
            seen = Counter()
            for r in recs:
                seen[r["question"]] += 1
                out[f"{conv}|{r['question']}#{seen[r['question']]}"] = {
                    "question": r["question"], "gold": r["answer"], "response": r["response"],
                    "type": f"category {r['category']}", "score": float(r["llm_score"]),
                }  # fmt: skip
    else:  # LongMemEval: [records]; key = question_id
        for r in d:
            qtype = "abstention" if "_abs" in r["question_id"] else r["question_type"]
            out[r["question_id"]] = {
                "question": r["question"], "gold": r["answer"], "response": r["generated_answer"],
                "type": qtype, "score": float(r["llm_score"]),
            }  # fmt: skip
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for name, (bench, argon_path, haiku_path) in SETS.items():
        a, h = records(argon_path), records(haiku_path)
        keys = sorted(a.keys() & h.keys())
        strata = {
            "argon-only-correct": [k for k in keys if a[k]["score"] > h[k]["score"]],
            "haiku-only-correct": [k for k in keys if h[k]["score"] > a[k]["score"]],
            "both-correct": [k for k in keys if a[k]["score"] == h[k]["score"] == 1.0],
            "both-wrong": [k for k in keys if a[k]["score"] == h[k]["score"] == 0.0],
        }
        take = {"argon-only-correct": PER_SET, "haiku-only-correct": None,
                "both-correct": CONTROLS_PER_LABEL, "both-wrong": CONTROLS_PER_LABEL}  # fmt: skip
        for stratum, n in take.items():
            pool = strata[stratum]
            picked = pool if n is None or n >= len(pool) else [pool[i] for i in sorted(rng.choice(len(pool), n, replace=False))]
            for k in picked:
                rows.append({"set": name, "benchmark": bench, "key": k, "stratum": stratum, **a[k],
                             "argon_judge": "CORRECT" if a[k]["score"] else "WRONG",
                             "haiku_judge": "CORRECT" if h[k]["score"] else "WRONG"})  # fmt: skip
    rows = [rows[i] for i in rng.permutation(len(rows))]

    sheet = REPO / "docs/judge-audit/audit_sheet.csv"
    sheet.parent.mkdir(parents=True, exist_ok=True)
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "benchmark", "question_type", "question", "gold_answer_or_rubric", "model_response",
                    "your_label (CORRECT/WRONG)", "notes"])  # fmt: skip
        for i, r in enumerate(rows, 1):
            w.writerow([f"A{i:03d}", r["benchmark"], r["type"], r["question"], r["gold"], r["response"], "", ""])
    with open(X / "audit_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "set", "stratum", "argon_judge", "haiku_judge", "key"])
        for i, r in enumerate(rows, 1):
            w.writerow([f"A{i:03d}", r["set"], r["stratum"], r["argon_judge"], r["haiku_judge"], r["key"]])
    print(f"{len(rows)} items -> {sheet}; key -> {X / 'audit_key.csv'}")
    print(Counter((r["set"], r["stratum"]) for r in rows))


if __name__ == "__main__":
    main()
