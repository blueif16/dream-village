"""Final results tables (Markdown): LoCoMo + LongMemEval subset-24, accuracy with 95% bootstrap CIs and paired
differences computed only on questions both systems answered. No API calls; reads eval JSONs only.

Manifest schema: {"locomo": {system: eval_path}, "lme": {system: eval_path}, "notes": {system: "provisional ..."}}
(paths relative to repo root; systems absent from a section are omitted).

Usage: uv run python scripts/final_table.py runs/final_manifest.json [--out docs/results.md]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from memobench.stats import cluster_bootstrap, paired_cluster_bootstrap

REPO = Path(__file__).resolve().parents[1]
PAPER = {"memora-p": 0.863, "memora-s": 0.849, "full-context": 0.825, "memora-noabs": 0.653, "rag": 0.633}
REFS = ("memora-p", "full-context")
CATS = ("1", "2", "3", "4")  # LoCoMo category 5 (adversarial) excluded
ORDER = [c["sample_id"] for c in json.loads((REPO / "data/locomo10.json").read_text())]
HELD_OUT = set(ORDER[5:])  # conversations 6-10 in locomo10.json order
LME_SUBSET = set(json.loads((REPO / "docs/lme_subset24.json").read_text())["question_ids"])


def _path(p):
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def load_locomo(path):
    """{conv: [{question, category, llm_score}]} -> {sample_id: {(question, category, occurrence): score}}.
    Memora keys conversations "0".."9", baselines by sample_id; repeated (question, category) pairs within a
    conversation are disambiguated by occurrence index. Category 5 is dropped."""
    ev = json.loads(_path(path).read_text())
    if not (isinstance(ev, dict) and all(isinstance(v, list) for v in ev.values())):
        raise ValueError(f"{path}: expected {{conv_key: [items]}}; add an adapter for this shape")
    out = {}
    for conv, items in ev.items():
        sid = ORDER[int(conv)] if conv.isdigit() else conv
        seen, keyed = defaultdict(int), {}
        for it in items:
            cat = str(it["category"])
            if cat not in CATS:
                continue
            k = (it["question"], cat)
            keyed[k + (seen[k],)] = float(it["llm_score"])
            seen[k] += 1
        out[sid] = keyed
    return out


def load_lme(path):
    """[{question_id, question_type, llm_score}] -> {question_id: (question_type, score)}."""
    ev = json.loads(_path(path).read_text())
    if not (isinstance(ev, list) and all("question_id" in it for it in ev)):
        raise ValueError(f"{path}: expected [{{question_id, question_type, llm_score}}]; add an adapter for this shape")
    return {it["question_id"]: (it["question_type"], float(it["llm_score"])) for it in ev}


def ci(est, lo, hi, signed=False):
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(est)} [{f.format(lo)}, {f.format(hi)}]"


def flag(notes, name):
    return notes.get(name, "")


def locomo_section(systems, notes):
    lines = ["## LoCoMo (categories 1-4; 95% CI = cluster bootstrap over conversations)", "",
             "| system | n | all-10 acc [95% CI] | held-out 6-10 acc [95% CI] | "
             + " | ".join(f"c{c} acc (n)" for c in CATS) + " | paper | note |",
             "|---|---:|---|---|" + "---:|" * len(CATS) + "---:|---|"]
    for name, ev in systems.items():
        allc = {c: list(q.values()) for c, q in ev.items() if q}
        held = {c: v for c, v in allc.items() if c in HELD_OUT}
        n = sum(len(v) for v in allc.values())
        cats = defaultdict(list)
        for q in ev.values():
            for (_, cat, _), s in q.items():
                cats[cat].append(s)
        percat = " | ".join(f"{sum(cats[c]) / len(cats[c]):.3f} ({len(cats[c])})" if cats[c] else "-" for c in CATS)
        paper = f"{PAPER[name]:.3f}" if name in PAPER else "-"
        lines.append(f"| {name} | {n} | {ci(*cluster_bootstrap(allc))} | {ci(*cluster_bootstrap(held))} | "
                     f"{percat} | {paper} | {flag(notes, name)} |")
    lines += ["", "### LoCoMo paired differences (system - reference) on shared questions", "",
              "| reference | system | diff [95% paired cluster CI] | held-out 6-10 diff | n shared |",
              "|---|---|---|---|---:|"]
    for ref in REFS:
        if ref not in systems:
            continue
        for name, ev in systems.items():
            if name == ref:
                continue
            r = systems[ref]
            common = {c: sorted(set(ev[c]) & set(r[c])) for c in ev if c in r}
            common = {c: ks for c, ks in common.items() if ks}
            a = {c: [r[c][k] for k in ks] for c, ks in common.items()}
            b = {c: [ev[c][k] for k in ks] for c, ks in common.items()}
            ha = {c: v for c, v in a.items() if c in HELD_OUT}
            hb = {c: v for c, v in b.items() if c in HELD_OUT}
            n = sum(len(v) for v in a.values())
            lines.append(f"| {ref} | {name} | {ci(*paired_cluster_bootstrap(a, b), signed=True)} | "
                         f"{ci(*paired_cluster_bootstrap(ha, hb), signed=True) if ha else '-'} | {n} |")
    return lines


def lme_section(systems, notes):
    lines = ["## LongMemEval subset-24 (95% CI = bootstrap with each question its own cluster)", "",
             "| system | n | accuracy [95% CI] | per question_type (correct/n) | note |", "|---|---:|---|---|---|"]
    for name, ev in systems.items():
        sub = {q: v for q, v in ev.items() if q in LME_SUBSET}
        types = defaultdict(list)
        for t, s in sub.values():
            types[t].append(s)
        compact = ", ".join(f"{t} {sum(v):g}/{len(v)}" for t, v in sorted(types.items()))
        est = ci(*cluster_bootstrap({q: [s] for q, (_, s) in sub.items()})) if sub else "-"
        lines.append(f"| {name} | {len(sub)} | {est} | {compact} | {flag(notes, name)} |")
    fc = systems.get("full-context")
    if fc is not None and len(fc) == 500:
        lines.append(f"| full-context (all 500) | 500 | {ci(*cluster_bootstrap({q: [s] for q, (_, s) in fc.items()}))} "
                     f"| not restricted to subset | {flag(notes, 'full-context')} |")
    lines += ["", "### LongMemEval paired differences (system - reference) on shared subset questions", "",
              "| reference | system | diff [95% paired CI] | n shared |", "|---|---|---|---:|"]
    for ref in REFS:
        if ref not in systems:
            continue
        for name, ev in systems.items():
            if name == ref:
                continue
            shared = sorted(LME_SUBSET & set(ev) & set(systems[ref]))
            if not shared:
                lines.append(f"| {ref} | {name} | - | 0 |")
                continue
            a = {q: [systems[ref][q][1]] for q in shared}
            b = {q: [ev[q][1]] for q in shared}
            lines.append(f"| {ref} | {name} | {ci(*paired_cluster_bootstrap(a, b), signed=True)} | {len(shared)} |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out", default="docs/results.md")
    args = ap.parse_args()
    man = json.loads(Path(args.manifest).read_text())
    notes = man.get("notes", {})
    lines = ["# Final results", ""]
    lines += locomo_section({k: load_locomo(v) for k, v in man.get("locomo", {}).items()}, notes) + [""]
    lines += lme_section({k: load_lme(v) for k, v in man.get("lme", {}).items()}, notes)
    md = "\n".join(lines) + "\n"
    print(md)
    _path(args.out).write_text(md)


if __name__ == "__main__":
    main()
