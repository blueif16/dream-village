"""Tables 5-6 of docs/results-for-slides.md: judge agreement on Argon answers, and Argon vs Haiku under the Haiku
judge (paired, cluster bootstrap). Reads the Argon/Haiku *_eval.json files and runs-argon/crossjudge/*__judge-haiku_eval.json.

Usage: uv run python scripts/argon_judge_analysis.py
"""
import json
from collections import defaultdict, Counter
import numpy as np
from memobench.stats import cluster_bootstrap, paired_cluster_bootstrap
L, M, X = "vendor/Memora/app/locomo/results", "vendor/Memora/app/longmemeval/results", "runs-argon/crossjudge"

def items(path):
    """{(cluster, item_id): score}; LoCoMo items keyed by conv + question text + occurrence, LME by question_id."""
    d = json.load(open(path)); out = {}
    if isinstance(d, dict):
        for conv, recs in d.items():
            seen = Counter()
            for r in recs:
                q = r["question"]; seen[q] += 1
                out[(conv, f"{q}#{seen[q]}")] = float(r["llm_score"])
    else:
        for r in d: out[(r["question_id"], r["question_id"])] = float(r["llm_score"])
    return out

def clusters(it):
    c = defaultdict(list)
    for (k, i), s in sorted(it.items()): c[k].append(s)
    return c

def kappa(a, b):
    a, b = np.array(a), np.array(b); po = (a == b).mean(); pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb); return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")

print("== Judge agreement on Argon's answers (Argon judge vs Haiku judge)")
sets = {"locomo-fc": f"{L}/full_context_20261025_232937/full_context_eval.json", "locomo-rag": f"{L}/rag_20261025_232937/rag_eval.json",
        "lme-fc": f"{M}/full_context_20261025_232936/full_context_eval.json", "lme-rag": f"{M}/rag_20261025_232936/rag_eval.json",
        "lme-sub24-memora-s": f"{M}/memora_semantic_argon-lme-sub24_20261025_235612/memora_semantic_eval.json",
        "lme-sub24-memora-p": f"{M}/memora_prompt_argon-lme-sub24-P_20261025_235605/memora_prompt_eval.json",
        "lme-sub24-memora-noabs": f"{M}/memora_semantic_argon-lme-sub24-noabs_20261025_235553/memora_semantic_eval.json"}
hj = {}
for name, p in sets.items():
    a = items(p); b = items(f"{X}/argon-{name}__judge-haiku_eval.json"); hj[name] = b
    keys = sorted(a.keys() & b.keys()); va = [a[k] for k in keys]; vb = [b[k] for k in keys]
    po, k = kappa(va, vb)
    print(f"  {name:24} n={len(keys):5}  Argon-judge {np.mean(va):.3f}  Haiku-judge {np.mean(vb):.3f}  agree {po:.3f}  kappa {k:.2f}"
          f"  (Argon-only CORRECT {sum(x > y for x, y in zip(va, vb))}, Haiku-only {sum(y > x for x, y in zip(va, vb))})")

print("== Same judge (Haiku) on both backbones' answers: Argon - Haiku, paired, cluster bootstrap")
haiku = {"locomo-fc": f"{L}/full_context_20261021_125141/full_context_eval.json", "locomo-rag": f"{L}/rag_20261021_125143/rag_eval.json",
         "lme-fc": f"{M}/full_context_20261021_135734/full_context_eval.json"}
pairs = {}
for name, p in haiku.items():
    h = items(p); g = hj[name]; keys = sorted(h.keys() & g.keys()); pairs[name] = (h, g, keys)
    ch = clusters({k: h[k] for k in keys}); cg = clusters({k: g[k] for k in keys})
    e, lo, hi = paired_cluster_bootstrap(ch, cg)
    print(f"  {name:12} n={len(keys):5}  Haiku {np.mean([h[k] for k in keys]):.3f}  Argon {np.mean([g[k] for k in keys]):.3f}  diff {100*e:+.1f} pts [{100*lo:+.1f}, {100*hi:+.1f}]")

print("== Does the full-context advantage over RAG shrink with the stronger model? (LoCoMo, Haiku judge)")
hf, gf, kf = pairs["locomo-fc"]; hr, gr, kr = pairs["locomo-rag"]; keys = sorted(set(kf) & set(kr))
for lbl, f, r in (("Haiku", hf, hr), ("Argon", gf, gr)):
    e, lo, hi = paired_cluster_bootstrap(clusters({k: r[k] for k in keys}), clusters({k: f[k] for k in keys}))
    print(f"  {lbl}: FC - RAG {100*e:+.1f} pts [{100*lo:+.1f}, {100*hi:+.1f}]")
dd = {k: (gf[k] - gr[k]) - (hf[k] - hr[k]) for k in keys}
e, lo, hi = cluster_bootstrap(clusters(dd))
print(f"  change in gap (Argon - Haiku): {100*e:+.1f} pts [{100*lo:+.1f}, {100*hi:+.1f}]")

print("== LongMemEval RAG, all 500, Haiku judge; and the FC - RAG gap on LongMemEval")
hr = items(f"{M}/rag_20261026_010557/rag_eval.json"); gr = hj["lme-rag"]; keys = sorted(hr.keys() & gr.keys())
e, lo, hi = paired_cluster_bootstrap(clusters({k: hr[k] for k in keys}), clusters({k: gr[k] for k in keys}))
print(f"  lme-rag      n={len(keys):5}  Haiku {np.mean([hr[k] for k in keys]):.3f}  Argon {np.mean([gr[k] for k in keys]):.3f}  diff {100*e:+.1f} pts [{100*lo:+.1f}, {100*hi:+.1f}]")
hf, gf, kf = pairs["lme-fc"]; keys = sorted(set(kf) & set(keys))
for lbl, f, r in (("Haiku", hf, hr), ("Argon", gf, gr)):
    e, lo, hi = paired_cluster_bootstrap(clusters({k: r[k] for k in keys}), clusters({k: f[k] for k in keys}))
    print(f"  {lbl}: FC - RAG {100*e:+.1f} pts [{100*lo:+.1f}, {100*hi:+.1f}]")
e, lo, hi = cluster_bootstrap(clusters({k: (gf[k] - gr[k]) - (hf[k] - hr[k]) for k in keys}))
print(f"  change in gap (Argon - Haiku): {100*e:+.1f} pts [{100*lo:+.1f}, {100*hi:+.1f}]")
sub = set(json.load(open("docs/lme_subset24.json"))["question_ids"])
print("== LongMemEval subset-24 (n=24), both judges")
for name, p in (("Haiku FC", f"{M}/full_context_20261021_135734/full_context_eval.json"),):
    s = items(p); print(f"  {name:22} Haiku-judge {np.mean([v for (k, _), v in s.items() if k in sub]):.3f}")
a = items(sets["lme-fc"]); b = hj["lme-fc"]
print(f"  {'Argon FC':22} Argon-judge {np.mean([v for (k, _), v in a.items() if k in sub]):.3f}  Haiku-judge {np.mean([v for (k, _), v in b.items() if k in sub]):.3f}")
for n in ("lme-sub24-memora-s", "lme-sub24-memora-p", "lme-sub24-memora-noabs"):
    a = items(sets[n]); b = hj[n]; print(f"  Argon {n[10:]:16} Argon-judge {np.mean(list(a.values())):.3f}  Haiku-judge {np.mean(list(b.values())):.3f}")
