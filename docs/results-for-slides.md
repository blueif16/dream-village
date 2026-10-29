# Results brief for the slide builder (v4, 2026-10-26)

**Status:** LoCoMo is FINAL for all methods, including Memora-J (ours) at an 8k recall budget. LongMemEval full context and RAG are FINAL on all 500. Every Memora row on LongMemEval is **BLOCKED** (it needs new embeddings, and no embedding key is available; disk too). Never estimate or invent a blocked or pending number; show "blocked" / "pending".

**New in v4:** a partial second-backbone run on **gemini-4-argon** (Tables 5–6, findings 7–9). **Rule: never put an Argon-judged number next to a Haiku-judged number.** Argon's own judge is lenient by 7–9 points, so the only comparable Argon numbers are the Haiku-judged ones in Table 6.

## Setup (one footnote line)
All systems use the same backbone, answerer and judge, **claude-haiku-5-5 (effort low)**, plus gemini-embedding-2. The paper used GPT-4.1-mini with a gpt-4o-mini judge, so paper numbers are reference points, not like-for-like.
- **LoCoMo:** 10 conversations, 1,540 questions (categories 1–4), one run.
- **LongMemEval_S:** full context on all 500 questions; Memora variants on a fixed stratified 24-question subset.
- **CIs:** 95% bootstrap, clustered by conversation (LoCoMo) or question (LongMemEval).

## Table 1 — LoCoMo accuracy (LLM judge)
| System | Accuracy [95% CI] | Held-out conv. 6–10 | Paper |
|---|---|---|---|
| Full context | **0.846** [0.820, 0.871] | 0.841 | 0.825 |
| Memora, no-abstraction ablation | 0.836 [0.812, 0.861] | 0.838 | 0.653 |
| Memora-P (policy retrieval) | 0.826 [0.808, 0.844] | 0.835 | 0.863 |
| Memora-S (semantic retrieval) | 0.816 [0.792, 0.838] | 0.819 | 0.849 |
| RAG (chunk 500, k=3) | 0.657 [0.638, 0.677] | 0.654 | 0.633 |
| **Memora-J (ours), 8k recall budget** | 0.777 [0.750, 0.804] | 0.788 | — |
| Memora-J (ours), 16k recall budget | 0.782 [0.755, 0.809] | 0.793 | — |
| *Memora-J, 2k budget (diagnostic only)* | *0.137* | *0.127* | — |

Per category (multi-hop / temporal / open-domain / single-hop):

| System | Multi-hop | Temporal | Open-domain | Single-hop |
|---|---|---|---|---|
| Full context | .780 | .844 | .615 | .895 |
| No-abstraction | .755 | .798 | .615 | .904 |
| Memora-P | .745 | .773 | .573 | .903 |
| Memora-S | .727 | .760 | .583 | .893 |
| RAG | .479 | .598 | .510 | .756 |
| Memora-J (8k) | .691 | .779 | .427 | .845 |
| Memora-J (16k) | .699 | .779 | .438 | .850 |

## Table 2 — LoCoMo paired differences (same questions)
| Comparison | Δ accuracy [95% CI] | Significant? |
|---|---|---|
| Full context − Memora-P | +2.0 pts [+0.2, +3.8] | yes (all 10); tie on held-out (+0.6 [−2.0, +3.4]) |
| No-abstraction − Memora-P | +1.0 pts [−0.7, +2.9] | no |
| Memora-P − Memora-S | +1.1 pts [−0.5, +2.6] | no |
| RAG − Memora-P | −16.9 pts [−19.1, −14.6] | yes |
| **Memora-J (8k) − Memora-P** | **−4.9 pts [−6.7, −2.9]** | yes (held-out −4.8 [−7.5, −1.8]) |
| **Memora-J (8k) − Full context** | **−6.9 pts [−8.9, −4.2]** | yes (held-out −5.3 [−8.3, −0.9]) |
| Memora-J (16k) − Memora-P | −4.5 pts [−6.4, −2.3] | yes (held-out −4.3 [−7.5, −1.0]) |
| Memora-J (16k) − Full context | −6.4 pts [−8.5, −3.8] | yes (held-out −4.8 [−8.2, −0.6]) |

## Table 3 — LoCoMo cost and efficiency (Haiku 5.5 prices, judge excluded)
| System | Build $/conv | Query $/conv | Prompt tokens per question | LLM calls per question |
|---|---|---|---|---|
| Full context | 0 | 0.118 (86% of input from prompt cache) | 38.5k | 1.0 |
| RAG | 0 | 0.022 | 2.9k | 1.0 |
| Memora-S | 0.10 | 0.244 | 17.0k | 0.9 |
| Memora-P | 0.10 | 0.579 | 20.5k | 4.6 |
| No-abstraction | 0.08 | 0.204 | 13.3k | 1.0 |
| **Memora-J (8k)** | 0.097 + Jev decisions* | **0.084** (real-time; Batch would halve it) | **5.8k** | 1 Haiku + ~4 Jev decisions; no LLM call in retrieval |

## Table 4 — LongMemEval_S
| System | All 500 | Subset-24 |
|---|---|---|
| Full context (~176k-token prompts) | **0.878** (paper: 0.656) | 0.917 (22/24) |
| RAG (chunk 500, k=3) | 0.508 [0.464, 0.552]* | 0.583 (14/24) |
| Memora-S | — | BLOCKED |
| Memora-P | — | BLOCKED |
| Memora-J | — | BLOCKED (needs new embeddings, ~4.4 GB disk) |

\* Ran 2026-10-26 from cached embeddings only (0 embedding calls); the Oct 21 attempt died on a Gemini quota error.

## Second backbone: gemini-4-argon (PARTIAL; stopped 2026-10-26 when the donated Gemini keys were revoked)
Same prompts, data, threshold and protocol as above; only the backbone changes, to gemini-4-argon (thinking level low) for the answerer, the judge and Memora's memory building. Calls ran real-time (no Batch) with Gemini's automatic prompt caching. Run cost: $244 (all Haiku work so far: ~$50, excluding Jev decisions).
- **Complete:** LoCoMo full context + RAG; LongMemEval full context + RAG (all 500); LongMemEval Memora-S / P / no-abstraction on subset-24.
- **Missing (show "not run"):** LoCoMo Memora-S, Memora-P and no-abstraction. Their builds died at 96–98% when the key was revoked.
- **Cross-judging:** Argon's answers were re-judged by the Haiku judge (Batch). The reverse direction never ran.

### Table 5 — Same Argon answers, two judges
| Argon's answers | Argon judge | Haiku judge | Judges agree | Disagreements (Argon says CORRECT / Haiku says CORRECT) |
|---|---|---|---|---|
| LoCoMo full context | 0.913 | 0.844 | 93.0% | 107 / 1 |
| LoCoMo RAG | 0.748 | 0.681 | 92.7% | 108 / 4 |
| LongMemEval full context | 0.986 | 0.908 | 92.2% | 39 / 0 |
| LongMemEval RAG | 0.622 | 0.532 | 91.0% | 45 / 0 |

Which judge follows the rubric is **not yet known**. A blinded human audit of 121 items (`docs/judge-audit/`) is pending. Say "the judges disagree by 7–9 points", not "Argon's judge is wrong".

### Table 6 — Argon vs Haiku under the SAME (Haiku) judge, paired on the same questions
| Benchmark · system | Haiku | Argon | Argon − Haiku [95% CI] | Significant? |
|---|---|---|---|---|
| LoCoMo · full context | 0.846 | 0.844 | −0.2 pts [−1.4, +0.9] | no |
| LoCoMo · RAG | 0.657 | 0.681 | +2.3 pts [+1.4, +3.2] | yes |
| LongMemEval · full context (500) | 0.878 | 0.908 | +3.0 pts [+0.0, +6.2] | borderline |
| LongMemEval · RAG (500) | 0.508 | 0.532 | +2.4 pts [−0.6, +5.4] | no |

Full context's lead over RAG (Haiku judge): LoCoMo +18.9 (Haiku) vs +16.4 (Argon), a change of −2.5 [−4.2, −1.0]; LongMemEval +37.0 vs +37.6, a change of +0.6 [−3.6, +4.8].

LongMemEval subset-24 under the Haiku judge (n=24, directional only; one question = 4.2 pts): full context 0.917 (Haiku 0.917) · Memora-S 0.875 · Memora-P 0.917 · no-abstraction 0.833.

## Headline findings (safe to put on slides)
1. **A 2026 backbone closes the gap.** With Haiku 5.5, full context scores **0.846** on LoCoMo and **0.878** on LongMemEval. That's at or above the paper's Memora (0.863 / 0.874); the paper had full context at only 0.825 / 0.656.
2. **On LoCoMo, Memora doesn't beat full context.** Memora-P trails full context by 2.0 points overall and ties on the held-out conversations.
3. **Abstraction adds ~nothing here.** Turning abstraction off scores 0.836, tied with Memora-P. The paper's 0.653 for this ablation equals its Mem0 row exactly, so it probably wasn't run separately. Say "matches the Mem0 row", not "copied".
4. **Policy retrieval (P) costs ~2.4× semantic retrieval (S)** at the query stage, for a statistically insignificant +1.1 points.
5. **Memora-J (ours) trades accuracy for context and query cost.** At an 8k recall budget it scores **0.777**, 4.9 points below Memora-P and 6.9 below full context (both significant). It uses **3.5× fewer prompt tokens per question than Memora-P** (5.8k vs 20.5k; 6.6× fewer than full context) and **~7× lower query cost** ($0.084 vs $0.579 per conversation), with about the same build cost.
6. **Above ~8k, Memora-J's budget isn't the limit; its untuned recall thresholds are.** The accuracy-vs-budget curve is 2k → **0.137** (the user profile consumed the budget), 8k → **0.777** (5.8k tokens/q), 16k → **0.782** (6.3k tokens/q). Memora-P is at 0.826 with 20.5k tokens/q. Doubling the budget barely adds context or accuracy, because recall stops before filling it. Tuning the recall thresholds is the next lever. The gap to Memora-P is largest on open-domain (0.427 vs 0.573) and multi-hop (0.691 vs 0.745) questions, which need many retrieved facts. Its thresholds are untuned and the budget wasn't tuned on results, so there's headroom.
7. **The judge matters as much as the model.** On the same Argon answers, Argon's judge scores 7–9 points higher than Haiku's (Table 5), and the disagreements are almost all one-way. Backbone comparisons are only valid with the judge held fixed.
8. **A frontier backbone adds little once the judge is fixed.** Under the Haiku judge, gemini-4-argon equals Haiku 5.5 on LoCoMo full context (−0.2 pts, n.s.), gains +2.3 on LoCoMo RAG, and +3.0 (borderline) / +2.4 (n.s.) on LongMemEval full context / RAG (Table 6). The apparent +6.7 on LoCoMo full context was the judge.
9. **RAG's deficit is about retrieval, not the answerer.** Full context beats RAG by ~17–19 points on LoCoMo and ~37 on LongMemEval with either backbone; the stronger model narrows the LoCoMo gap by only 2.5 points and the LongMemEval gap not at all.

## Slide suggestion: accuracy vs context per question (LoCoMo)
Points (tokens/q → accuracy): RAG 2.9k → 0.657 · Memora-J 2k-budget 2.9k → 0.137 (diagnostic) · Memora-J 8k 5.8k → 0.777 · Memora-J 16k 6.3k → 0.782 · No-abstraction 13.3k → 0.836 · Memora-S 17.0k → 0.816 · Memora-P 20.5k → 0.826 · Full context 38.5k → 0.846.

## Reproducibility check
The design session's new harness replays our LoCoMo conv-26 runs byte-identically at $0. No-abstraction, full context and RAG match 152/152 on answers and judge labels; Memora-S matches 151/152 (one question differs because Chroma rebuilds its index per process).

## Caveats (footnotes)
- **Run setup:** one run per system. Haiku 5.5 doesn't accept temperature or seed, so variance is unmeasured.
- **Thresholds:** Memora's consolidation threshold was recalibrated for Gemini embeddings (t = 0.9291 instead of 0.8). Other paper deviations are in `docs/deviations.md`.
- **Memora-P:** 1,537 of 1,540 questions answered (3 lost to a network hang). Its prompts embed wall-clock timings, so it can't be replayed exactly.
- **Ground truth:** an independent audit reports ~6.4% of LoCoMo gold answers are wrong, so ~93.6% is the practical ceiling.
- **Memora-J:** its memory decisions are made by the Jev decision model (jev-1.13.0), an intended upgrade over Haiku-made decisions; its answerer and judge are the frozen Haiku setup. The recall budget (8k) was set by a design rule (at most Memora-P's mean context), not tuned on results. Thresholds are untuned defaults (0.5).
- **Argon run:** one partial run; its LoCoMo Memora rows don't exist, so there is no Argon Memora-vs-full-context comparison on LoCoMo. Do not call LongMemEval_S "saturated": that came from the lenient Argon judge (0.986); under the Haiku judge Argon full context is 0.908.
- **Jev decisions are a paid hosted API** (TypeSafe), billed on input tokens at an unknown rate; Memora-J costs above exclude them.
- \* Memora-J build cost excludes ~3,020 Jev decision requests per conversation (price not yet known; only input tokens are billed).
