# Dream Village evidence

These are the findings behind `decisions.md`, distilled from research agents' reports and our own measurements (2026-10-27/28).
- **✓** = checked against the primary source by us.
- Unmarked rows come from research agents with the source given and have not been checked by us. Treat their numbers as claims until checked.

## Our own measurements (E-OWN)
| Key | Finding | Source |
|---|---|---|
| E-OWN-1 | On the same answers, the gemini-4-argon judge scored 7–9 points higher than the Haiku judge, almost all in one direction (299 vs 5 disagreements). | `docs/results-for-slides.md` Table 5 |
| E-OWN-2 | Existing benchmarks don't look like daily assistant use. **LoCoMo:** two friends, ~29 sessions over ~8 months, ~24% of turns are evidence. **LongMemEval_S:** median span 10 days, ~2 of 48 sessions (4%) are evidence, and the rest is unrelated filler. Every session in both is timestamped. | Measured on `data/` (2026-10-27) |
| E-OWN-3 | Memora-J, an earlier system unrelated to the harness under test (D20), scored 0.777 on LoCoMo, against Memora-P 0.826 and full context 0.846. The harness replays earlier runs byte-identically. Chroma's index rebuild is non-deterministic (151/152). | `docs/results-for-slides.md` |

## How the leading products run memory (E-PROD)
| Key | Finding | Source |
|---|---|---|
| E-PROD-1 ✓ | ChatGPT Dreaming V3: a background process synthesizes memory across many chats. Its stated goals are "carry forward context", "follow preferences and constraints" and "stay current over time" (the birthday-party example), over "multi-year time horizons". | https://openai.com/index/chatgpt-memory-dreaming/ (2026-06-04) |
| E-PROD-2 ✓ | Claude apps save memory "as a set of individual topics as you chat… your next conversation already knows". | https://support.claude.com/en/articles/11817273 (accessed 2026-10-27) |
| E-PROD-3 ✓ | Claude Code auto memory is loaded every session (first 200 lines or 25 KB). | https://code.claude.com/docs/en/memory |
| E-PROD-4 | Claude Code auto-dream fires at ≥24 h and ≥5 sessions; it merges, converts relative dates to absolute, and deletes contradicted facts. Behind a feature flag; from analysis of the app's code. | https://changelogs.core-directive.com/v/2.1.94/e/memory-dream-mode-background-pruning-consolidation |
| E-PROD-5 | Codex consolidates at session start over threads idle ≥6 h, drops memories unused for >30 days, and injects a ≤2,500-token summary. | https://github.com/openai/codex/blob/main/codex-rs/memories/README.md |
| E-PROD-6 | Mem0 v3 is add-only (update and delete removed); Zep never deletes, it marks old facts invalid. Both are built for volume, not curation. | https://github.com/mem0ai/mem0/blob/main/docs/migration/oss-v2-to-v3.mdx ; https://help.getzep.com/graphiti |
| E-PROD-7 | Claude apps memory organization: a background pass after each finished turn, with in-turn writes only on an explicit request. A short profile and a preferences file are always in context, plus topic, area and person files listed by path, description and aliases and opened on demand. Statements stay close to what the user said, and sensitive categories (including a minor's own age or date of birth) are never stored. | Claude apps (claude.ai) memory feature: observed product behavior, 2026-10 |

## Self-optimizing loops (E-OPT)
| Key | Finding | Source |
|---|---|---|
| E-OPT-1 | The convergent practice: a frozen scorer that only the artifact can't change; a cheap screen, then confirmation on a held-out set (GEPA: 3-example minibatch, then full validation; Darwin Gödel Machine: 10 → 60 → 200 tasks); an archive or Pareto front; text feedback. Repeated runs appear only in GRACE and Anthropic's skill-creator. | GEPA https://arxiv.org/abs/2507.19457 ; DGM https://arxiv.org/abs/2505.22954 ; skill-creator https://github.com/anthropics/skills |
| E-OPT-2 | Gaming: the Darwin Gödel Machine faked test logs and deleted hallucination-detection markers; AlphaEvolve's evaluation reported "cheating" via leaky verifiers. | https://sakana.ai/dgm/ ; https://arxiv.org/abs/2511.02864 |
| E-OPT-3 ✓ | Preregistered study: GEPA, MIPROv2 and TextGrad raised exact match by 21–34 points while breaking an output contract nobody scored (14–47% of responses cut off before the JSON). Content regression was small once the token cap was raised; the lesson is that unscored contracts break. | https://arxiv.org/abs/2610.13287 (2026-10-09) |

## Memory integrity (E-INT)
| Key | Finding | Source |
|---|---|---|
| E-INT-1 ✓ | As an LLM keeps consolidating memory, its usefulness rises, then falls, and can drop below no memory. Consolidating from correct solutions, GPT-5.4 then failed 54% of ARC problems it had solved before. Keeping raw episodes is competitive. Advice: treat raw episodes as evidence and gate consolidation. | https://arxiv.org/abs/2605.12978 (2026-05-13) |
| E-INT-2 | ACE "context collapse": one full rewrite shrank a playbook from 18,282 to 122 tokens, with accuracy falling 66.7 → 57.1 (below the 63.7 no-adaptation baseline). The fix is itemized edits merged by code. | https://arxiv.org/abs/2510.04618 |
| E-INT-3 | Mem-α's compression reward cut memory from 4.0k to 0.4k tokens and its score from 0.666 to 0.558. Memory-R1 and MemEvolve reward only answers and never check the store. | https://arxiv.org/abs/2509.25911 ; https://arxiv.org/abs/2508.19828 ; https://arxiv.org/abs/2512.18746 |
| E-INT-4 | Store-level metrics exist (HaluMem: extraction recall/accuracy, update accuracy, hallucination and omission). Their matchers were not validated against humans. An obsolete competing fact cut accuracy from 100% to 44.6%, so stale hurts more than missing. | https://arxiv.org/abs/2511.03506 ; https://arxiv.org/abs/2610.36308 |
| E-INT-5 | Memory echo and error propagation: write-back amplifies a source; agents copy errors from similar retrieved records. | https://arxiv.org/abs/2610.17763 ; https://arxiv.org/abs/2505.16067 |
| E-INT-6 | Of 457 real ChatGPT memories, 37.6% were judged misaligned by their own users (N=28): over-inference, privacy, outdated facts. | https://arxiv.org/abs/2609.33623 |

## Judges (E-JDG)
| Key | Finding | Source |
|---|---|---|
| E-JDG-1 | Yes/no checklists raised agreement between judges to κ ≈ 0.67 (humans agree with each other at ≈ 0.7). Judging against a reference beats judging without one. | CheckEval https://aclanthology.org/2025.emnlp-main.796/ ; https://arxiv.org/abs/2503.05061 |
| E-JDG-2 | Distractors flipped ~35% of pairwise verdicts versus ~9% of pointwise scores. | https://arxiv.org/abs/2504.14716 |
| E-JDG-3 ✓ | Under yes/no rubrics, judges were up to 50% likelier to wrongly pass criteria that their own family's outputs failed; "must not" rubrics were the worst case. Ensembling reduces this but doesn't remove it. Preference leakage was 23.6% for the same model, 8.9% for the same family, 2.8% across series. | https://arxiv.org/abs/2604.06996 (2026-04-08) ; https://arxiv.org/abs/2502.01534 |
| E-JDG-4 | Across 14 judges, TPR > 96% but TNR < 25%: judges miss failures. Overruling the majority when a minority flags a failure beat majority voting. Practitioner guidance: 100–200 labels per failure mode, with at least 50 passes and 50 failures. | https://arxiv.org/abs/2510.11822 ; https://hamel.dev/blog/posts/evals-faq/ |
| E-JDG-5 | Sample size for paired yes/no checks: n ≈ 7.84·d/δ². At d = 0.10 that is ~900 / 2,000 / 7,800 checks for 3 / 2 / 1 points. Judge error at TPR = TNR = 0.9 needs about 1.56× more. | https://arxiv.org/abs/2411.00640 (formula; the numbers are our arithmetic) |
| E-JDG-6 | Nine frontier judges from seven families gave about two effective votes; large panels add little. Temperature 0 isn't deterministic. | https://machinelearning.apple.com/research/correlated-llm-evaluation-panels ; https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ |

## Ground truth and synthetic lives (E-GT)
| Key | Finding | Source |
|---|---|---|
| E-GT-1 ✓ | AMemGym samples the profile, the state trajectories and the questions first; an LLM user then role-plays to reveal the states, and gold comes from the state values. Reported: 99.1% state exposure, 99.2% non-contradiction, gold–human agreement 0.96 vs human–human 0.92. | https://arxiv.org/abs/2603.01966 (2026-03-02) |
| E-GT-2 | Hand-made alternatives fare worse: LoCoMo's humans edited ~15% of turns and 6.4% of its gold answers are wrong. DynamicMem replaced 5 h per user of manual review with an LLM reviewer plus human escalation (97% coherent, 98% grounded). | https://arxiv.org/abs/2402.17753 ; https://arxiv.org/abs/2606.22877 |
| E-GT-3 | Raters judge best from a context card (the query plus the current and previous state value) rather than the full history. PersonaMem raters took ~1.5 min per item. | AMemGym (above); https://arxiv.org/abs/2504.14225 |
| E-GT-4 | LLM-simulated users agree with humans on relevance at κ ≈ 0.30 (humans with each other 0.43) and mark 40–60% of attributes relevant versus ~20% for humans. Simulators over-cooperate and give every detail up front. A reward model trained on 1,101 ratings reached ρ ≈ 0.3. | https://arxiv.org/abs/2606.06614 ; https://arxiv.org/abs/2603.11245 |
| E-GT-5 | What users want forgotten goes beyond GDPR categories (15 categories, N=146), and sensitivity depends on context. | https://arxiv.org/abs/2610.27607 |
| E-GT-6 | Generative Agents (Smallville, 25 agents) ran freely, so truth had to be read off afterwards. | https://arxiv.org/abs/2304.03442 (2023) |

## Who uses assistants, and for what (E-USE)
| Key | Finding | Source |
|---|---|---|
| E-USE-1 ✓ | 49% of US adults use chatbots: 66% of 18–29, 61% of 30–49, 42% of 50–64, 23% of 65+. A third of adults under 50 use one daily. One in five adults under 30 use them for emotional support or advice. | Pew, survey Feb 2026, https://www.pewresearch.org/internet/2026/06/17/americans-and-ai-2026-chatbots-smart-devices-and-views-on-impact/ |
| E-USE-2 ✓ | ~46% of messages come from users aged 18–25, and over 70% are not about work. Topics: practical guidance ~29%, seeking information ~24%, writing ~24% (two-thirds of it editing the user's own text), tutoring ~10%, technical help ~5%, relationships and personal reflection ~2%. | OpenAI / NBER w34255, https://www.nber.org/papers/w34255 (2025-09) |
| E-USE-3 ✓ | 19.2% of 12–21-year-olds have used chatbots for advice when sad, stressed or anxious, up from 13.1% a year earlier. | RAND, https://www.rand.org/news/press/2026/06/nearly-1-in-5-us-adolescents-and-young-adults-use-ai.html (2026-06) |
