# Deviations log (addendum §9 — goes in the slide footnote)

| | Paper / upstream | Ours | Decided |
|---|---|---|---|
| Backbone | gpt-4.1-mini | claude-haiku-5-5, effort low | addendum |
| Judge | gpt-4o-mini | claude-haiku-5-5, effort low | addendum |
| Embeddings | text-embedding-3-small | gemini-embedding-2, task prefixes, 2 vectors per abstraction | addendum |
| Consolidation threshold | 0.8 | t, calibrated (addendum §5) | addendum |
| Seed / temperature | seed 42; temperature 0 on answer/judge/policy/segmentation | none (400 on Haiku 5.5); full request/response logs | addendum |
| JSON stages | OpenAI structured outputs / json_object | Claude structured outputs, prompt text unchanged | addendum |
| GRPO retriever | Qwen, trained | skipped | addendum |
| LoCoMo answer prompt | Mem0 `ANSWER_PROMPT` (frozen constant); repo uses per-system templates | Memora `ANSWER_PROMPT_COMBINED` as the single fixed answer prompt for every system | Shiran, 2026-10-21 |
| RAG baseline | repo config: chunk 1000, k=1 | chunk 500, k=3 (plan) | Shiran, 2026-10-21 |
| Memora-P policy steps | config says 4 (unused) | 5 (what the code runs) | Shiran, 2026-10-21 |
| No-abstraction ablation | no flag in repo | new flag: embed the fact value instead of the abstraction index; extraction + consolidation unchanged | Shiran, 2026-10-21 |
| LoCoMo category 5 | answered, then dropped at judging | not answered (score unaffected) | cost-only |
| LongMemEval answer template | addendum assumed `nemori` | repo default `longmemeval` | addendum intent: "use the default" |
| LongMemEval judge | repo: Nemori/Zep-style copy | upstream `evaluate_qa.py` verbatim (`vendor/upstream/`) | frozen constant |
| LongMemEval ingest | sessions >32 messages keep only the first 32 (upstream bug) | kept as upstream: 46/23,867 sessions, 0 evidence sessions | reproduction fidelity |
| Prompt caching | none | user message split into text blocks at the fixed-prefix boundary with `cache_control`; prompt text unchanged | plumbing only |
| Segmentation schema | `Episode.indices` described as "0-based" while the prompt numbers messages 1-based and the code subtracts 1 → Haiku followed the schema, every segmentation failed validation and fell back to fixed batches | schema description aligned with the prompt ("1-based, as numbered in the prompt"); prompt text unchanged; 3/3 sessions now segment via LLM | bug fix, 2026-10-21 |
| Consolidation threshold value | 0.8 (text-embedding-3-small) | **t = 0.9291** (frozen): 22.9% pass rate on LoCoMo conv 1 (327 candidates, prior = other batches); at 0.8 the pass rate would be 95.4% | addendum §5, 2026-10-21 |
| Retrieval score floor | `memory.query_score_threshold: 0.4` tuned for text-embedding-3-small | unchanged; Gemini cosines run higher, so the floor rarely binds and top_k=30 caps retrieval instead (not calibrated; flagged) | open |
| Second backbone (Argon rerun) | — | gemini-4-argon (thinking level low, mapped from effort low) for answerer, judge and Memora build; same prompts/data/threshold; Anthropic-shaped requests translated to `generate_content` with identical prompt text (`memobench/llm.py`); Gemini implicit caching; real-time, no Batch | Shiran, 2026-10-25 |
| Argon rerun completeness | — | stopped when both donated Gemini keys were revoked (2026-10-26 ~00:22 and ~00:35 PT); LoCoMo Memora S/P/no-abstraction never finished (stores quarantined as `*.PARTIAL-key-revoked`) | key revoked |
| Cross-backbone comparison | — | only under one judge: Argon's answers re-judged by the Haiku judge (Batch; `scripts/crossjudge.sh`). Argon's judge disagrees one-sidedly (299 Argon-only CORRECT vs 5 reverse); human audit in `docs/judge-audit/` | 2026-10-26 |
| LongMemEval RAG (Haiku, all 500) | — | completed 2026-10-26 from cached embeddings only (0 embedding calls; Oct 21 attempt hit a Gemini 429) | 2026-10-26 |
