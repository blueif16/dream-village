# Baseline protocol — how Memora v2 gets compared against these numbers

Every number in the baseline table was produced under exactly this setup. A new system is comparable only if it changes **nothing below except the memory system itself**.

## Frozen (do not change)
- **Backbone:** `claude-haiku-5-5`, effort `low`, for every memory-system call, the answerer and the judge. No temperature or seed (Haiku 5.5 rejects them).
- **Embeddings:** `gemini-embedding-2`, 3072 dims, with the addendum §5 prefixes (`memobench.embed`).
- **Consolidation threshold:** t = 0.9291, calibrated on LoCoMo conversation 1 (`runs/calibration/calib-c1.json`).
- **Answer prompts:**
  - LoCoMo uses Memora `ANSWER_PROMPT_COMBINED` for every system (`vendor/Memora/app/locomo/answerer.py`).
  - LongMemEval uses the repo-default `longmemeval` template (`vendor/Memora/app/longmemeval/answerer.py`).
- **Judges:**
  - LoCoMo uses Mem0's `ACCURACY_PROMPT`, byte-identical to upstream.
  - LongMemEval uses upstream `evaluate_qa.py` `get_anscheck_prompt`, including abstention.
  - Labels come back through structured outputs.
- **Data:** `data/SHA256SUMS`. LoCoMo uses categories 1–4 (1,540 questions); category 5 is never answered. LongMemEval uses `longmemeval_s_cleaned.json`.
- **LongMemEval subset for Memora systems:** `docs/lme_subset24.json` (24 questions: 4 per type, 1 abstention, seed 0). Full context also has all 500.
- **Every deviation from the paper:** `docs/deviations.md`.

## Commands that produced the baselines
LoCoMo, run from `vendor/Memora/app/locomo`, with env `MEMOBENCH_RUN_ID=locomo-full MEMOBENCH_BUDGET_USD=<cap> MEMOBENCH_MAX_ATTEMPTS=30`:
```
C="memory.enable_episodic_memory=True memory.use_segments_as_episodic=True memory.enable_cue_index=True memory.update_score_threshold=0.9291"
uv run python run_memora.py memory.memory_store=memora-seg-cue   retrieval.strategy=semantic $C   # Memora-S (builds the store)
uv run python run_memora.py memory.memory_store=memora-seg-cue-P retrieval.strategy=prompt   $C   # Memora-P (copy of the S store)
uv run python run_memora.py memory.memory_store=memora-seg-cue-noabs +memory.embed_field=value retrieval.strategy=semantic $C
uv run python run_experiments.py memory.type=full_context
uv run python run_experiments.py memory.type=rag
```
LongMemEval, run from `vendor/Memora/app/longmemeval`:
```
uv run python run_experiments.py memory.type=full_context                         # all 500 (Batch)
SUB=$(python3 -c "import json;print(json.load(open('../../../../docs/lme_subset24.json'))['subset_idx'])")
uv run python run_memora.py memory.memory_store=lme-sub24 retrieval.strategy=semantic memory.update_score_threshold=0.9291 "eval.subset_idx=\"$SUB\""
```
Results table: `uv run python scripts/locomo_table.py <system>=<path/to/*_eval.json> ...`. It gives LLM-judge accuracy with 95% CIs bootstrapped by conversation, the held-out conversations 6–10, the paired P−S difference, and cost per conversation.

## Making Memora v2 comparable
1. **Same harness, answerer and judge.** Run Memora v2 so that only the retrieved context reaching the fixed answerer differs. The design session's harness contract (`docs/design-session-prompt.md`) formalizes this as `systems/<name>/`.
2. **Same question sets.** That means all 1,540 LoCoMo questions, the 24-question LongMemEval subset, and optionally all 500 for full context.
3. **Paired comparisons.** Compare per question against the baseline `*_eval.json` files with `memobench.stats.paired_cluster_bootstrap`, clustered by conversation (LoCoMo) or question (LongMemEval).
4. **Caching.** Reusing the response cache (`runs/cache.sqlite`) and the embedding cache (`runs/embed_cache.sqlite`) makes re-running any baseline free and exact.
