# Addendum to the Memora Experiment Plan (Oct 20, 2026)

Read this together with the original plan. Where they conflict, this file wins.

## 1. Decisions (from Shiran)

- **Judge: `claude-haiku-5-5` only.** Effort low. Use Mem0's `llm_judge` prompt verbatim for LoCoMo and LongMemEval's `evaluate_qa.py` prompt for LongMemEval. Return the label through structured outputs. These are internal comparisons only: the paper judged with gpt-4o-mini, so its numbers are reference points, not like-for-like. Tag judge calls `stage=judge` and leave them out of per-system cost.
- **Embeddings: `gemini-embedding-2`** via the Gemini API (`GEMINI_API_KEY`). Not covered by Claude credits; cost is negligible. No OpenAI key is needed anywhere.
- **LongMemEval_S subset size: not decided yet.** Measure first (section 8), then ask Shiran.

## 2. Verify in the repo before writing code

I couldn't read source files from chat (GitHub blocked automated access), so check these first and send Shiran a summary of 10 lines or less:

1. The policy retriever prompt in `src/memora/retriever`. It is **not** in the paper's Appendix A. Record its token size per step.
2. The LoCoMo `conf/config.yaml`: consolidation top-k, similarity threshold, retrieval top-k, policy budget, and max steps.
3. Consolidation: is the update LLM call **skipped** when no existing abstraction reaches the threshold? This is the largest cost lever.
4. Cue anchors: are they generated per entry right after create-or-update, or in one final pass? This decides whether they can be batched later.
5. The factual extraction output format and its parser (MemIndex/MemValue text, or JSON).
6. LoCoMo eval: confirm category 5 (adversarial) is excluded, and that the answer prompt is Mem0's `ANSWER_PROMPT`.
7. Where embeddings are created: a ChromaDB embedding function, or an explicit client call. Replace them at that spot.
8. Every OpenAI-specific parameter in use: `temperature`, `seed`, `response_format`, `logprobs`, `n`.

## 3. Headline configuration

The paper's 0.863 config uses raw segments as episodic memory, plus factual memory, cue anchors, and policy retrieval. The README shows these flags only in separate commands, so turn them all on together:

```
cd app/locomo
python run_memora.py \
  llm.model="claude-haiku-5-5" \
  memory.memory_store="memora-seg-cue" \
  memory.enable_episodic_memory=True \
  memory.use_segments_as_episodic=True \
  memory.enable_cue_index=True \
  retrieval.strategy="prompt"
```

For Memora-S, run the same command with `retrieval.strategy="semantic"` and the **same** `memory_store`. The runner only builds a store when it doesn't exist yet, so S and P share one build. The segment-as-episodic config makes no episodic summarization LLM call.

Full-context baseline: `python run_experiments.py memory.type=full_context llm.model="claude-haiku-5-5"`.

## 4. Anthropic adapter

- Replace the OpenAI client (in `src/memora/utils`) with an adapter that keeps the same interface. Use model `claude-haiku-5-5`.
- Drop `temperature`, `top_p`, `top_k`, and `seed`, because they return 400 errors. Don't use assistant prefill.
- Use effort `low` for every stage tonight, including policy steps and answering. The answering settings must be identical across systems. Parameter shape: https://platform.claude.com/docs/en/build-with-claude/effort
- Stages that return JSON (segmentation, the update decision, the judge label) use structured outputs. Keep the prompt text verbatim from Appendix A or the repo, so only the output-format plumbing changes. Record this as a deviation.
- Retry with backoff on 429 and 5xx errors, and log every retry.

### Logging

Write one JSONL record per request with these fields:

- `run_id`, `system`, `conv_id`, `question_id`, `stage`, `model`, `effort`
- the full request and the full response
- `usage`: `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`
- `latency_ms` and `cost_usd`

Thinking tokens are billed inside `output_tokens`. Also log the token count of the visible output, so the thinking share per stage can be measured. This is the biggest cost unknown.

### Pricing function (verified Oct 20, 2026)

Compute `prompt_len = input + cache_creation + cache_read`. A request is long-context if `prompt_len > 100_000`. The whole request is priced at that tier.

| $/MTok | ≤100k prompt | >100k prompt |
|---|---|---|
| Input | 0.10 | 0.50 |
| Output | 0.50 | 2.50 |
| 5m cache write | 0.125 | 0.625 |
| Cache read | 0.01 | 0.05 |

Batch API requests cost 50% of these rates.

### Ledger rules

- Keep a running total per run and overall.
- **Hard stop at $10 tonight.**
- Print a cost estimate before every run.
- If any run is estimated above $50, stop and ask Shiran.

## 5. gemini-embedding-2 port

Source: https://ai.google.dev/gemini-api/docs/embeddings

- Model code: `gemini-embedding-2`. **`task_type` is not supported.** Put the task as a prefix in the input text instead.
- **Silent-failure trap:** if you pass a list of strings directly, you get ONE combined embedding back. Wrap each string in its own `Content` object (or use `batchEmbedContents`), and assert `len(outputs) == len(inputs)`.
- Use the default 3072 dimensions (lower dimensions are auto-normalized, but the stores are small). The 8,192-token input limit doesn't matter here.

### Prefix scheme

The docs say to use each format consistently, and not to use the sentence-similarity format for retrieval.

| Index | Used for | Format |
|---|---|---|
| Consolidation index | abstraction ↔ abstraction (symmetric) | `task: sentence similarity \| query: {abstraction}` |
| Retrieval index | abstractions and cue anchors, stored as documents | `title: none \| text: {abstraction or cue}` |
| Queries | including policy REFINE re-queries | `task: question answering \| query: {query}` |

So each abstraction gets two vectors, stored in two Chroma collections.

### Cost and caching

- Log every embedding call in the ledger as `stage=embed`.
- Confirm the price at https://ai.google.dev/gemini-api/docs/pricing. A third-party listing says ~$0.20/M text tokens. Expect under $0.02 per LoCoMo conversation.
- Cache embeddings by a hash of the prefix-formatted text.

### Threshold calibration

The paper's 0.8 threshold was tuned on text-embedding-3-small and doesn't transfer to this model.

1. For conversation 1, run segmentation and factual extraction only. Cache the outputs by segment hash so the real build reuses them at no extra Claude cost.
2. Embed every candidate abstraction with the consolidation prefix.
3. Going in session order, compute `s_i` = the maximum cosine similarity between candidate `i` and every earlier candidate.
4. Choose a threshold `t` such that **21–25%** of candidates have `s_i ≥ t`. Why this target:
   - The paper's update rate at 0.8 is 21% (88 of 432 candidates).
   - At threshold 0.6 the LLM merged 68% of candidates (Table 17), so the LLM accepts most of what passes. That makes the pass rate a good stand-in for the update rate.
5. Build with `t` and report the update rate it actually produced:
   - Acceptable range: 15–30%. Outside that range, stop and ask Shiran.
   - Freeze `t` for all conversations and benchmarks.
   - Save the histogram of `s_i`.

## 6. Pilot runbook (tonight)

0. **Shiran:** check the Console balance. Export `ANTHROPIC_API_KEY` (the Console org with the credits) and `GEMINI_API_KEY`.
1. Send Shiran the section 2 findings.
2. Port the adapter and the embeddings. Unit test on one session: the parse rate should be 100%, and any failure is logged with its raw output.
3. Smoke test: conversation 1, session 1, plus 10 questions, through Memora-S, Memora-P, and full-context. Check the ledger math against Console usage.
4. Calibrate `t` on conversation 1 (section 5).
5. Build stores for conversations 1 and 2 in parallel. These are the first two in `locomo10.json` order, fixed in advance. Within a conversation, consolidation runs sequentially in session order.
6. Run queries with a worker pool across questions:
   - Memora-S and Memora-P on the shared store.
   - Full-context. Cache the conversation prefix only if the answer template already puts the conversation before the question. Otherwise skip caching: it's ~$0.40 per conversation and not worth changing the protocol.
7. Judge all answers (`stage=judge`).
8. Report the results in section 7's format.

**Stop and report if:** the ledger passes $10, the parse failure rate exceeds 2%, or the achieved update rate falls outside 15–30%.

## 7. What the pilot reports

### Slide table

| System | LLM-judge (95% CI) | Paper (reference) | $ / conversation (excl. judge) | Retrieved context tokens / query | LLM calls / query |
|---|---|---|---|---|---|
| Memora-P | | 0.863 | | | |
| Memora-S | | 0.849 | | | |
| Full context | | 0.825 | | | 1 |

Footnote: *Pilot: 2 of 10 LoCoMo conversations (n≈300 questions, categories 1–4), single run, no seed. Backbone and judge `claude-haiku-5-5` at low effort; the paper used gpt-4.1-mini and a gpt-4o-mini judge. Embeddings `gemini-embedding-2`, consolidation threshold recalibrated to t=__. Memora build cost is shared by S and P.*

### Confidence intervals

- With n≈300, a score near 0.85 has a 95% CI of about ±4 points.
- For S vs P, report the paired difference on the same questions (bootstrap or McNemar). Note that this is still underpowered tonight.

### Metrics for extrapolation

- Per stage: number of calls, input/output tokens, and cost.
- Candidates per conversation, consolidation LLM calls per candidate, and the update rate.
- Store size: entries and cue anchors.
- Policy steps per query.
- Build wall-clock time, and query latency p50/p95.
- Thinking share of output per stage.
- Parse failures.
- Extrapolated cost of full LoCoMo with all five systems.

## 8. LongMemEval_S: deciding the subset size

1. After the LoCoMo pilot, ingest **2** LongMemEval_S histories through Memora on the standard API (~$1–2). LoCoMo's rates won't carry over, because these sessions are filler-heavy with long assistant turns.
2. Check whether haystack sessions repeat across the 500 questions:
   - If they do, cache segmentation and extraction keyed on (session text, timestamp).
   - Consolidation can't be shared, because it depends on store state.
3. Bring Shiran:
   - the measured cost per question;
   - estimates for 50 / 100 / 500 questions, split into standard API and batch;
   - the full-context baseline cost. Its ~115k-token histories probably exceed 100k tokens in Haiku 5.5's tokenizer, which means long-context rates. Each question is independent, so it can use the Batch API.
4. Run nothing until Shiran picks a size. Any single run above $50 needs her approval.
5. Stratify the subset by `question_type` and keep abstention (`_abs`) items in proportion.
6. Answer template: the repo default is `nemori`, and the paper doesn't say which it used. Use the default and log it.

## 9. Deviations log (keep it current; it goes in the slide footnote)

| | Paper | Ours |
|---|---|---|
| Backbone | gpt-4.1-mini | claude-haiku-5-5, effort low |
| Judge | gpt-4o-mini | claude-haiku-5-5 |
| Embeddings | text-embedding-3-small | gemini-embedding-2, task prefixes, 2 vectors per abstraction |
| Consolidation threshold | 0.8 | t, calibrated (section 5) |
| Seed | 42 | none; full request/response logs |
| JSON stages | prompt-formatted | structured outputs, prompt text unchanged |
| GRPO retriever | Qwen, trained | skipped |

## 10. Caveats for the talk

- **Small sample.** With n≈300, the CI is about ±4 points. The paper's S-vs-P gap is 1.4 points and P-vs-full-context is about 4 points, so the pilot shows where Memora on Haiku 5.5 lands and what it costs. It can't rank the systems.
- **Same-family judge.** Haiku grades Haiku's answers. Later, hand-label ~50 pilot answers as part of Phase 2's ~100-item judge validation, to measure how often the judge agrees.
- **Mem0 ablation row.** Table 3's "w/o abstraction (= Mem0)" row equals the Mem0 baseline in every column, so it was likely reused rather than run separately. Our no-abstraction ablation isn't directly comparable to 0.653.
- **Backbone strength.** A 2026 backbone on ~26k-token conversations may close the gap to full context. The paper's scale argument gets a fairer test on LongMemEval.
