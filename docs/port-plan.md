# Memora → Claude Haiku 5.5 + gemini-embedding-2 port plan

Authority: `docs/memora-plan-addendum.md` (§4 adapter, §5 embeddings), `docs/deviations.md` (decisions), `docs/llm-layer-spec.md` (LLM layer API).
Paths below are relative to `vendor/Memora/` unless they start with `memobench/`, `docs/`, `data/`, `tests/`.

## Shared pieces (already exist — use, don't rewrite)
- `memobench/runtime.py: get_llm()` → the process-wide `LLM` (spec API: `call(Req)`, `run(reqs, mode)`). Currently a stub that raises NotImplementedError; it is being implemented in parallel. Code against the spec.
- `memobench/context.py: set_meta(**kw) / get_meta()` → per-thread metadata. Every worker task sets `system`, `conv_id`, `question_id` before LLM calls; the adapter adds `stage`.
- Frozen request shape: `model="claude-haiku-5-5"`, `output_config={"effort": "low", ...}`, `max_tokens` set explicitly (thinking counts toward it), never `temperature/top_p/top_k/seed`.
- Structured outputs: `output_config["format"] = {"type": "json_schema", "schema": anthropic.transform_schema(PydanticModel)}` (exported by anthropic 2.0.0; verify its signature in `.venv/.../anthropic/lib/_parse/_transform.py`), then `PydanticModel.model_validate_json(result.text)`.
- Prompt caching = split ONE user message into consecutive text blocks whose concatenation is byte-identical to the original prompt; put `"cache_control": {"type": "ephemeral"}` on the block that ends the reusable prefix (`"ttl": "1h"` for batch runs). Only worth it when the prefix is ≥512 Haiku tokens (≈2,000+ chars).
- System names (meta `system`): `memora-s`, `memora-p`, `memora-noabs`, `full-context`, `rag`. Stages: `segmentation, extraction, episodic, cue, update, policy, answer, judge, embed, other`.

## Part 1 — core (src/memora + memobench/embed.py)

**1a. LLM wrapper** `src/memora/utils/llm.py` (`ChatCompletionModel`, `invoke` at :141-148):
- `_determine_model_type` (:76-82, 93-110): names starting with `claude` → new type `"anthropic"` (today any non-GPT name loads a local HuggingFace model).
- Move `import torch` / `transformers` (:12-13) inside the HF branch so they're only imported when used. Same for any other heavy import on the anthropic path.
- `invoke` for anthropic: str input → `.format(**prompt_args)` if prompt_args (same as :170-182) → one user message; list input → leading `system` messages become the `system` param. Drop OpenAI-only kwargs (temperature, seed, top_p, n, response_format dicts) silently-but-once-logged. `max_tokens=8192`.
- `response_format` (pydantic class) → structured outputs as above; return the parsed object exactly where upstream returns `.message.parsed` (:262-269, 308-310). Parse failure / `stop_reason in {max_tokens, refusal}` → append `{stage, source, raw_text, error}` to `runs/logs/parse_failures.jsonl` and return what upstream returns on a content-filter error (:278-302).
- No `response_format` → return the text (`Result.text`), like `.message.content` today.
- Stage = map `source` (e.g. `PromptedPolicyRetriever` → `policy`); meta = `{**get_meta(), "stage": stage, "source": source}`; call `get_llm().call(Req(params, meta))`; if `result.error` → raise.
- Cache split for the policy prompt: `LLM_POLICY_PROMPT` (`src/memora/retriever/prompted_policy_retriever.py:27-94`) is pre-formatted before `invoke` (:263-280). Register its static prefix (template text before the first `{` placeholder, ≈620 tokens) keyed by source; if the incoming user text starts with that prefix, send `[prefix + cache_control][remainder]`.
- Wrapper call sites to keep working (all must route through the new path): segmentation `app/locomo/providers/memora/segmentation.py:196`; extraction `src/memora/builder/chat_memory_builder.py:103`; episodic `chat_memory_builder.py:178`; cue `src/memora/core/cue_index_generator.py:200`; update `src/memora/builder/memory_builder.py:614`; policy `prompted_policy_retriever.py:276`.

**1b. Embeddings** — new `memobench/embed.py`:
- `embed(texts: list[str], kind: Literal["consolidation", "document", "query"]) -> list[list[float]]`; prefixes (addendum §5): consolidation `task: sentence similarity | query: {t}`, document `title: none | text: {t}`, query `task: question answering | query: {t}`.
- Gemini `gemini-embedding-2` via `google.genai` (`GEMINI_API_KEY`), one `types.Content` per text (a bare list of strings returns ONE combined embedding), ≤100 texts per request, `assert len(out) == len(in)`, 3072 dims, retry with backoff on 429/5xx.
- Cache by sha256 of the prefixed text in `runs/embed_cache.sqlite`; thread-safe; ledger line per API call in `runs/ledger.jsonl` with `stage="embed"`, token estimate (chars/4) and cost at `EMBED_USD_PER_MTOK = 0.20` (unverified constant, keep it named).
- Replace `src/memora/utils/embedding.py:93-108` (`generate_embeddings`) and the Chroma embedding function `src/memora/db_clients/chromadb_client.py:19-28, 71-75`. Chroma 1.5.9 calls `embed_query(input=...)` for `collection.query(query_texts=...)` and `__call__` for documents (`.venv/.../chromadb/api/models/CollectionCommon.py:751-797`).
- Two vectors per abstraction = two collections per user: retrieval `agent_memory_<user>` (docs → `document`, queries → `query`) and consolidation `agent_memory_<user>__consol` (docs and queries → `consolidation`). Every upsert/delete of a factual abstraction mirrors into both; cue-anchor rows go to the retrieval collection only. The consolidation candidate search (`memory_builder.py:348-355` and the store method it calls) queries the consolidation collection; every retrieval query (`src/memora/core/memory.py:432-442` and friends) queries the retrieval collection. Store code: `src/memora/core/local_memory_store.py:84-98, 153-166, 208`.

**1c. No-abstraction ablation flag** `cfg.memory.get("embed_field", "index")`: when `"value"`, the text that is embedded (both collections) and BM25-indexed for a factual entry is its value instead of its index. Extraction and consolidation logic unchanged.

## Part 2 — apps (app/locomo, app/longmemeval)

**2a. Plumbing**
- Data: runners resolve `./data/...` relative to the launch dir. Create symlinks `app/locomo/data/locomo10.json → <repo>/data/locomo10.json` and `app/longmemeval/data/longmemeval_s_cleaned.json → <repo>/data/longmemeval_s_cleaned.json` (absolute targets).
- `app/locomo/utils.py:10` imports mem0 at module level → import it lazily inside the mem0 functions. Same for `bert_score`/`sentence_transformers` anywhere on the eval path.
- Config (`app/locomo/conf/config.yaml`): `llm.model: claude-haiku-5-5`; RAG `memory.chunk_size: 500`, `memory.num_chunks: 3`; add `eval.conversations: null` (list of conversation indices to run, null = all), `eval.max_questions: null` (per conversation), `memory.max_sessions: null` (per conversation) for smoke tests.
- Category 5 is never answered (`app/locomo/evals.py:37-39` drops it at judging anyway; skip it in every answering loop).
- `set_meta(system=..., conv_id=..., question_id=...)` at the start of every worker task / question.

**2b. One fixed answering agent** (decision: Memora `ANSWER_PROMPT_COMBINED`, `app/locomo/prompts.py:108-146`, verbatim, rendered exactly as `app/locomo/providers/memora/search.py:415-418` renders it) used by every system. Put it in one function (e.g. `app/locomo/answerer.py: build_answer_req(memories_text, question, *, cache_memories: bool) -> Req`), `max_tokens=4096`, stage `answer`. Blocks: `[text before {{memories}} + cache_control]`, `[memories text (+ cache_control 1h when cache_memories)]`, `[question tail]`.
- Memora S/P/noabs (`app/locomo/providers/memora/search.py:454, 467, 477`, client at :53): build the req with the retrieved memories and call `get_llm().call(...)` inline (real-time). Run conversations in parallel (thread pool over conversations, questions sequential within).
- Full context (`app/locomo/providers/full_context.py`): reuse its conversation-rendering code to make the context text; it goes in the `{{memories}}` slot with `cache_memories=True`; collect all reqs and submit once via `get_llm().run(reqs, mode="batch")` with `meta["prefix_group"] = conv_id`.
- RAG (`app/locomo/providers/rag.py`): chunk 500 tokens, k=3, embeddings via `memobench.embed` (chunks `document`, questions `query`); retrieved chunks go in the `{{memories}}` slot; answers via one `run(mode="batch")`.
- `app/locomo/run_experiments.py:88` calls `generate_scores` with 1 arg against a 2-arg signature (`evals.py:97`) — fix the call.

**2c. Judge (batch)**
- LoCoMo: `app/locomo/metrics/llm_judge.py` `ACCURACY_PROMPT` (byte-identical to Mem0's) verbatim; structured output `{"label": "CORRECT" | "WRONG"}`; all answers of a results file → one `get_llm().run(reqs, mode="batch")`, stage `judge`, `max_tokens=2048`; write results in the format `evals.py`/`generate_scores` already read.
- LongMemEval: replace the repo's question-type prompts (`app/longmemeval/evals.py:24-134`) with `get_anscheck_prompt` from `vendor/upstream/lme_evaluate_qa.py` verbatim (incl. the abstention variant for `_abs` ids); structured output `{"label": "yes" | "no"}`; batch.
- LongMemEval Memora answers (`app/longmemeval/providers/memora/search.py:197, 206, 215`) → real-time via `get_llm().call`, repo default template `longmemeval`.
