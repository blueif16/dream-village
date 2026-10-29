# LLM layer spec (`memobench/`)

Every Claude call in this project goes through this layer: Memora's adapter, baselines, answering, judging.
It owns: request guards, the response cache, real-time and Batch execution, pricing, the cost ledger, and the request log.
Constants and rules come from `docs/memora-plan-addendum.md` §4 (adapter, logging, pricing, ledger).

## Public API

```python
from memobench.pricing import cost_usd
from memobench.llm import LLM, Req, Result, BudgetExceeded, NeedsApproval, CostOverrun

cost_usd(usage: dict, *, batch: bool) -> float

req = Req(params: dict, meta: dict = {})
#   params = complete Messages API params (model, max_tokens, system?, messages, output_config?, ...)
#   meta   = run_id/system/conv_id/question_id/stage (+ optional prefix_group); logged, NOT part of the key
req.key -> str   # sha256 hex (64 chars) of canonical JSON of params (sort_keys=True, separators=(",", ":"))

llm = LLM(root="runs", run_id: str, budget_usd: float = 10.0, approval_usd: float = 50.0,
          client=None,                      # None -> anthropic.Anthropic(max_retries=0); tests inject a fake
          allowed_models=("claude-haiku-5-5",),
          max_batch_requests: int = 100_000, max_batch_bytes: int = 200_000_000,
          max_attempts: int = 6, sleep=time.sleep)   # sleep injectable so tests don't wait

llm.call(req) -> Result                                   # one real-time request, cache-first
llm.run(reqs, mode: "sync" | "batch", *, estimate_usd: float | None = None, approved: bool = False,
        concurrency: int = 16, poll_s: float = 30, batch_timeout_s: float | None = None) -> dict[str, Result]  # keyed by req.key
llm.estimate(reqs, *, batch: bool) -> float               # pre-run USD estimate (printed by run())
llm.lookup(req) -> Result | None                         # cache read only (C6); never calls the API
llm.spent_usd -> float                                    # this LLM instance's run total (excludes cache hits)
llm.total_usd -> float                                    # all runs, from the ledger file

@dataclass Result: key, text, message (dict, full response), stop_reason, usage (dict), cost_usd,
                   latency_ms, cached (bool), mode ("sync"|"batch"), batch_id (str|None), error (str|None)
```

## Behavior contract

### Guards (before any network call)
- G1. `params` containing `temperature`, `top_p`, `top_k`, or `seed` -> `ValueError`. These 400 on Haiku 5.5 and are frozen out.
- G2. `params["model"]` not in `allowed_models` -> `ValueError`.
- G3. `run()`: if `estimate_usd` (or `llm.estimate(...)` when None) exceeds `approval_usd` and not `approved` -> `NeedsApproval`; no API call is made.
- G4. Budget: before dispatching each request (or each batch), if `total_usd` already >= `budget_usd` -> `BudgetExceeded`; no further API calls.

### Estimate (no API calls)
- E1. `estimate(reqs, batch=...)` = Σ over cache misses of `cost_usd` with `input_tokens = ceil(len(json.dumps([system, messages])) / 3)` and `output_tokens = min(max_tokens, 1024)`. `run()` prints one line: `[memobench] <run_id> <mode> n=<misses>/<total> est=$<x.xxxx>`.

### Cache (never pay twice)
- C1. Responses are stored by `req.key` in `root/cache.sqlite`. A request whose key is cached returns `Result(cached=True)` without touching the client, and adds nothing to spend.
- C2. Key ignores dict ordering and `meta`; it changes when any param changes.
- C3. Within one `run()`, duplicate keys are executed once; every input req still gets its Result.
- C4. Only successful responses are cached (not errors).
- C5. A cache hit returns `cost_usd=0.0`, `error=None`, and writes no `requests.jsonl`/ledger line.
- C6. `lookup(req)` reads the cache only: a hit returns the same `Result` a cached `call()` returns (`cached=True`, C5); a miss returns `None`. It never touches the client and writes no log/ledger line (capture/prefetch tooling uses it to tell which requests still need the API).

### Pricing (`cost_usd`) — addendum §4, verified Oct 20 2026
- `prompt_len = input_tokens + cache_creation_input_tokens + cache_read_input_tokens`; long-context iff `prompt_len > 100_000`; the whole request uses that tier.
- $/MTok, (≤100k | >100k): input 0.10 | 0.50 · output 0.50 | 2.50 · 5m cache write 0.125 | 0.625 · 1h cache write 0.20 | 1.00 · cache read 0.01 | 0.05.
- If `usage["cache_creation"]` has `ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens`, price each at its TTL; otherwise treat all `cache_creation_input_tokens` as 5m.
- `batch=True` -> × 0.5. Missing/None usage fields count as 0.

### Real-time path (`call`, `run(mode="sync")`)
- S1. Thread pool of `concurrency` workers.
- S2. Retry own loop (client has `max_retries=0`): retry on 429, 5xx (incl. 529 overloaded), and connection errors, with exponential backoff + jitter, honoring a `retry-after` header when present, up to `max_attempts` total calls (including the first); after the last attempt the last exception is re-raised. Each retry appends `{ts, key, attempt, error, sleep_s}` to `root/logs/retries.jsonl`. Non-retryable 4xx (400/401/403/404/413) raise immediately.
- S3. Overrun stop: once actual spend for this `run()` exceeds 1.5 × the estimate, stop dispatching new requests and raise `CostOverrun` (requests already finished stay cached/logged).

### Batch path (`run(mode="batch")`)
- B1. Submit only cache misses (after C3 dedup). `custom_id = req.key`.
- B2. Chunk so each batch has ≤ `max_batch_requests` requests and ≤ `max_batch_bytes` of serialized params.
- B3. Write `root/batches/<batch_id>.json` (batch_id, created_at, run_id, custom_ids, metas) immediately after `batches.create` returns, BEFORE polling.
- B4. Resume: at the start of `run()`, any manifest under `root/batches/` that isn't marked done and whose custom_ids cover requested keys is polled and collected instead of being resubmitted. A crashed process must never cause a second submission of the same request.
- B5. Poll `batches.retrieve` every `poll_s` until `processing_status == "ended"`, then iterate `batches.results`; match by `custom_id` (results come in any order). Mark the manifest done after collection (`"done": true` in the manifest JSON).
- B6. `succeeded` -> cache + log + ledger (priced with `batch=True`). `errored` with an invalid-request error -> `Result.error` set, not retried. `errored` (other), `expired`, `canceled` -> resubmitted once in a new batch; if still failing -> `Result.error`.
- B7. `batch_timeout_s`: if a batch hasn't ended in time, call `batches.cancel`, wait for "ended", collect what succeeded, and run the remaining requests through the real-time path.
- B8. Warm-up for prompt caching: if reqs carry `meta["prefix_group"]`, the first request of each group runs real-time first (and completes) before the batch with the rest is submitted.

### Thread safety
- T1. One `LLM` instance is shared by Memora's worker threads: concurrent `call()`s from many threads must not lose or corrupt cache entries, log lines, or ledger lines, and `spent_usd` must equal the sum of their costs.

### Log + ledger
- L1. One JSONL record per API response in `root/logs/requests.jsonl`: `run_id, system, conv_id, question_id, stage, model, effort, mode, batch_id, key, request, response, usage{input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens}, visible_text_chars, stop_reason, latency_ms, cost_usd, ts`.
- L2. `Result.text` = concatenation of `text` blocks only (Haiku 5.5 may emit `thinking` blocks first).
- L3. `root/ledger.jsonl` appends `{run_id, stage, key, cost_usd, mode, ts}` per paid response; `total_usd` = sum over the file (all runs); `spent_usd` = this instance's sum.
- L4. `stop_reason` of `max_tokens` or `refusal` is recorded in the log and on `Result` (not raised).

## Out of scope here
Memora adapter, embeddings, prompt templates, cache_control placement (callers put `cache_control` into params themselves), CLI.
