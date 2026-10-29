# LongMemEval Experiments

This module evaluates Memora on the [LongMemEval](https://arxiv.org/abs/2410.10813) benchmark, which tests long-term memory capabilities of conversational agents across multiple question types.

## Setup

1. Install dependencies from the root of the repository:

```bash
pip install -e .
```

2. Download the LongMemEval dataset into the `app/longmemeval/data/` directory:

```bash
mkdir -p app/longmemeval/data
# Download from the LongMemEval repository:
# https://github.com/xiaowu0162/LongMemEval
# Place the dataset files in app/longmemeval/data/
```

3. Configure your Azure OpenAI (or OpenAI) credentials in `conf/config.yaml`:
   - Set `openai.llm_api_base` to your Azure OpenAI endpoint
   - Set `openai.embedding_api_base` to your embedding endpoint
   - Set `openai.api_type` to `"azure"` or `"openai"`
   - Set `openai.managed_identity` if using Azure Managed Identity

## Running Experiments

### Sequential (Default)

From the `app/longmemeval/` directory:

```bash
# Quick debug run (small subset)
python run_memora.py general.debug=True

# Basic run with default settings
python run_memora.py \
    llm.model="gpt-4.1-mini" \
    memory.memory_store="memora-semantic" \
    memory.enable_episodic_memory=True \
    retrieval.strategy="semantic"

# Run with cue index and prompted retrieval
python run_memora.py \
    llm.model="gpt-4.1-mini" \
    memory.memory_store="memora-cue-prompt" \
    memory.enable_cue_index=True \
    retrieval.strategy="prompt"

# Full dataset evaluation
python run_memora.py \
    llm.model="gpt-4.1-mini" \
    memory.memory_store="memora-full" \
    eval.subset_idx=-1

# Force rebuild memory store
python run_memora.py \
    llm.model="gpt-4.1-mini" \
    memory.memory_store="memora-rebuild" \
    memory.force_rebuild=True
```

### Parallel Execution

There are two levels of parallelism available:

#### 1. In-Process Parallel Search (Single Process, Multiple Threads)

Enable `eval.parallel_search=true` to process questions concurrently within a single run using a thread pool. Output is written in JSONL format (line-append safe).

```bash
python run_memora.py \
    eval.subset_idx="1:100" \
    eval.parallel_search=true \
    eval.max_workers=6 \
    memory.memory_store="memora-parallel"
```

#### 2. Multi-Process Parallel Splits (Multiple Processes via tmux)

For full-dataset runs, split the 500 questions across 5 isolated tmux sessions — each with its own memory store to avoid ChromaDB lock contention.

**Prerequisites:**
- `tmux` installed (`sudo apt install tmux` or `brew install tmux`)
- Azure OpenAI endpoint configured via `AZURE_OPENAI_ENDPOINT` env var

**Run:**

```bash
# Standard parallel run (5 splits of 100 questions each)
./run_parallel_splits.sh

# Force kill existing sessions and restart
./run_parallel_splits.sh --force

# Retry only failed questions (skip questions that already have memories)
./run_parallel_splits.sh --skip-existing

# Force rebuild all memory stores from scratch
./run_parallel_splits.sh --force-rebuild

# Use a custom output directory
./run_parallel_splits.sh --output-dir longmemeval_outputs_v2
```

**Monitor progress:**

```bash
./check_status.sh longmemeval_outputs

# Or attach to a specific session
tmux attach -t longmemeval_split_0
```

**Aggregate results after all splits complete:**

```bash
./gather_results.sh longmemeval_outputs

# Or directly:
python aggregate_scores.py longmemeval_outputs --num-splits 5
```

This merges per-split score files and produces weighted-average metrics (overall and per question type).

**Load balancing across multiple endpoints:**

Edit the `ENDPOINTS` array in `run_parallel_splits.sh` to distribute splits across different Azure OpenAI endpoints:

```bash
ENDPOINTS=(
    "https://endpoint-1.openai.azure.com/"
    "https://endpoint-2.openai.azure.com/"
    "https://endpoint-3.openai.azure.com/"
    "https://endpoint-4.openai.azure.com/"
    "https://endpoint-5.openai.azure.com/"
)
```

**Cleanup:**

```bash
# Kill all tmux sessions
for i in {0..4}; do tmux kill-session -t "longmemeval_split_$i" 2>/dev/null || true; done

# Remove outputs and memory stores
rm -rf longmemeval_outputs/split_*
rm -rf memory_store/memora_split_*
```

See `run.sh` for more example commands.

## Configuration

All configuration is managed via [Hydra](https://hydra.cc/) and defined in `conf/config.yaml`. Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `llm.model` | LLM model name | `gpt-4.1-mini` |
| `memory.memory_store` | Name for the memory store (used as folder name) | `memora_debug` |
| `memory.enable_episodic_memory` | Enable episodic memory extraction | `True` |
| `memory.enable_cue_index` | Enable cue indexing for retrieval | `True` |
| `memory.force_rebuild` | Force rebuilding the memory store | `False` |
| `memory.skip_existing` | Skip questions that already have memories (for retries) | `False` |
| `retrieval.strategy` | Retrieval strategy: `semantic`, `prompt`, or `grpo` | `prompt` |
| `eval.subset_idx` | Subset index: int, range `"1:100"`, or comma-list `"1,5,10"` | `-1` (full) |
| `eval.parallel_search` | Enable multi-threaded question processing | `False` |
| `eval.max_workers` | Thread pool size for parallel search | `10` |
| `eval.answer_template` | Answer prompt style: `nemori`, `evermemos`, `longmemeval` | `nemori` |
| `general.debug` | Debug mode (smaller dataset) | `False` |

## What the Experiment Does

1. **Builds memory** from conversations (if the memory store doesn't exist or `force_rebuild=True`)
2. **Retrieves and answers** questions using the configured retrieval strategy
3. **Evaluates results** using BLEU, F1, and LLM-as-judge metrics
4. **Saves outputs** to `results/` with timestamped folders containing:
   - `parameters.yaml` — experiment configuration
   - `memora_<strategy>_output.json` — model outputs (JSON dict in sequential mode, JSONL in parallel mode)
   - `memora_<strategy>_eval.json` — evaluation results

## Evaluation Metrics

- **BLEU Score**: N-gram overlap between model response and ground truth
- **F1 Score**: Token-level precision and recall
- **LLM Score**: Binary correctness score from an LLM judge
