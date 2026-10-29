## 🚀 Running Experiments for LoCoMo Dataset

### Dataset Setup

Download the LoCoMo dataset (JSON format) into the `app/locomo/data/` directory:

```bash
mkdir -p app/locomo/data
# Download from the LoCoMo repository:
# https://github.com/snap-research/locomo/tree/main/data
# Place locomo10.json in app/locomo/data/
```

The expected data path is `app/locomo/data/locomo10.json`.

### Configuration

All configuration is in `conf/config.yaml`. You will need to set:
- `openai.llm_api_base` — your Azure OpenAI (or OpenAI) endpoint
- `openai.embedding_api_base` — your embedding endpoint
- `openai.api_type` — `"azure"` or `"openai"`
- `openai.managed_identity` — Azure Managed Identity client ID (if using Azure)

### Running Memora

The entry point is `run_memora.py` with [Hydra](https://hydra.cc/) configuration — all parameters can be overridden from the command line.

**Default Memora (episodic + factual memory):**
```bash
python run_memora.py \
    llm.model="gpt-4.1-mini" \
    memory.memory_store="memora-semantic" \
    memory.enable_episodic_memory=True \
    retrieval.strategy="semantic"
```

**Segment-as-episodic variant** (uses raw segments as episodic memory, no LLM summarization):
```bash
python run_memora.py \
    llm.model="gpt-4.1-mini" \
    memory.memory_store="memora-segment-episodic" \
    memory.enable_episodic_memory=True \
    memory.use_segments_as_episodic=True \
    retrieval.strategy="semantic"
```

**With cue index and prompted retrieval:**
```bash
python run_memora.py \
    llm.model="gpt-4.1-mini" \
    memory.memory_store="memora-cue-prompt" \
    memory.enable_cue_index=True \
    retrieval.strategy="prompt"
```

**Retrieval strategies:**
- `retrieval.strategy="semantic"` → Memora (S): pure semantic similarity retrieval
- `retrieval.strategy="prompt"` → Memora (P): LLM-guided iterative policy retrieval

See `run.sh` for more complete example commands.

This script:
1. Builds the memory based on the conversations if the memory store doesn't exist.
2. Generates the answers for each question in `${result_folder}/memora_output.json`.
3. Evaluates the results using BLEU, F1 and LLM-AS-JUDGE in `${result_folder}/memora_eval.json`.
4. Generates the final result scores in `${result_folder}/memora_scores.json`.

Example output:
```
Mean Scores Per Category:
         bleu_score  f1_score  llm_score  count
category                                       
1           0.xxxx    0.xxxx     0.xxxx     xx
2           0.xxxx    0.xxxx     0.xxxx     xx
3           0.xxxx    0.xxxx     0.xxxx     xx

Overall Mean Scores:
bleu_score    0.xxxx
f1_score      0.xxxx
llm_score     0.xxxx
```

## Baselines

We support the following baselines: **Mem0**, **RAG**, and **full-context**.

```bash
# Run Mem0 baseline
python run_experiments.py memory.type=mem0 llm.model="gpt-4.1-mini" memory.memory_store="mem0-4.1-all"

# Run RAG baseline
python run_experiments.py memory.type=rag llm.model="gpt-4.1-mini" memory.memory_store="rag-4.1-all"

# Run full-context baseline (no memory, uses entire conversation)
python run_experiments.py memory.type=full_context llm.model="gpt-4.1-mini"
```

## 📏 Evaluation Metrics

We use several metrics to evaluate the performance of different memory techniques:

1. **BLEU Score**: Measures the similarity between the model's response and the ground truth
2. **F1 Score**: Measures the harmonic mean of precision and recall
3. **LLM Score**: A binary score (0 or 1) determined by an LLM judge evaluating the correctness of responses