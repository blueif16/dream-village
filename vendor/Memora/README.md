<h1 align="center">Memora: A Harmonic Memory Representation Balancing Abstraction and Specificity</h1>

<div align="center">

![Python Version](https://img.shields.io/badge/Python-3776AB?&logo=python&logoColor=white-blue&label=3.12)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)
[![arXiv](https://img.shields.io/badge/arXiv-2602.03315-b31b1b.svg)](https://arxiv.org/abs/2602.03315)

</div>

## Introduction

Memora is designed to take the headache out of agent memory. Instead of worrying about what and when to store memory or how to pull back the right memory at the right moment, agents can rely on Memora to handle it all. By providing an easy-to-use framework, Memora frees developers from low-level memory management and makes intelligent memory a built-in capability.

> **📄 Paper:** [Memora: A Harmonic Memory Representation Balancing Abstraction and Specificity](https://arxiv.org/abs/2602.03315)

### What Makes Memora Different

Memora differs from RAG pipelines, flat memory stores, and graph-based knowledge bases by introducing a **harmonic memory representation that separates what is stored from how it is organised and accessed**. Instead of indexing all memory content directly, Memora builds a scaffolding layer (using primary abstractions and cue anchors) that provides structure over the memory without constraining the underlying data.
 
At the core of this design is the representation of each memory entry, which consists of three components:
- **Memory value (not indexed)**: the full stored information, preserving fine-grained details without compression or loss
- **Primary abstraction (indexed)**: a one-to-one summary that captures what the memory is about and serves as the canonical unit for updates and aggregation
- **Cue anchors (indexed)**: multiple semantic entry points (e.g. ```<entity> + <key aspects>``` ) that connect related memories through a many-to-many structure

Crucially, only the primary abstractions and cue anchors are indexed, while the memory values themselves are not. This allows Memora to retain rich, high-fidelity information without introducing the fuzziness typically associated with indexing or embedding raw content. At the same time, the abstraction layer provides a structured way to organise and access memory, guiding retrieval through well-defined representations rather than relying on similarity over unstructured data.
 
As a result, Memora remains more flexible than graph-based approaches while still providing strong organisational structure. It preserves detail in the stored memory, while using lightweight abstractions to enable **more precise, controlled access** to that information.

## Key Features

- **Lightweight integration:**
Designed to plug into existing agent systems with minimal changes, without requiring a full redesign of memory handling.
- **Structured memory representation:**
Provides a clear separation between memory values and their abstractions, enabling consistent updates, deduplication, and organisation over time.
- **Shared memory space:**
Supports a unified memory layer that can be accessed across agents within the same environment, enabling coordination and reuse of knowledge.
- **Support for diverse memory types:**
Can represent different forms of memory (e.g. factual, episodic, procedural), depending on how memory values and abstractions are constructed.
- **Controlled access and isolation:**
Allows memory to be scoped and managed across agents or roles, supporting privacy and selective sharing.
- **Flexible storage backend:**
Compatible with different storage setups (local or remote), without tightly coupling the memory representation to a specific infrastructure.

## How Memora Works

Memora provides a complete memory lifecycle for AI agents:

1. **Memory Ingestion** — When an agent processes a conversation or document, Memora automatically extracts relevant facts, episodes, and procedural knowledge. It segments conversations into topical episodes and distills key information into structured memory entries.

2. **Intelligent Storage** — Memories are stored in a vector database (ChromaDB) with semantic embeddings. Memora deduplicates, merges, and updates existing memories to keep the store clean and current. An optional **cue index** maps high-level cues to memories for fast, structured lookups.

3. **Adaptive Retrieval** — At query time, Memora supports multiple retrieval strategies:
   - **Semantic** — Vector similarity search over the memory store
   - **Prompted** — An LLM-guided multi-step retrieval policy that iteratively refines the search
   - **Hybrid** — Combines semantic search with BM25/keyword matching for better recall
   - **GRPO** *(experimental)* — Reinforcement learning–based retrieval policy trained via Group Relative Policy Optimization

4. **Answer Generation** — Retrieved memories are formatted and injected into the LLM prompt to produce grounded, context-aware responses.

## Installation

**Prerequisites:** Python >= 3.10

```bash
# Install from source
git clone https://github.com/microsoft/Memora
cd Memora
pip install -e .
```

## Quick Start

See [`quickstart.py`](quickstart.py) for a complete runnable example.

```python
from memora.memora_client import MemoraClient

# Initialize with your config (see quickstart.py for config setup)
memory_client = MemoraClient(cfg=cfg, user_id="my_user")

# Add context to memory
memory_client.add("Alice is moving to Seattle for a new job.", type="doc")

# Query memory (semantic search)
results = memory_client.query("Where is Alice moving?", top_k=5)
for entry in results:
    print(f"{entry.index}: {entry.value}")

# Advanced query with prompted retrieval policy
results = memory_client.advance_query("Where is Alice moving?", query_type="prompt", top_k=5)
```

### Using Memora in an Agent

```python
from memora.memora_client import MemoraClient

class MyAgent:
    def __init__(self, cfg):
        self.memory_client = MemoraClient(cfg=cfg, user_id="agent_user")

    def generate_response(self, user_message):
        # Retrieve relevant memories
        memories = self.memory_client.query(user_message, top_k=5)

        # Your agent logic here
        response = ...

        # Store conversation in memory
        conversation = f"User: {user_message}\nAssistant: {response}"
        self.memory_client.add(conversation, type="doc")
        return response
```

## Running Benchmark Experiments

Memora includes experiment runners for two established benchmarks. Both use [Hydra](https://hydra.cc/) for configuration — all parameters can be overridden from the command line.

### LoCoMo

The [LoCoMo](https://arxiv.org/abs/2402.17753) benchmark evaluates long-conversation memory across single-hop, multi-hop, temporal, and open-domain question types.

```bash
cd app/locomo

# Run Memora on LoCoMo
python run_memora.py \
    llm.model="gpt-4.1-mini" \
    memory.memory_store="memora-cue" \
    memory.enable_cue_index=True \
    retrieval.strategy="prompt"
```

For full details, see [app/locomo/README.md](app/locomo/README.md).

### LongMemEval

The [LongMemEval](https://arxiv.org/abs/2410.10813) benchmark tests long-term memory capabilities across multiple question types.

```bash
cd app/longmemeval

# Run Memora on LongMemEval
python run_memora.py \
    llm.model="gpt-4.1-mini" \
    memory.memory_store="memora-semantic" \
    memory.enable_episodic_memory=True \
    retrieval.strategy="semantic"
```

For full details, see [app/longmemeval/README.md](app/longmemeval/README.md).

### GRPO Experiments *(experimental)*

Memora supports training a retrieval policy via Group Relative Policy Optimization (GRPO). This uses reinforcement learning to learn which memories to retrieve for a given query, replacing prompted LLM calls with a fine-tuned local model (e.g., Qwen 3B/7B with LoRA).

The GRPO pipeline consists of:
1. **Trajectory collection** — Sample retrieval trajectories using the current policy
2. **Trajectory scoring** — Score trajectories based on groundedness, redundancy, and cost
3. **Policy training** — Train the retrieval policy using GRPO with group-relative advantages

#### Training

```bash
# Train a retrieval policy with GRPO (requires GPU)
python -m memora.rl.grpo_trainer \
    --config app/locomo/conf/config.yaml \
    --output_dir ./grpo_output \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --num_train_epochs 5 \
    --batch_size 4 \
    --group_size 4 \
    --learning_rate 1e-5 \
    --checkpoint_every 10 \
    --temperature 1.0
```

#### Inference with trained policy

```python
from memora import MemoraClient

client = MemoraClient(cfg=cfg, user_id="user_123")

# Use GRPO-trained policy for retrieval
memories = client.advance_query(
    context="What did we discuss about the project?",
    query_type="grpo",
    checkpoint_path="./grpo_output/final",
    top_k=5,
)
```

#### Collecting trajectories separately

```bash
python -m memora.rl.collect_trajectories \
    --config app/locomo/conf/config.yaml \
    --data_path app/locomo/data/locomo10.json \
    --output trajectories.json
```

## Configuration

Both LoCoMo and LongMemEval experiments are configured via YAML files in their respective `conf/` directories. Key settings you'll need to configure:

| Setting | Description |
|---------|-------------|
| `openai.api_type` | `"azure"` or `"openai"` |
| `openai.managed_identity` | Azure Managed Identity client ID (for `api_type: "azure"`) |
| `openai.api_key` | OpenAI API key (for `api_type: "openai"`, or set `OPENAI_API_KEY` env var) |
| `openai.llm_api_base` | Azure OpenAI endpoint for chat completions (e.g., `https://<resource>.openai.azure.com/`) |
| `openai.embedding_api_base` | Azure OpenAI endpoint for embeddings (can be the same as `llm_api_base`) |
| `openai.embedding_api_version` | API version for embeddings (e.g., `2024-02-01`) |
| `openai.embedding_deployment_name` | Azure deployment name for the embedding model |
| `openai.embedding_model` | Model name for embeddings (e.g., `text-embedding-3-small`) |

**Azure setup example:**

```bash
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
export AZURE_MANAGED_IDENTITY_CLIENT_ID="<your-client-id>"
```

```yaml
openai:
  api_type: "azure"
  managed_identity: "${oc.env:AZURE_MANAGED_IDENTITY_CLIENT_ID}"
  llm_api_base: "${oc.env:AZURE_OPENAI_ENDPOINT}"
  embedding_api_base: "${oc.env:AZURE_OPENAI_ENDPOINT}"
  embedding_api_version: "2024-02-01"
  embedding_deployment_name: "text-embedding-3-small"
  embedding_model: "text-embedding-3-small"
```

**OpenAI setup example:**

```bash
export OPENAI_API_KEY="sk-..."
```

```yaml
openai:
  api_type: "openai"
  api_key: "${oc.env:OPENAI_API_KEY}"
  embedding_model: "text-embedding-3-small"
```

## Project Structure

```
Memora/
├── src/memora/          # Core Memora library
│   ├── core/            # Memory store, entry models, segmentation, cue indexing
│   ├── builder/         # Memory builders (chat, document)
│   ├── processors/      # Document processors (PDF, DOCX, Excel, Markdown, etc.)
│   ├── browser/         # Interactive memory store browser & viewer
│   ├── retriever/       # Retrieval strategies (semantic, prompted policy, local GRPO policy)
│   ├── rl/              # GRPO training (policy, trainer, trajectory collection & scoring)
│   ├── db_clients/      # Database backends (ChromaDB, Redis)
│   └── utils/           # Logging, LLM clients, helpers
├── app/
│   ├── locomo/          # LoCoMo benchmark experiments
│   └── longmemeval/     # LongMemEval benchmark experiments
├── quickstart.py        # Quick start example
└── requirements.txt     # Python dependencies
```

## Citation

If you find Memora useful, please cite our paper:

```bibtex
@misc{xia2026memoraharmonicmemoryrepresentation,
      title={Memora: A Harmonic Memory Representation Balancing Abstraction and Specificity}, 
      author={Menglin Xia and Xuchao Zhang and Shantanu Dixit and Paramaguru Harimurugan and Rujia Wang and Victor Ruhle and Robert Sim and Chetan Bansal and Saravan Rajmohan},
      year={2026},
      eprint={2602.03315},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.03315}, 
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Trademarks 

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft’s Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.
