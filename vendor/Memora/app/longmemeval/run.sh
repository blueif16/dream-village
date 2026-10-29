#!/bin/bash

# ============================================================
# LongMemEval Experiment Runner
# ============================================================
# Usage: bash run.sh
# Override any Hydra config parameter via command line args.
# See conf/config.yaml for all available parameters.
# ============================================================

# Quick debug run (small subset)
# python run_memora.py general.debug=True

# Semantic retrieval with episodic memory
# python run_memora.py \
#     llm.model="gpt-4.1-mini" \
#     memory.memory_store="memora-semantic" \
#     memory.enable_episodic_memory=True \
#     retrieval.strategy="semantic"

# Prompted retrieval with cue index
# python run_memora.py \
#     llm.model="gpt-4.1-mini" \
#     memory.memory_store="memora-cue-prompt" \
#     memory.enable_cue_index=True \
#     retrieval.strategy="prompt"

# Full dataset evaluation
# python run_memora.py \
#     llm.model="gpt-4.1-mini" \
#     memory.memory_store="memora-full" \
#     eval.subset_idx=-1

# Run a specific range of questions (e.g., questions 1-100)
# python run_memora.py \
#     eval.subset_idx="1:100" \
#     memory.memory_store="memora-split-0"

# Parallel search (within a single process)
# python run_memora.py \
#     eval.subset_idx="1:100" \
#     eval.parallel_search=true \
#     eval.max_workers=6 \
#     memory.memory_store="memora-parallel"

# Skip existing memories (retry failed questions only)
# python run_memora.py \
#     memory.skip_existing=true \
#     memory.memory_store="memora-retry"

# Use evermemos answer template (chain-of-thought)
# python run_memora.py \
#     eval.answer_template="evermemos" \
#     memory.memory_store="memora-evermemos"

# For multi-process parallelism across 5 splits, use:
#   ./run_parallel_splits.sh

# Default run (sequential)
python run_memora.py \
    llm.model="gpt-4.1-mini" \
    memory.memory_store="memora-default" \
    memory.enable_episodic_memory=True \
    retrieval.strategy="semantic"