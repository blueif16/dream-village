#!/bin/bash

# ============================================================
# LoCoMo Experiment Runner
# ============================================================
# Usage: bash run.sh
# Override any Hydra config parameter via command line args.
# See conf/config.yaml for all available parameters.
# ============================================================

# --- Memora Experiments ---

# Quick debug run (small subset, force rebuild)
# python run_memora.py \
#     llm.model="gpt-4.1-mini" \
#     memory.memory_store="memora-debug" \
#     general.debug=True \
#     memory.force_rebuild=True

# Subset evaluation with semantic retrieval
# python run_memora.py \
#     llm.model="gpt-4.1-mini" \
#     memory.memory_store="memora-subset1" \
#     eval.subset_idx=1 \
#     retrieval.strategy="semantic"

# Full dataset with prompted retrieval and cue index
# python run_memora.py \
#     llm.model="gpt-4.1-mini" \
#     memory.memory_store="memora-full-prompt" \
#     eval.subset_idx=-1 \
#     memory.enable_cue_index=True \
#     retrieval.strategy="prompt"

# Episodic memory with segments as episodes
# python run_memora.py \
#     llm.model="gpt-4.1-mini" \
#     memory.memory_store="memora-episodic" \
#     memory.enable_episodic_memory=True \
#     memory.use_segments_as_episodic=True \
#     retrieval.strategy="semantic"

# --- Baseline Experiments ---

# Mem0 baseline
# python run_experiments.py \
#     memory.type=mem0 \
#     openai.model="gpt-4.1-mini" \
#     memory.memory_store="mem0-4.1-all" \
#     eval.subset_idx=-1

# RAG baseline
# python run_experiments.py \
#     memory.type=rag \
#     openai.model="gpt-4.1-mini"

# Full context baseline (no memory, uses entire conversation)
# python run_experiments.py \
#     memory.type=full_context \
#     openai.model="gpt-4.1-mini"

# --- Default run ---
python run_memora.py \
    llm.model="gpt-4.1-mini" \
    memory.memory_store="memora-cue" \
    eval.subset_idx=1 \
    memory.enable_cue_index=True \
    retrieval.strategy="prompt"
