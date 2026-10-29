# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Locomo Experiment Runner

This module provides a comprehensive experimental framework for evaluating different
memory techniques on the Locomo dataset. It supports multiple memory backends including
Memora, Mem0, and RAG-based approaches.
"""

# memo-bench: Memora (vendor/Memora/src) and the shared LLM layer (memobench, at the repo root) are not
# installed packages; make both importable when launched from this directory.
import sys
from pathlib import Path

for _path in (Path(__file__).resolve().parents[4], Path(__file__).resolve().parents[2] / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from datetime import datetime
import logging
import os
# Disable telemetry to prevent hanging background threads
import os
os.environ["MEM0_TELEMETRY"] = "False"   # Disable mem0 telemetry

from datetime import datetime
import time
import hydra
from omegaconf import DictConfig

from evals import evaluate, generate_scores
from memora.utils.log import configure_logging

# Memory system implementations (the mem0 baseline is imported inside _run_mem0_experiment: mem0 is not installed)
from providers.memora.search import MemoraSearch
from providers.memora.add import MemoraADD

from providers.rag import RAGManager
from providers.full_context import FullContextManager

logger = logging.getLogger(__name__)

from utils import format_duration, init_mem0_client, measure_execution_time, memora_system_name


def _run_memora_experiment(cfg: DictConfig, data_file: str):
    """
    Execute Memora memory technique experiment.

    This function handles the complete Memora evaluation pipeline:
    1. Building memory from conversations
    2. Searching and evaluating memory performance

    Args:
        cfg: Configuration object containing experiment parameters
        data_file: Path to the dataset file
    """
    logger.info("\n==== Building Memora Memory ====")
    memory_manager = MemoraADD(cfg, data_path=data_file)

    # Build memory with timing
    _, build_duration = measure_execution_time(memory_manager.process_all_conversations)
    logger.info(f"Memory processing completed in {format_duration(build_duration)}")

    logger.info("\n==== Searching Memora Memory and Evaluating ====")

    # Create output folder based on method name and timestamp
    method = "memora"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.general.results_path, f"{method}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n==== Output Directory: {output_dir} ====")

    # Prepare output directory and file
    output_file = os.path.join(cfg.general.results_path, "memora_output.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Initialize search and evaluation
    memory_searcher = MemoraSearch(cfg, output_file, cfg.memory.top_k)

    # Run inference with timing
    _, eval_duration = measure_execution_time(
        memory_searcher.process_data_file, data_file
    )
    logger.info(f"Memory inference completed in {format_duration(eval_duration)}")

    # Run evaluation
    eval_file = os.path.join(cfg.general.output_path, "memora_eval.json")
    evaluate(cfg, output_file, eval_file, system=memora_system_name(cfg, "semantic"))

    # generate scores
    score_file = os.path.join(cfg.general.output_path, "memora_scores.json")
    generate_scores(eval_file, score_file)


def _run_mem0_experiment(cfg: DictConfig, data_file: str, output_file: str):
    """
    Execute Mem0 memory technique experiment.

    Args:
        cfg: Configuration object containing experiment parameters
        data_file: Path to the dataset file
    """
    from mem0 import Memory
    from providers.memzero.add import MemoryADD
    from providers.memzero.search import MemorySearch

    # Build memory
    method = "mem0"

    # Create output folder based on method name and timestamp
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now().strftime("%Y%m%d")
    subset = cfg.eval.subset_idx if cfg.eval.subset_idx >= 0 else "full"
    output_dir = os.path.join(cfg.general.results_path, f"{method}_{subset}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n==== Output Directory: {output_dir} ====")

    output_file = os.path.join(output_dir, f"{method}_output.json")

    build_memory = (
        not os.path.exists(cfg.memory.persist_path) or cfg.memory.force_rebuild
    )

    memory: Memory = init_mem0_client(cfg)

    if build_memory:
        # Build memory with timing
        print(f"\n==== Building Memora Memory in {cfg.memory.persist_path} ====")
        memory_manager = MemoryADD(
            cfg, memory, data_path=data_file, is_graph=cfg.memory.is_graph
        )

        tic = time.time()
        for idx, item in enumerate(memory_manager.data):
            memory_manager.process_conversation(item, idx)
        build_duration = time.time() - tic
        print(f"Mem0 memory processing completed in {build_duration:.2f} seconds")

    print("\n==== Searching Mem0 Memory and Evaluating ====")
    # Initialize search and run evaluation
    memory_searcher = MemorySearch(
        cfg,
        memory,
        output_file,
        cfg.memory.top_k,
        cfg.memory.filter_memories,
        cfg.memory.is_graph,
    )
    _, eval_duration = measure_execution_time(
        memory_searcher.process_data_file, data_file
    )
    print(f"Mem0 evaluation completed in {format_duration(eval_duration)}")

    # # Run evaluation
    eval_file = os.path.join(output_dir, f"{method}_eval.json")
    evaluate(cfg, output_file, eval_file)

    # generate scores
    score_file = os.path.join(output_dir, f"{method}_scores.json")
    generate_scores(eval_file, score_file)


def _run_rag_experiment(cfg: DictConfig, rag_data_file: str):
    """
    Execute RAG (Retrieval-Augmented Generation) experiment.

    Args:
        cfg: Configuration object containing experiment parameters
        rag_data_file: Path to the RAG-specific dataset file
    """
    print("\n==== Running RAG Experiment ====")

    # Create output folder based on method name and timestamp
    method = "rag"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.general.results_path, f"{method}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n==== Output Directory: {output_dir} ====")

    # Prepare output file
    output_file_path = os.path.join(
        output_dir,
        f"{method}_output.json",
    )

    # Initialize RAG manager and run experiment
    rag_manager = RAGManager(
        cfg=cfg,
        data_path=rag_data_file,
        chunk_size=cfg.memory.chunk_size,
        k=cfg.memory.num_chunks,
    )
    _, duration = measure_execution_time(
        rag_manager.process_all_conversations, output_file_path
    )
    print(f"RAG generation completed in {format_duration(duration)}")

    # Run evaluation
    print("\n==== Running Evaluation ====")
    eval_file = os.path.join(output_dir, f"{method}_eval.json")
    evaluate(cfg, output_file_path, eval_file, system="rag")

    # Generate scores
    print("\n==== Generating Scores ====")
    score_file = os.path.join(output_dir, f"{method}_scores.json")
    generate_scores(eval_file, score_file)


def _run_full_context_experiment(cfg: DictConfig, data_file: str):
    """
    Execute Full Context baseline experiment.

    This baseline uses the entire conversation history as context without
    any memory extraction or retrieval - just passes everything to the LLM.

    Args:
        cfg: Configuration object containing experiment parameters
        data_file: Path to the dataset file
    """
    print("\n==== Running Full Context Baseline ====")

    # Create output folder based on method name and timestamp
    method = "full_context"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.general.results_path, f"{method}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n==== Output Directory: {output_dir} ====")

    # Prepare output file
    output_file_path = os.path.join(output_dir, f"{method}_output.json")

    # Initialize Full Context manager and run experiment
    full_context_manager = FullContextManager(cfg=cfg, data_path=data_file)
    _, duration = measure_execution_time(
        full_context_manager.process_all_conversations, output_file_path
    )
    print(f"Full context generation completed in {format_duration(duration)}")

    # Run evaluation
    print("\n==== Running Evaluation ====")
    eval_file = os.path.join(output_dir, f"{method}_eval.json")
    evaluate(cfg, output_file_path, eval_file, system="full-context")

    # Generate scores
    print("\n==== Generating Scores ====")
    score_file = os.path.join(output_dir, f"{method}_scores.json")
    generate_scores(eval_file, score_file)



@hydra.main(version_base=None, config_path="./conf", config_name="config")
def run(cfg: DictConfig):
    """
    Main experiment runner function.

    This function orchestrates the execution of different memory technique experiments
    based on the provided configuration. It supports multiple backends and provides
    comprehensive timing and logging.

    Args:
        cfg: Hydra configuration object containing all experiment parameters

    Raises:
        ValueError: If an unsupported memory technique type is specified
    """
    # Configure logging system
    configure_logging()
    # Note: Uncomment below to suppress mem0 verbose logging
    # logging.getLogger("mem0.memory.main").setLevel(logging.WARNING)

    # Display experiment configuration
    print("=" * 60)
    print("LOCOMO MEMORY EXPERIMENT RUNNER")
    print("=" * 60)
    print(f"Memory Technique: {cfg.memory.type}")
    print(f"Dataset Path: {cfg.general.data_path}")
    print(f"Output Path: {cfg.general.output_path}")
    print("=" * 60)

    # Prepare data file paths
    data_file = os.path.join(cfg.general.data_path, "locomo10.json")
    rag_data_file = os.path.join(cfg.general.data_path, "locomo10_rag.json")

    # Prepare output directory and file
    output_file = os.path.join(
        cfg.general.output_path, f"{cfg.memory.type}_output.json"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Execute experiment based on selected memory technique
    if cfg.memory.type == "memora":
        _run_memora_experiment(cfg, data_file)

    elif cfg.memory.type == "mem0":
        _run_mem0_experiment(cfg, data_file, output_file)

    elif cfg.memory.type == "rag":
        _run_rag_experiment(cfg, data_file)

    elif cfg.memory.type == "full_context":
        _run_full_context_experiment(cfg, data_file)

    else:
        raise ValueError(
            f"Invalid memory technique type: '{cfg.memory.type}'. "
            f"Supported types: ['memora', 'mem0', 'rag', 'full_context']"
        )


if __name__ == "__main__":
    run()
