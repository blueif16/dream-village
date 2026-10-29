# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
LoCoMo Memora Experiment Runner

Evaluates Memora on the LoCoMo dataset with configurable retrieval strategies.
Uses Hydra for configuration management - all parameters can be overridden via CLI.

Usage:
    python run_memora.py llm.model="gpt-4.1-mini" memory.memory_store="memora-cue" retrieval.strategy="prompt"
"""

# memo-bench: Memora (vendor/Memora/src) and the shared LLM layer (memobench, at the repo root) are not
# installed packages; make both importable when launched from this directory.
import sys
from pathlib import Path

for _path in (Path(__file__).resolve().parents[4], Path(__file__).resolve().parents[2] / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import logging
import os
import time
import json
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

from providers.memora.add import MemoraADD
from providers.memora.search import MemoraSearch
from evals import evaluate, generate_scores
from memora.utils.log import configure_logging
from utils import memora_system_name

logger = logging.getLogger(__name__)


def run_memora_experiment(
    cfg: DictConfig, debug: bool, subset_idx: int, memora_store: str, retrieval_strategy: str = "semantic",
):
    """
    Run a complete Memora experiment: build memory, search, and evaluate.

    Args:
        cfg: Hydra configuration object
        debug: Whether to use debug mode (smaller dataset)
        subset_idx: Dataset subset index (-1 for full dataset)
        memora_store: Name of the memory store
        retrieval_strategy: Retrieval strategy ("semantic", "prompt", or "grpo")
    """
    log_level = cfg.general.get("log_level", "INFO")
    configure_logging(log_level)

    logger.info("\n===== Memory Config =====\n" + OmegaConf.to_yaml(cfg.memory))
    logger.info(f"\n===== Retrieval Strategy: {retrieval_strategy} =====")

    cfg.general.debug = debug
    cfg.eval.subset_idx = subset_idx
    cfg.memory.memory_store = memora_store

    # Prepare paths
    data_file = os.path.join(cfg.general.data_path, "locomo10.json")
    method = f"memora_{retrieval_strategy}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.general.results_path, f"{method}_{cfg.memory.memory_store}_{timestamp}")  # store in the name: parallel runs never share a dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"\n==== Output Directory: {output_dir} ====")

    # Step 1: Build memory (if needed)
    build_memory = (
        not os.path.exists(cfg.memory.persist_path) or cfg.memory.force_rebuild
    )

    if build_memory:
        logger.info(f"\n==== Building Memora Memory in {cfg.memory.persist_path} ====")
        build_start = time.time()

        try:
            memory_manager = MemoraADD(cfg, data_path=data_file)
            memory_manager.process_all_conversations()
        except Exception as e:
            logger.error(f"Error building memory: {str(e)}")
            raise e

        build_duration = time.time() - build_start
        build_duration_str = time.strftime("%H:%M:%S", time.gmtime(build_duration))
        logger.info(f"Memory building completed in {build_duration_str}")

        # Save build timing
        timing_data = {
            "start_time": datetime.fromtimestamp(build_start).strftime('%Y-%m-%d %H:%M:%S'),
            "end_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "duration": build_duration_str,
            "memory_store": cfg.memory.memory_store,
        }
        with open(os.path.join(output_dir, "build_timing.json"), 'w') as f:
            json.dump(timing_data, f, indent=2)

        # Save build stats
        with open(os.path.join(output_dir, "build_stats.json"), 'w') as f:
            json.dump(memory_manager.build_log, f, indent=2)

    # Save experiment parameters
    params_file = os.path.join(output_dir, "parameters.yaml")
    with open(params_file, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Experiment parameters saved to: {params_file}")

    # Step 2: Search and answer questions
    output_file = os.path.join(output_dir, f"{method}_output.json")

    try:
        memory_searcher = MemoraSearch(cfg, output_file, cfg.memory.top_k, retrieval_strategy=retrieval_strategy)
        memory_searcher.process_data_file(data_file)
    except Exception as e:
        logger.error(f"Error during memory search and evaluation: {str(e)}")
        raise e

    # Step 3: Evaluate results
    logger.info(f"\n==== Start evaluation ====")
    eval_file = os.path.join(output_dir, f"{method}_eval.json")
    evaluate(cfg, output_file, eval_file, system=memora_system_name(cfg, retrieval_strategy))

    # Step 4: Generate score table
    logger.info(f"\n==== Start generating score table ====")
    score_file = os.path.join(output_dir, f"{method}_scores.json")
    generate_scores(eval_file, score_file)


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def run(cfg: DictConfig):
    """Entry point for Hydra-based experiment execution."""
    run_memora_experiment(
        cfg,
        debug=cfg.general.debug,
        subset_idx=cfg.eval.subset_idx,
        memora_store=cfg.memory.memory_store,
        retrieval_strategy=cfg.retrieval.strategy,
    )


if __name__ == "__main__":
    run()
