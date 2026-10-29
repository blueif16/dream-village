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

import logging
import os
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Memory system implementations
from providers.memora.add import MemoraADD
from providers.memora.search import MemoraSearch
from evals import evaluate
from memora.utils.log import configure_logging
from utils import memora_system_name

logger = logging.getLogger(__name__)


def run_memora_experiment(
    cfg: DictConfig, debug: bool, subset_idx, memora_store: str, retrieval_strategy: str = "semantic",
):
    """
    Main experiment runner function.

    """
    # Configure logging system
    log_level = cfg.general.get("log_level", "INFO")
    configure_logging(log_level)

    logger.info("\n===== Memory Config =====\n" + OmegaConf.to_yaml(cfg.memory))
    logger.info(f"\n===== Retrieval Strategy: {retrieval_strategy} =====")
    # set debug mode

    cfg.general.debug = debug
    cfg.eval.subset_idx = subset_idx
    cfg.memory.memory_store = memora_store

    # Execute experiment
    data_file = os.path.join(cfg.general.data_path, "longmemeval_s_cleaned.json")

    memory_store_exists = os.path.exists(cfg.memory.persist_path)

    # Determine whether to build memory and whether to skip existing
    skip_existing = False
    if cfg.memory.force_rebuild:
        build_memory = True
        print(f"\n==== Force Rebuilding Memora Memory (rebuild all) in {cfg.memory.persist_path} ====")
    elif not memory_store_exists:
        build_memory = True
        print(f"\n==== Building Memora Memory (new store) in {cfg.memory.persist_path} ====")
    elif cfg.memory.get("skip_existing", False):
        build_memory = True
        skip_existing = True
        print(f"\n==== Building Memora Memory (skip existing) in {cfg.memory.persist_path} ====")
    else:
        build_memory = False
        print(f"\n==== Using existing Memora Memory in {cfg.memory.persist_path} ====")

    if build_memory:
        try:
            memory_manager = MemoraADD(cfg, data_path=data_file, skip_existing_memories=skip_existing)

            # parallelizing across questions 
            memory_manager.process_all_conversations()

        except Exception as e:
            logger.error(f"Error building memory: {str(e)}")
            raise e

    # logging completion
    logger.info(f"\n==== Memory building completed ====")

    # Prepare output directory and file
    method = f"memora_{retrieval_strategy}"

    # Check if output_dir is explicitly provided via config override (for parallel splits)
    if hasattr(cfg.general, 'output_dir') and cfg.general.output_dir:
        output_dir = cfg.general.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"\n==== Output Directory: {output_dir} ====")
    else:
        # Default: create timestamped output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(cfg.general.results_path, f"{method}_{cfg.memory.memory_store}_{timestamp}")  # store in the name: parallel runs never share a dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"\n==== Output Directory: {output_dir} ====")

    # Save experiment parameters
    params_file = os.path.join(output_dir, "parameters.yaml")
    with open(params_file, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Experiment parameters saved to: {params_file}")

    output_file = os.path.join(output_dir, f"{method}_output.json")

    # Clear existing output file to avoid appending to old results
    if os.path.exists(output_file):
        logger.info(f"Clearing existing output file: {output_file}")
        os.remove(output_file)

    # Initialize search and evaluation
    try:
        memory_searcher = MemoraSearch(cfg, output_file, cfg.memory.top_k, retrieval_strategy=retrieval_strategy)
        memory_searcher.process_data_file(data_file)
    except Exception as e:
        logger.error(f"Error during memory search and evaluation: {str(e)}")
        raise e

    # Run evaluation
    logger.info(f"\n==== Start evaluation ====")
    eval_file = os.path.join(output_dir, f"{method}_eval.json")
    evaluate(cfg, output_file, eval_file, use_question_type_eval=cfg.eval.use_question_type_eval,
             system=memora_system_name(cfg, retrieval_strategy))


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

    run_memora_experiment(
        cfg,
        debug=cfg.general.debug,
        subset_idx=cfg.eval.subset_idx,
        memora_store=cfg.memory.memory_store,
        retrieval_strategy=cfg.retrieval.strategy,
    )


if __name__ == "__main__":
    run()
