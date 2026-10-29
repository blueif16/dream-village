"""
LongMemEval baseline runner (memo-bench): full context and RAG.

    uv run python run_experiments.py memory.type=full_context eval.subset_idx="1:100"
    uv run python run_experiments.py memory.type=rag eval.subset_idx="1:100"

Every answer goes through the one fixed answerer (answerer.py, the template Memora answers with) in ONE Batch API
run; the answers are then judged by evals.evaluate (LongMemEval's evaluate_qa.py prompt, one batch), which writes
<method>_eval.json and <method>_scores.json exactly as for the Memora runner (run_memora.py).
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
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

from evals import evaluate
from memora.utils.log import configure_logging
from providers.full_context import FullContextManager
from providers.full_context import SYSTEM as FULL_CONTEXT_SYSTEM
from providers.rag import RAGManager
from providers.rag import SYSTEM as RAG_SYSTEM

logger = logging.getLogger(__name__)

SYSTEMS = {"full_context": FULL_CONTEXT_SYSTEM, "rag": RAG_SYSTEM}


def run_baseline(cfg: DictConfig) -> str:
    """Answer (one batch), judge (one batch), score. Returns the output directory."""
    method = cfg.memory.type
    if method not in SYSTEMS:
        raise ValueError(f"Invalid memory technique type: '{method}'. Supported types: {sorted(SYSTEMS)} "
                         f"(Memora runs through run_memora.py)")

    data_file = os.path.join(cfg.general.data_path, "longmemeval_s_cleaned.json")

    # Output directory: general.output_dir when given (as in run_memora.py), else a timestamped folder
    output_dir = cfg.general.get("output_dir", None)
    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(cfg.general.results_path, f"{method}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"\n==== Output Directory: {output_dir} ====")

    with open(os.path.join(output_dir, "parameters.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    output_file = os.path.join(output_dir, f"{method}_output.json")
    if method == "full_context":
        manager = FullContextManager(cfg=cfg, data_path=data_file)
    else:
        manager = RAGManager(cfg=cfg, data_path=data_file, chunk_size=cfg.memory.chunk_size, k=cfg.memory.num_chunks)
    manager.process_all_questions(output_file)

    logger.info("\n==== Start evaluation ====")
    eval_file = os.path.join(output_dir, f"{method}_eval.json")
    evaluate(cfg, output_file, eval_file, use_question_type_eval=cfg.eval.use_question_type_eval,
             system=SYSTEMS[method])
    return output_dir


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def run(cfg: DictConfig):
    configure_logging(cfg.general.get("log_level", "INFO"))
    logger.info(f"\n===== LongMemEval baseline: {cfg.memory.type}, subset_idx={cfg.eval.subset_idx} =====")
    run_baseline(cfg)


if __name__ == "__main__":
    run()
