# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import concurrent.futures
import json
import logging
import os
import threading
import pandas as pd
from collections import defaultdict

import hydra
from omegaconf import DictConfig

from memora.utils.log import configure_logging
from memobench import runtime
from metrics.llm_judge import build_judge_req, judge_score
from metrics.utils import calculate_bleu_scores, calculate_metrics
from tqdm import tqdm

logger = logging.getLogger(__name__)

def evaluate(cfg: DictConfig, input_file: str, output_file: str, system: str = None):
    """Judge every answer of a results file in ONE Batch API run, then save the results in the same format
    as before (one record per question with bleu_score / f1_score / llm_score) for generate_scores."""

    with open(input_file, "r") as f:
        data = json.load(f)

    pending = []  # (key, record, req)
    for k, v in data.items():
        for i, item in enumerate(v):
            gt_answer = str(item["answer"])
            pred_answer = str(item["response"])
            category = str(item["category"])
            question = str(item["question"])

            # Skip category 5
            if category == "5":
                continue

            # Local metrics first, so a failure here happens before anything is paid for
            metrics = calculate_metrics(pred_answer, gt_answer)
            bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)

            record = {
                "question": question,
                "answer": gt_answer,
                "response": pred_answer,
                "category": category,
                "bleu_score": bleu_scores["bleu1"],
                "f1_score": metrics["f1"],
                # Extract formatted memories if available
                "formatted_speaker_1_memories": item.get("formatted_speaker_1_memories", []),
                "formatted_speaker_2_memories": item.get("formatted_speaker_2_memories", []),
            }
            meta = {"system": system, "conv_id": item.get("conv_id", str(k)),
                    "question_id": item.get("question_id", f"{k}/{i}")}
            pending.append((k, record, build_judge_req(question, gt_answer, pred_answer, meta=meta)))

    batch_results = runtime.get_llm().run([req for _, _, req in pending], mode="batch")

    results = defaultdict(list)
    failures = []
    for k, record, req in tqdm(pending, desc="Collecting judge labels"):
        try:
            llm_score = judge_score(batch_results[req.key])
        except Exception as e:  # upstream raises on a judge failure; collect them all first, then raise
            failures.append((record["question"], repr(e)))
            continue
        results[k].append({**record, "llm_score": llm_score})

    if failures:
        for question, err in failures:
            logger.error(f"LLM judge failed for question {question!r}: {err}")
        raise RuntimeError(f"LLM judge failed for {len(failures)}/{len(pending)} answers (successful labels are "
                           f"cached; re-running resubmits only the failures)")

    # Save results to JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Results saved to {output_file}")


def generate_scores(metrics_file: str, score_file: str):
    # Load the evaluation metrics data
    with open(metrics_file, "r") as f:
        data = json.load(f)

    # Flatten the data into a list of question items
    all_items = []
    for key in data:
        all_items.extend(data[key])

    # Convert to DataFrame
    df = pd.DataFrame(all_items)

    # Convert category to numeric type
    df["category"] = pd.to_numeric(df["category"])

    # Calculate mean scores by category
    result = (
        df.groupby("category")
        .agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"})
        .round(4)
    )

    # Add count of questions per category
    result["count"] = df.groupby("category").size()

    # Print the results
    logger.info(f"\nMean Scores Per Category:\n{result}\n")

    # Calculate overall means
    overall_means = df.agg(
        {"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}
    ).round(4)

    logger.info(f"\nOverall Mean Scores:\n{overall_means}\n")

    # Prepare results for saving to file
    scores_data = {
        "mean_scores_per_category": result.to_dict(),
        "overall_mean_scores": overall_means.to_dict(),
        "summary": {
            "total_questions": len(df),
            "categories_evaluated": sorted(df["category"].unique().tolist()),
            "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        },
    }

    # Save results to score file
    with open(score_file, "w") as f:
        json.dump(scores_data, f, indent=4)

    logger.info(f"\nScores saved to {score_file}")

    return scores_data


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def run(cfg: DictConfig):

    # configure logging
    configure_logging()

    input_file = os.path.join(cfg.general.output_path, cfg.eval.result_file)
    output_file = os.path.join(cfg.general.output_path, cfg.eval.metrics_file)
    score_file = os.path.join(
        cfg.general.output_path, cfg.eval.get("score_file", "scores.json")
    )

    # evaluate
    evaluate(cfg, input_file, output_file)

    # generate scores
    scores_data = generate_scores(output_file, score_file)

    return scores_data

if __name__ == "__main__":
    run()
