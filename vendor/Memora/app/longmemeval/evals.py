# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import os
import pandas as pd
from typing import Dict, List, Any, Literal, Optional

import anthropic
import hydra
from omegaconf import DictConfig
from pydantic import BaseModel

from memora.utils.log import configure_logging
from memobench import runtime
from memobench.context import get_meta
from memobench.llm import Req
from metrics.anscheck import get_anscheck_prompt
from metrics.utils import calculate_bleu_scores, calculate_metrics
from tqdm import tqdm

logger = logging.getLogger(__name__)

JUDGE_MODEL = runtime.MODEL
JUDGE_MAX_TOKENS = 2048

# Repo's question type-specific prompts (Nemori/Zep copies), kept for reference only: the judge is LongMemEval's
# own evaluate_qa.py prompt, verbatim (metrics/anscheck.py), including its abstention variant for `_abs` ids.
TEMPORAL_REASONING_PROMPT = """
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

<QUESTION>
B: {question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
A: {response}
</RESPONSE>

Please answer 'yes' or 'no':"""

KNOWLEDGE_UPDATE_PROMPT = """
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

<QUESTION>
B: {question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
A: {response}
</RESPONSE>

Please answer 'yes' or 'no':"""

SINGLE_SESSION_PREFERENCE_PROMPT = """
I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

<QUESTION>
B: {question}
</QUESTION>
<RUBRIC>
{gold_answer}
</RUBRIC>
<RESPONSE>
A: {response}
</RESPONSE>

Please answer 'yes' or 'no':"""

DEFAULT_PROMPT = """
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

<QUESTION>
B: {question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
A: {response}
</RESPONSE>

Please answer 'yes' or 'no':"""


class JudgeLabel(BaseModel):
    """The prompt's own output contract ("Answer yes or no only."), enforced via structured outputs."""

    label: Literal["yes", "no"]


def build_judge_req(entry: Dict, meta: Optional[Dict] = None) -> Req:
    """Upstream get_anscheck_prompt(question_type, question, answer, response, abstention='_abs' in question_id)."""
    question_id = entry.get("question_id", "")
    prompt = get_anscheck_prompt(entry.get("question_type", ""), entry["question"], entry["answer"],
                                 entry["generated_answer"], abstention="_abs" in question_id)
    params = {
        "model": JUDGE_MODEL,
        "max_tokens": JUDGE_MAX_TOKENS,
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": anthropic.transform_schema(JudgeLabel)},
        },
        "messages": [{"role": "user", "content": prompt}],
    }
    return Req(params=params, meta={**get_meta(), **(meta or {}), "stage": "judge"})


def judge_score(result) -> float:
    """1.0 if the label is yes else 0.0 (upstream: 'yes' in the reply). Raises on an errored/unparseable reply."""
    if result.error:
        raise RuntimeError(f"judge request failed: {result.error}")
    return 1.0 if JudgeLabel.model_validate_json(result.text).label == "yes" else 0.0


def _load_input_data(input_file: str) -> List[Dict]:
    """
    Load input data from JSON dict (sequential mode) or JSONL (parallel mode).
    Returns a flat list of entry dicts.
    """
    with open(input_file, "r") as f:
        content = f.read().strip()

    if not content:
        return []

    # Try standard JSON first (sequential mode writes a dict)
    try:
        json_data = json.loads(content)
        if isinstance(json_data, dict):
            return [v for v in json_data.values() if isinstance(v, dict)]
        if isinstance(json_data, list):
            return json_data
    except json.JSONDecodeError:
        pass

    # Fall back to JSONL (parallel mode writes one JSON object per line)
    data = []
    for line in content.split("\n"):
        line = line.strip()
        if line:
            data.append(json.loads(line))
    return data


def evaluate(cfg: DictConfig, input_file: str, output_file: str,
             use_question_type_eval: bool = False, score_file: str = None, system: str = None):
    """
    Judge every answer in ONE Batch API run with LongMemEval's evaluate_qa.py prompt, save the results and scores.

    Args:
        cfg: Configuration object
        input_file: Path to input results file
        output_file: Path to output evaluation file
        use_question_type_eval: ignored; the upstream judge is always question-type-specific
        score_file: Path to output score file (optional, auto-generated if not provided)
        system: system name for the request log (e.g. memora-s)
    """

    # Auto-generate score file path if not provided
    if score_file is None:
        output_dir = os.path.dirname(output_file)
        output_basename = os.path.basename(output_file)
        if output_basename.endswith('_eval.json'):
            score_filename = output_basename.replace('_eval.json', '_scores.json')
        else:
            name, ext = os.path.splitext(output_basename)
            score_filename = f"{name}_scores{ext}"
        score_file = os.path.join(output_dir, score_filename)

    data = _load_input_data(input_file)

    if not data:
        logger.warning(f"No data loaded from {input_file}")
        return None

    if not use_question_type_eval:
        logger.warning("use_question_type_eval=false is ignored: judging with LongMemEval's evaluate_qa.py prompt")

    pending = []  # (record, req)
    for entry in data:
        question = str(entry["question"])
        gt_answer = entry["answer"]
        generated_answer = entry["generated_answer"]
        question_type = entry.get("question_type", "")
        question_id = entry.get("question_id", "")

        # Local metrics first, so a failure here happens before anything is paid for
        metrics = calculate_metrics(generated_answer, gt_answer)
        bleu_scores = calculate_bleu_scores(generated_answer, gt_answer)

        record = {
            "question_id": question_id,
            "question": question,
            "answer": gt_answer,
            "generated_answer": generated_answer,
            "question_type": question_type,
            "bleu_score": bleu_scores["bleu1"],
            "f1_score": metrics["f1"],
        }
        meta = {"system": system, "conv_id": question_id, "question_id": question_id}
        pending.append((record, build_judge_req(entry, meta=meta)))

    batch_results = runtime.get_llm().run([req for _, req in pending], mode="batch")

    results = []
    failures = []
    for record, req in tqdm(pending, desc="Collecting judge labels"):
        try:
            results.append({**record, "llm_score": judge_score(batch_results[req.key])})
        except Exception as e:
            failures.append((record["question_id"], repr(e)))

    if failures:
        for question_id, err in failures:
            logger.error(f"LLM judge failed for question {question_id}: {err}")
        raise RuntimeError(f"LLM judge failed for {len(failures)}/{len(pending)} answers (successful labels are "
                           f"cached; re-running resubmits only the failures)")

    # Save results to JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Results saved to {output_file}")

    # Calculate and print aggregated scores
    scores_data = generate_scores(data=results, score_file=score_file)
    return scores_data


def generate_scores(metrics_file: str = None, score_file: str = None, data: List[Dict] = None):
    """
    Generate evaluation scores from metrics file or data.
    
    Args:
        metrics_file: Path to metrics file (optional if data is provided)
        score_file: Path to output score file (optional)
        data: List of evaluation results (optional if metrics_file is provided)
    """
    # Load the evaluation metrics data
    if data is None:
        with open(metrics_file, "r") as f:
            json_data = json.load(f)

        # Handle both list and dict formats
        if isinstance(json_data, dict):
            all_items = []
            for key in json_data:
                if isinstance(json_data[key], list):
                    all_items.extend(json_data[key])
                else:
                    all_items.append(json_data[key])
            data = all_items
        elif isinstance(json_data, list):
            data = json_data

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Use question_type for grouping (LongMemEval format)
    group_by_col = "question_type"

    # Calculate mean scores by question type
    result = (
        df.groupby(group_by_col)
        .agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"})
        .round(4)
    )

    # Add count of questions per type
    result["count"] = df.groupby(group_by_col).size()

    # Print the results
    logger.info(f"\n{'='*60}")
    logger.info(f"Mean Scores Per Question Type:")
    logger.info(f"{'='*60}")
    for qtype in result.index:
        stats = result.loc[qtype]
        logger.info(f"{qtype:30s} | Count: {stats['count']:4.0f} | "
                   f"BLEU: {stats['bleu_score']:.4f} | "
                   f"F1: {stats['f1_score']:.4f} | "
                   f"LLM: {stats['llm_score']:.4f}")

    # Calculate overall means
    overall_means = df.agg(
        {"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}
    ).round(4)

    logger.info(f"\n{'='*60}")
    logger.info(f"Overall Mean Scores:")
    logger.info(f"{'='*60}")
    logger.info(f"BLEU Score:  {overall_means['bleu_score']:.4f}")
    logger.info(f"F1 Score:    {overall_means['f1_score']:.4f}")
    logger.info(f"LLM Score:   {overall_means['llm_score']:.4f}")
    logger.info(f"Total Questions: {len(df)}")

    # Abstention questions (`_abs` ids, judged with the abstention prompt) also count in their question type;
    # LongMemEval reports their accuracy separately as well.
    abstention = df[df["question_id"].astype(str).str.contains("_abs")] if "question_id" in df else df.iloc[0:0]
    abstention_score = round(float(abstention["llm_score"].mean()), 4) if len(abstention) else None
    logger.info(f"Abstention:  {abstention_score} ({len(abstention)} questions)")
    logger.info(f"{'='*60}\n")

    # Prepare results for saving to file
    scores_data = {
        "mean_scores_per_question_type": result.to_dict(),
        "overall_mean_scores": overall_means.to_dict(),
        "abstention": {"count": len(abstention), "llm_score": abstention_score},
        "summary": {
            "total_questions": len(df),
            "question_types_evaluated": sorted(df[group_by_col].unique().tolist()),
            "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        },
    }

    # Save results to score file
    if score_file:
        with open(score_file, "w") as f:
            json.dump(scores_data, f, indent=4)
        logger.info(f"Scores saved to {score_file}")

    return scores_data


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def run(cfg: DictConfig):
    """
    Main evaluation function.
    
    Args:
        cfg: Hydra configuration object
    """

    # configure logging
    configure_logging()

    input_file = os.path.join(cfg.general.output_path, cfg.eval.result_file)
    output_file = os.path.join(cfg.general.output_path, cfg.eval.metrics_file)
    score_file = os.path.join(
        cfg.general.output_path, cfg.eval.get("score_file", "scores.json")
    )

    # Check if we should use question type-specific evaluation
    use_question_type_eval = cfg.eval.get("use_question_type_eval", False)
    
    if use_question_type_eval:
        logger.info("Using Nemori-style question type-specific evaluation prompts")
    else:
        logger.info("Using standard LLM judge evaluation")

    # evaluate
    evaluate(cfg, input_file, output_file, use_question_type_eval=use_question_type_eval)

    # generate scores
    scores_data = generate_scores(output_file, score_file)

    return scores_data


if __name__ == "__main__":
    run()