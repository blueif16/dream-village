# Full-context baseline for LongMemEval: the whole haystack history of each question goes into the {{ memories }}
# slot of the one fixed answerer (answerer.py, the repo-default `longmemeval` template), no extraction or retrieval.
# Mirrors app/locomo/providers/full_context.py.

import json
import logging

from omegaconf import DictConfig
from tqdm import tqdm

from answerer import answer_template, build_answer_req, parse_answer
from memobench import runtime
from utils import render_history, select_questions

logger = logging.getLogger(__name__)

SYSTEM = "full-context"


class FullContextManager:
    """Full context baseline that uses the entire haystack history as context."""

    def __init__(self, cfg: DictConfig, data_path: str):
        self.cfg = cfg
        self.data_path = data_path

    def build_request(self, item):
        """(record, req) for one question: its whole history in the memories slot. Every question has its own
        haystack, so the history is not cached and there is no warm-up group (a group of one would run real-time)."""
        question_id = item["question_id"]
        req = build_answer_req(
            render_history(item), item["question"], item["question_date"], template=answer_template(self.cfg),
            cache_memories=False, batch=True,
            meta={"system": SYSTEM, "conv_id": question_id, "question_id": question_id},
        )
        record = {
            "question_id": question_id,
            "question": item["question"],
            "answer": item["answer"],
            "question_type": item["question_type"],
            "question_date": item["question_date"],
            "answer_session_ids": item.get("answer_session_ids", []),
            # the history itself is not stored (~0.5 MB per question); it is a pure function of the dataset
            "num_sessions": len(item["haystack_sessions"]),
        }
        return record, req

    def build_requests(self):
        """[(record, req)]: one answer request per selected question (eval.subset_idx)."""
        with open(self.data_path, "r") as f:
            data = select_questions(json.load(f), self.cfg)
        return [self.build_request(item) for item in data]

    def process_all_questions(self, output_file_path):
        """Answer every selected question in one Batch API run; write {question_id: result} (evals.py's format)."""
        built = self.build_requests()
        results = runtime.get_llm().run([req for _, req in built], mode="batch")

        final_results = {}
        errors = 0
        for record, req in tqdm(built, desc="Collecting answers"):
            res = results[req.key]
            if res.error:
                errors += 1
                generated_answer = f"ERROR: LLM call failed: {res.error}"
            else:
                generated_answer = parse_answer(res.text)
            final_results[record["question_id"]] = {
                **record, "generated_answer": generated_answer, "response_time": res.latency_ms / 1000.0,
            }

        if errors:
            logger.warning(f"{errors}/{len(built)} full-context answers failed (recorded as ERROR answers)")

        with open(output_file_path, "w") as f:
            json.dump(final_results, f, indent=4)
        return final_results
