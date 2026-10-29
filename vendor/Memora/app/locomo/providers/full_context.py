# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import time
from collections import defaultdict

from jinja2 import Template
from omegaconf import DictConfig
from tqdm import tqdm

from utils import conversation_id, load_data, select_conversations, select_questions, session_count
from answerer import build_answer_req, parse_answer
from memobench import runtime

# Repo's own full-context template, kept for reference only: answers now go through the one fixed answerer
# (answerer.py, ANSWER_PROMPT_COMBINED) with the conversation in its {{memories}} slot.
PROMPT = """
    You are an intelligent memory assistant that answers questions based on conversation history.

    # CONTEXT:
    You have access to the full conversation history between two speakers. The conversation contains timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze the conversation history
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the conversation
    4. If the conversation contains contradictory information, prioritize the most recent information in terms of timestamp
    5. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the conversation timestamp. For example, if a conversation from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, convert "last year" to "2022" or "two months ago" to "March 2023" based on the conversation timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the conversation. Do not confuse character names mentioned in conversation with the actual speakers.
    8. The answer should be less than 5-6 words.


    # APPROACH (Think step by step):
    1. First, examine all parts of the conversation that contain information related to the question
    2. Examine the timestamps and content carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the conversation
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Conversation History:

    {{CONTEXT}}

    Question: {{QUESTION}}

    Answer:
    """


class FullContextManager:
    """
    Full context baseline that uses the entire conversation history as context
    without any memory extraction or retrieval.
    """

    def __init__(self, cfg: DictConfig, data_path: str):
        self.cfg = cfg
        self.data_path = data_path

    def clean_chat_history(self, chat_history):
        """Format chat history as a readable string."""
        cleaned_chat_history = ""
        for chat in chat_history:
            timestamp = chat.get("timestamp", "")
            speaker = chat.get("speaker", "")
            text = chat.get("text", "")
            cleaned_chat_history += f"{timestamp} | {speaker}: {text}\n"
        return cleaned_chat_history

    def build_requests(self):
        """[(key, record, req)]: one answer request per answerable question (category 5 skipped), the whole
        conversation in the memories slot, cached 1h and grouped by conversation (prefix_group = conv_id)."""
        data = load_data(self.data_path)

        out = []
        for idx, item in select_conversations(data, self.cfg):
            # Get conversation ID from sample_id if available, otherwise use index
            key = conversation_id(item, idx)

            # Extract all chat sessions and flatten them into a single chat history
            conversation = item["conversation"]
            chat_history = []

            # Iterate through all sessions (memory.max_sessions caps them for smoke tests)
            for session_idx in range(1, session_count(conversation, self.cfg) + 1):
                session_chats = conversation[f"session_{session_idx}"]
                session_datetime = conversation.get(f"session_{session_idx}_date_time", "")

                # Add each chat in the session
                for chat in session_chats:
                    chat_with_timestamp = chat.copy()
                    if session_datetime and "timestamp" not in chat_with_timestamp:
                        chat_with_timestamp["timestamp"] = session_datetime
                    chat_history.append(chat_with_timestamp)

            # Format the entire conversation as context
            full_context = self.clean_chat_history(chat_history)

            for q_idx, qa_item in select_questions(item["qa"], self.cfg):
                question = qa_item["question"]
                question_id = f"{key}/q{q_idx}"
                req = build_answer_req(
                    full_context, question, cache_memories=True, batch=True,
                    meta={"system": "full-context", "conv_id": key, "question_id": question_id, "prefix_group": key},
                )
                record = {
                    "question": question,
                    "answer": qa_item.get("answer", ""),
                    "category": str(qa_item["category"]),
                    "context": full_context,  # Store full context for reference
                    "conv_id": key,
                    "question_id": question_id,
                }
                out.append((key, record, req))
        return out

    def process_all_conversations(self, output_file_path):
        """Process all conversations using full context: every answer in one Batch API run."""
        built = self.build_requests()
        results = runtime.get_llm().run([req for _, _, req in built], mode="batch")

        FINAL_RESULTS = {}
        errors = 0
        for key, record, req in tqdm(built, desc="Collecting answers"):
            res = results[req.key]
            if res.error:
                errors += 1
                response = f"ERROR: LLM call failed: {res.error}"
            else:
                response = parse_answer(res.text)
            FINAL_RESULTS.setdefault(key, []).append(
                {**record, "response": response, "response_time": res.latency_ms / 1000.0}
            )

        if errors:
            print(f"WARNING: {errors}/{len(built)} full-context answers failed (recorded as ERROR responses)")

        # Final save
        with open(output_file_path, "w+") as f:
            json.dump(FINAL_RESULTS, f, indent=4)
