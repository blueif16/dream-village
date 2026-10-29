# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import time
from collections import defaultdict

import numpy as np
import tiktoken
from dotenv import load_dotenv
from jinja2 import Template
from omegaconf import DictConfig
from tqdm import tqdm

from utils import conversation_id, load_data, select_conversations, select_questions, session_count
from answerer import build_answer_req, parse_answer
from memobench import runtime
from memobench.context import set_meta

load_dotenv()

# Repo's own RAG template, kept for reference only: answers now go through the one fixed answerer
# (answerer.py, ANSWER_PROMPT_COMBINED) with the retrieved chunks in its {{memories}} slot.
PROMPT = """
    You are an intelligent memory assistant that answers questions based on retrieved context.

    # CONTEXT:
    You have access to retrieved context from a conversation between two speakers. This context contains timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze the provided context
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the context
    4. If the context contains contradictory information, prioritize the most recent information in terms of timestamp
    5. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the context timestamp. For example, if context from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, convert "last year" to "2022" or "two months ago" to "March 2023" based on the context timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the context. Do not confuse character names mentioned in context with the actual speakers.
    8. The answer should be less than 5-6 words.


    # APPROACH (Think step by step):
    1. First, examine all parts of the context that contain information related to the question
    2. Examine the timestamps and content carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the context
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Retrieved Context:

    {{CONTEXT}}

    Question: {{QUESTION}}

    Answer:
    """


class RAGManager:
    def __init__(self, cfg: DictConfig, data_path="dataset/locomo10.json", chunk_size=500, k=1):
        self.cfg = cfg
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.k = k

    def clean_chat_history(self, chat_history):
        cleaned_chat_history = ""
        for c in chat_history:
            cleaned_chat_history += f"{c['timestamp']} | {c['speaker']}: {c['text']}\n"

        return cleaned_chat_history

    def calculate_embeddings(self, texts, kind):
        """gemini-embedding-2 via memobench.embed: chunks as kind="document", questions as kind="query"."""
        from memobench.embed import embed  # imported on use, so the runners import without the embedding stack

        return embed(list(texts), kind)

    def calculate_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def search(self, query_embedding, chunks, embeddings, k=1):
        """
        Search for the top-k most similar chunks to the query.

        Args:
            query_embedding: Embedding of the query (kind="query")
            chunks: List of text chunks
            embeddings: List of embeddings for each chunk
            k: Number of top chunks to return (default: 1)

        Returns:
            combined_chunks: The combined text of the top-k chunks
            search_time: Time taken for the search
        """
        t1 = time.time()
        similarities = [self.calculate_similarity(query_embedding, embedding) for embedding in embeddings]

        # Get indices of top-k most similar chunks
        if k == 1:
            # Original behavior - just get the most similar chunk
            top_indices = [np.argmax(similarities)]
        else:
            # Get indices of top-k chunks
            top_indices = np.argsort(similarities)[-k:][::-1]

        # Combine the top-k chunks
        combined_chunks = "\n<->\n".join([chunks[i] for i in top_indices])

        t2 = time.time()
        return combined_chunks, t2 - t1

    def create_chunks(self, chat_history, chunk_size=500):
        """
        Create chunks using tiktoken for more accurate token counting
        """
        # Token counting as in the repo: tiktoken.encoding_for_model("text-embedding-3-small") is cl100k_base
        encoding = tiktoken.get_encoding("cl100k_base")

        documents = self.clean_chat_history(chat_history)

        if chunk_size == -1:
            return [documents], []

        chunks = []

        # Encode the document
        tokens = encoding.encode(documents)

        # Split into chunks based on token count
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk = encoding.decode(chunk_tokens)
            chunks.append(chunk)

        embeddings = self.calculate_embeddings(chunks, "document")

        return chunks, embeddings

    def build_requests(self):
        """[(key, record, req)]: retrieve top-k chunks per answerable question (category 5 skipped)."""
        data = load_data(self.data_path)

        out = []
        for idx, item in select_conversations(data, self.cfg):
            # Get conversation ID from sample_id if available, otherwise use index
            key = conversation_id(item, idx)
            set_meta(system="rag", conv_id=key, question_id=None)  # attributes the embedding calls

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

            questions = select_questions(item["qa"], self.cfg)

            chunks, embeddings = self.create_chunks(chat_history, self.chunk_size)
            if self.chunk_size != -1 and questions:
                query_embeddings = self.calculate_embeddings([qa["question"] for _, qa in questions], "query")

            for n, (q_idx, qa_item) in enumerate(questions):
                question = qa_item["question"]
                question_id = f"{key}/q{q_idx}"

                if self.chunk_size == -1:
                    context = chunks[0]
                    search_time = 0
                else:
                    context, search_time = self.search(query_embeddings[n], chunks, embeddings, k=self.k)

                req = build_answer_req(
                    context, question, cache_memories=False, batch=True,
                    meta={"system": "rag", "conv_id": key, "question_id": question_id},
                )
                record = {
                    "question": question,
                    "answer": qa_item.get("answer", ""),
                    "category": str(qa_item["category"]),
                    "context": context,
                    "search_time": search_time,
                    "conv_id": key,
                    "question_id": question_id,
                }
                out.append((key, record, req))
        return out

    def process_all_conversations(self, output_file_path):
        """Retrieve for every question, then answer all of them in one Batch API run."""
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
            print(f"WARNING: {errors}/{len(built)} RAG answers failed (recorded as ERROR responses)")

        # Final save
        with open(output_file_path, "w+") as f:
            json.dump(FINAL_RESULTS, f, indent=4)
