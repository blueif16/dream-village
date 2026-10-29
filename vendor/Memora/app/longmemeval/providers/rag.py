# RAG baseline for LongMemEval: each question's haystack history is split into fixed-size token chunks, the top-k
# chunks by cosine similarity to the question go into the {{ memories }} slot of the one fixed answerer
# (answerer.py, the repo-default `longmemeval` template). Mirrors app/locomo/providers/rag.py.

import json
import logging
import time

import numpy as np
import tiktoken
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from answerer import answer_template, build_answer_req, parse_answer
from memobench import runtime
from memobench.context import set_meta
from utils import render_history, select_questions

logger = logging.getLogger(__name__)

SYSTEM = "rag"


class RAGManager:
    def __init__(self, cfg: DictConfig, data_path: str, chunk_size=500, k=3):
        self.cfg = cfg
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.k = k

    def calculate_embeddings(self, texts, kind):
        """gemini-embedding-2 via memobench.embed: chunks as kind="document", questions as kind="query"."""
        from memobench.embed import embed  # imported on use, so the runners import without the embedding stack

        return embed(list(texts), kind)

    def calculate_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def search(self, query_embedding, chunks, embeddings, k=1):
        """Top-k most similar chunks to the query, joined as in the LoCoMo RAG baseline; returns (text, seconds)."""
        t1 = time.time()
        similarities = [self.calculate_similarity(query_embedding, embedding) for embedding in embeddings]
        if k == 1:
            top_indices = [np.argmax(similarities)]
        else:
            top_indices = np.argsort(similarities)[-k:][::-1]
        combined_chunks = "\n<->\n".join([chunks[i] for i in top_indices])
        return combined_chunks, time.time() - t1

    def create_chunks(self, history: str, chunk_size=500):
        """Token chunks of the rendered history (cl100k_base, the LoCoMo RAG baseline's counter), embedded as
        documents. Special-token strings in the data (one turn contains "<|endoftext|>") are encoded as plain text."""
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(history, disallowed_special=())
        chunks = [encoding.decode(tokens[i : i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
        return chunks, self.calculate_embeddings(chunks, "document")

    def build_requests(self):
        """[(record, req)]: retrieve the top-k chunks of each selected question's own haystack."""
        with open(self.data_path, "r") as f:
            data = select_questions(json.load(f), self.cfg)
        template = answer_template(self.cfg)

        def one(item):
            question_id = item["question_id"]
            question = item["question"]
            set_meta(system=SYSTEM, conv_id=question_id, question_id=question_id)  # attributes the embedding calls

            chunks, embeddings = self.create_chunks(render_history(item), self.chunk_size)
            (query_embedding,) = self.calculate_embeddings([question], "query")
            context, search_time = self.search(query_embedding, chunks, embeddings, k=self.k)

            req = build_answer_req(
                context, question, item["question_date"], template=template, cache_memories=False, batch=True,
                meta={"system": SYSTEM, "conv_id": question_id, "question_id": question_id},
            )
            record = {
                "question_id": question_id,
                "question": question,
                "answer": item["answer"],
                "question_type": item["question_type"],
                "question_date": item["question_date"],
                "answer_session_ids": item.get("answer_session_ids", []),
                "context": context,
                "num_chunks": len(chunks),
                "search_time": search_time,
            }
            return record, req

        # questions are independent: embed several haystacks at once (order preserved by map)
        with ThreadPoolExecutor(max_workers=int(self.cfg.eval.get("max_workers", 10))) as pool:
            return list(tqdm(pool.map(one, data), total=len(data), desc="Chunking and retrieving"))

    def process_all_questions(self, output_file_path):
        """Retrieve for every selected question, then answer all of them in one Batch API run; write
        {question_id: result} (evals.py's format)."""
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
            logger.warning(f"{errors}/{len(built)} RAG answers failed (recorded as ERROR answers)")

        with open(output_file_path, "w") as f:
            json.dump(final_results, f, indent=4)
        return final_results
