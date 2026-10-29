# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import gc
import time
import threading
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from omegaconf import DictConfig, OmegaConf
from memora.memora_client import MemoraClient
from memobench import runtime
from memobench.context import set_meta
from memobench.llm import BudgetExceeded, CostOverrun, NeedsApproval
from answerer import answer_template, build_answer_req, parse_answer
from tqdm import tqdm
from utils import memora_system_name

# Initialize logger
logger = logging.getLogger(__name__)

class MemoraSearch:

    def __init__(self, cfg: DictConfig, output_path="results.json", top_k=30, retrieval_strategy="semantic"):

        self.cfg = cfg
        self.top_k = top_k
        self.retrieval_strategy = retrieval_strategy
        self.results = defaultdict(list)
        self.output_path = output_path

        # Select answer template based on config (answerer.py: the one fixed answerer shared with the baselines)
        self.ANSWER_PROMPT = answer_template(self.cfg)

        self.system = memora_system_name(cfg, retrieval_strategy)
        self.return_history = self.cfg.memory.get("return_history", False)
        self.enable_hybrid_search = self.cfg.memory.get("enable_hybrid_search", False)
        self.use_latest_episode_only = self.cfg.memory.get("use_latest_episode_only", False)
        self.retrieval_strategy = retrieval_strategy

        # cached clients with LRU eviction to prevent file descriptor exhaustion
        self._max_cached_clients = self.cfg.eval.get("max_cached_clients", 20)
        self.cached_clients = OrderedDict()
        self._client_lock = threading.Lock()
        self._file_lock = threading.Lock()

        # parallel search config
        self.parallel_search = self.cfg.eval.get("parallel_search", False)

    def _evict_oldest_client(self):
        """Evict oldest cached client to free file descriptors. Must be called with _client_lock held."""
        if len(self.cached_clients) >= self._max_cached_clients:
            evicted_id, evicted_client = self.cached_clients.popitem(last=False)
            del evicted_client
            gc.collect()
            logger.debug(f"Evicted cached client for {evicted_id} (cache size: {len(self.cached_clients)})")

    def get_memory_client(self, user_id: str):
        # Check if client exists (with lock for thread safety)
        with self._client_lock:
            if user_id in self.cached_clients:
                self.cached_clients.move_to_end(user_id)
                return self.cached_clients[user_id]

        # Create new client with retry logic for tenant errors
        max_retries = 5
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                client = MemoraClient(self.cfg, user_id=user_id)

                # Add to cache (with lock, double-check pattern)
                with self._client_lock:
                    if user_id not in self.cached_clients:
                        self._evict_oldest_client()
                        self.cached_clients[user_id] = client
                    else:
                        self.cached_clients.move_to_end(user_id)
                    return self.cached_clients[user_id]

            except Exception as e:
                error_msg = str(e)
                if "tenant" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"ChromaDB tenant error for user {user_id}, attempt {attempt+1}/{max_retries}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Failed to create client for {user_id} after {max_retries} attempts: {e}")
                        raise
                else:
                    raise

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0

        client = self.get_memory_client(user_id)
        while retries < max_retries:
            try:
                if self.retrieval_strategy in ("prompt", "grpo"):
                    memories = client.advance_query(
                        query,
                        top_k=self.top_k,
                        query_type=self.retrieval_strategy,
                    )
                else:
                    memories = client.query(
                        query,
                        top_k=self.top_k,
                        enable_hybrid_search=self.enable_hybrid_search,
                        where={"memory_type": {"$eq": "factual"}},
                    )
                break
            except Exception as e:
                print(f"Retrying...{e}")
                retries += 1
                if retries >= max_retries:
                    return [], 0.0
                time.sleep(retry_delay)

        end_time = time.time()

        semantic_memories = []
        for memory in memories:
            memory_dict = {
                "index": memory.index,
                "value": memory.get_memory_value(return_history=self.return_history),
                "timestamp": memory.timestamp,
                "score": round(memory.score, 2),
                "memory_type": memory.memory_type,
            }
            # Include image URLs if multimodal support is enabled and images exist
            if getattr(self, 'multimodal_support', False) and memory.image_urls:
                memory_dict["image_urls"] = memory.image_urls
            
            if memory.episodic_memory_ids:
                memory_dict["episodic_memory_ids"] = memory.episodic_memory_ids 
            semantic_memories.append(memory_dict)

        return semantic_memories, end_time - start_time

    def process_question(self, item, idx):
        """Process a single LongMemEval question."""
        question_id = item.get("question_id", "")
        question = item.get("question", "")
        answer = item.get("answer", "")
        question_type = item.get("question_type", "")
        question_date = item.get("question_date", "")  
        answer_session_ids = item.get("answer_session_ids", [])
        set_meta(system=self.system, conv_id=question_id, question_id=question_id)

        user_id = f"question_{question_id}"
        
        memories, memory_time = self.search_memory(user_id, question)

        # Format memories with episodic context if enabled
        formatted_memories = self._format_memories(memories, user_id)
        
        # Collect image URLs
        all_image_urls = set()
        for item_mem in memories:
            if item_mem.get('image_urls', None):
                for img_url in item_mem['image_urls']:
                    all_image_urls.add(img_url)

        all_images = [
            {"type": "image_url", "image_url": {"url": url}}
            for url in all_image_urls
        ][:500]

        # Render prompt (answerer.py: same Template(...).render(...), split into cacheable blocks)
        req = build_answer_req("\n".join(formatted_memories), question, question_date, template=self.ANSWER_PROMPT)

        t1 = time.time()
        try:
            # Real time through the shared LLM layer (it retries 429/5xx itself); images are not ported
            response = runtime.get_llm().call(req)
            if response.error:
                raise RuntimeError(response.error)
        except (BudgetExceeded, NeedsApproval, CostOverrun):
            raise
        except Exception as e:
            logger.error(f"LLM call failed for question {question_id}: {e}")
            return {
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "question_type": question_type,
                "question_date": question_date,
                "answer_session_ids": answer_session_ids,
                "generated_answer": f"ERROR: LLM call failed: {e}",
                "memories": memories,
                "formatted_memories": formatted_memories,
                "num_memories": len(memories),
                "memory_time": memory_time,
                "response_time": 0.0,
            }
        
        t2 = time.time()
        response_time = t2 - t1

        # Extract answer from response (FINAL ANSWER extraction, answerer.py)
        generated_answer = parse_answer(response.text)

        result = {
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "question_type": question_type,
            "question_date": question_date,
            "answer_session_ids": answer_session_ids,
            "generated_answer": generated_answer,
            "memories": memories,
            "formatted_memories": formatted_memories,
            "num_memories": len(memories),
            "memory_time": memory_time,
            "response_time": response_time,
        }

        return result

    def _process_and_write_question(self, item, idx):
        """Process a single question and write result to file (thread-safe)."""
        result = self.process_question(item, idx)
        question_id = result["question_id"]

        # Thread-safe write to results dict and file
        with self._file_lock:
            self.results[question_id] = result
            with open(self.output_path, "a") as f:
                f.write(json.dumps(result) + "\n")

        return result

    def load_data(self, file_path):
        """Load and filter data from file."""
        with open(file_path, "r") as f:
            data = json.load(f)

        subset_idx = self.cfg.eval.subset_idx

        # Parse subset_idx: supports -1 (all), single number, range "1:10", or comma-separated
        if isinstance(subset_idx, str):
            if ":" in subset_idx:
                start_str, end_str = subset_idx.split(":")
                start_idx = int(start_str)
                end_idx = int(end_str)
                if start_idx > 0 and end_idx > 0:
                    data = data[start_idx - 1 : end_idx]
                    logger.info(f"Loading questions {start_idx} to {end_idx} (inclusive)")
            elif "," in subset_idx:
                question_indices = [int(idx.strip()) for idx in subset_idx.split(",")]
                data = [data[idx - 1] for idx in question_indices if 0 < idx <= len(data)]
                logger.info(f"Loading {len(data)} specific questions")
            else:
                idx_val = int(subset_idx)
                if idx_val > 0:
                    data = data[idx_val - 1 : idx_val]
        elif isinstance(subset_idx, int) and subset_idx > 0:
            data = data[subset_idx - 1 : subset_idx]
        else:
            logger.info(f"Loading all {len(data)} questions")

        if self.cfg.general.debug:
            data = data[:1]

        self.data = data
        logger.info(f"Loaded {len(self.data)} questions for processing.")
        return self.data

    def process_data_file(self, file_path):
        """Process LongMemEval data file. Supports both sequential and parallel modes."""
        
        self.load_data(file_path)

        if not self.data:
            raise ValueError("No data loaded. Please check the file path and subset_idx configuration.")

        print(f"Total questions to process: {len(self.data)}")

        if self.parallel_search:
            # Parallel mode: process questions with ThreadPoolExecutor
            self._process_all_questions_parallel()
        else:
            # Sequential mode: original behavior
            self._process_all_questions_sequential()

    def _process_all_questions_sequential(self):
        """Process all questions sequentially (original behavior)."""
        for idx, item in tqdm(enumerate(self.data), total=len(self.data), desc="Processing questions"):
            result = self.process_question(item, idx)
            question_id = result["question_id"]
            self.results[question_id] = result
            
            # Save results after each question
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)
        
        # Final save
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

    def _process_all_questions_parallel(self):
        """Process all questions in parallel with ThreadPoolExecutor."""
        # Clear the output file for JSONL append mode
        with open(self.output_path, "w") as f:
            pass

        with ThreadPoolExecutor(max_workers=self.cfg.eval.max_workers) as executor:
            futures = {
                executor.submit(self._process_and_write_question, item, idx): (item, idx)
                for idx, item in enumerate(self.data)
            }

            completed = 0
            failed = []

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing questions"):
                item, idx = futures[future]
                try:
                    future.result()
                    completed += 1
                except Exception as e:
                    question_id = item.get("question_id", f"idx_{idx}")
                    failed.append((question_id, str(e)))
                    logger.error(f"Failed to process question {question_id}: {e}")

                    # Write error result
                    error_result = {
                        "question_id": question_id,
                        "question": item.get("question", ""),
                        "answer": item.get("answer", ""),
                        "question_type": item.get("question_type", ""),
                        "question_date": item.get("question_date", ""),
                        "answer_session_ids": item.get("answer_session_ids", []),
                        "generated_answer": f"ERROR: Processing failed: {str(e)[:200]}",
                        "memories": [],
                        "formatted_memories": [],
                        "num_memories": 0,
                        "memory_time": 0.0,
                        "response_time": 0.0,
                    }
                    with self._file_lock:
                        with open(self.output_path, "a") as f:
                            f.write(json.dumps(error_result) + "\n")

            # Summary
            print(f"\n{'='*60}")
            print(f"Search & Inference Complete:")
            print(f"  Successful: {completed}/{len(self.data)}")
            print(f"  Failed: {len(failed)}")
            if failed:
                print(f"  Failed questions: {[qid for qid, _ in failed]}")
            print(f"{'='*60}\n")

    def _retrieve_and_cluster_episodic_memories(self, memories, user_id):
        """
        Retrieve episodic memories and group factual memories by their linked episodes.
        Creates multi-episode clusters when factual memories share the same set of episodes.
        
        Args:
            memories: List of factual memory dictionaries
            user_id: User ID to get the memory client
            
        Returns:
            Tuple of (episodic_clusters, orphan_memories, episodic_memories_dict)
            where episodic_clusters is a list of (episodic_ids_tuple, cluster_data) tuples
        """
        client = self.get_memory_client(user_id)
        
        # Step 1: Collect all episodic IDs that need to be retrieved
        all_episodic_ids = set()
        for memory in memories:
            episodic_ids_list = memory.get('episodic_memory_ids', [])
            if episodic_ids_list:
                all_episodic_ids.update(episodic_ids_list)
            elif memory.get('memory_type') == 'episodic':
                # This is an episodic memory retrieved directly
                all_episodic_ids.add(memory['index'])
        
        # Step 2: Retrieve all episodic memories
        episodic_memories_dict = {}
        for episodic_id in all_episodic_ids:
            episodic_entry = client.get(episodic_id)
            if episodic_entry:
                episodic_memories_dict[episodic_id] = episodic_entry
        
        # Step 3: Cluster factual memories by their episodic IDs
        episodic_clusters = {}  # tuple of episodic_ids -> {'memories': [...], 'max_score': float}
        orphan_memories = []  # memories with no episodic link
        
        for memory in memories:
            episodic_ids_list = memory.get('episodic_memory_ids', [])
            if episodic_ids_list:
                # Use tuple of episodic IDs as cluster key (preserves order)
                cluster_key = tuple(episodic_ids_list)
                if cluster_key not in episodic_clusters:
                    episodic_clusters[cluster_key] = {
                        'memories': [],
                        'max_score': memory['score']
                    }
                else:
                    # Update max_score if this memory has a higher score
                    episodic_clusters[cluster_key]['max_score'] = max(
                        episodic_clusters[cluster_key]['max_score'],
                        memory['score']
                    )
                episodic_clusters[cluster_key]['memories'].append(memory)
            else:
                # No episodic links - could be factual memory or episodic memory retrieved directly
                orphan_memories.append(memory)
        
        # Step 4: Handle standalone episodic memories (retrieved directly but not referenced)
        remaining_orphans = []
        for memory in orphan_memories:
            if memory.get('memory_type') == 'episodic' and memory['index'] in episodic_memories_dict:
                # Check if this episodic memory is already referenced by a cluster
                is_referenced = any(memory['index'] in cluster_key for cluster_key in episodic_clusters.keys())
                if not is_referenced:
                    # Create a standalone cluster for this episodic memory with its own score
                    cluster_key = (memory['index'],)
                    episodic_clusters[cluster_key] = {
                        'memories': [],
                        'max_score': memory['score']  # Use the episodic memory's own score
                    }
            else:
                remaining_orphans.append(memory)
        
        orphan_memories = remaining_orphans
        
        # Step 5: Sort factual memories within each cluster by score (descending)
        for cluster_data in episodic_clusters.values():
            cluster_data['memories'].sort(key=lambda x: x['score'], reverse=True)
        
        # Step 6: Sort clusters by their maximum score (descending)
        sorted_clusters = sorted(
            episodic_clusters.items(),
            key=lambda x: x[1]['max_score'],
            reverse=True
        )
        
        return sorted_clusters, orphan_memories, episodic_memories_dict

    def _extract_user_messages(self, original_text):
        """
        Extract only user messages from a conversation transcript.

        Args:
            original_text: The original conversation text (user-assistant dialogue)

        Returns:
            String containing only user messages, formatted one per line
        """
        user_messages = []
        lines = original_text.split('\n')

        for line in lines:
            line = line.strip()
            # Match patterns like "User:", "user:", "USER:", or similar
            if line.lower().startswith('user:'):
                # Extract message after "User:"
                message = line.split(':', 1)[1].strip()
                if message:
                    user_messages.append(message)

        return '\n'.join(user_messages) if user_messages else original_text

    def _format_memories(
        self, memories, user_id
    ):
        """
        Format memories for a speaker, with or without episodic context.
        All necessary information (including episodic_memory_ids) is already in the memories.
        
        Args:
            memories: List of memory dictionaries for this speaker (already has episodic_memory_ids if present)
            user_id: User ID to get the memory client (for fetching episodic memories)
            
        Returns:
            List of formatted memory strings
        """
        
        # Get memory presentation flags
        enable_episodic = self.cfg.memory.get("enable_episodic", True)
        enable_factual = self.cfg.memory.get("enable_factual", True)

        # Episodic format: "original_text", "hybrid", "extracted", "user_text_only"
        episodic_format = self.cfg.memory.get("episodic_format", "original_text")

        # Always format episodic and factual memories separately (default behavior)
        client = self.get_memory_client(user_id)

        # Collect episodic IDs with their relevance scores
        episodic_data = {}  # episodic_id -> (max_score, memory_object)

        for memory in memories:
            # Collect episodic IDs from factual memories with their scores
            episodic_ids_list = memory.get('episodic_memory_ids', [])
            if episodic_ids_list:
                for episodic_id in episodic_ids_list:
                    if episodic_id not in episodic_data:
                        episodic_data[episodic_id] = (memory['score'], None)
                    else:
                        # Keep the maximum score for this episode
                        episodic_data[episodic_id] = (
                            max(episodic_data[episodic_id][0], memory['score']),
                            episodic_data[episodic_id][1]
                        )

        # Fetch episodic memories and store them
        for episodic_id in episodic_data.keys():
            try:
                episodic_memory = client.get(episodic_id)
                if episodic_memory:
                    score, _ = episodic_data[episodic_id]
                    episodic_data[episodic_id] = (score, episodic_memory)
            except Exception as e:
                logger.warning(f"Failed to retrieve episodic memory {episodic_id}: {e}")

        # Sort episodes by relevance (score) in descending order
        sorted_episodes = sorted(
            [(ep_id, score, mem) for ep_id, (score, mem) in episodic_data.items() if mem is not None],
            key=lambda x: x[1],
            reverse=True
        )

        formatted_memories = []

        # Section 1: Episodes (sorted by relevance, deduplicated by episodic_id)
        if enable_episodic and sorted_episodes:
            formatted_memories.append("=== Episodes ===\n")
            for episodic_id, score, episodic_memory in sorted_episodes:
                timestamp_str = episodic_memory.timestamp if episodic_memory.timestamp is not None else "Unknown date"

                if episodic_format == "hybrid":
                    # Hybrid mode: Extract user messages from original + add episodic summary
                    original_text = episodic_memory.get_memory_value(return_history=self.return_history, use_original_text=True)
                    user_messages = self._extract_user_messages(original_text)

                    # Get extracted episodic summary
                    episodic_summary = episodic_memory.get_memory_value(return_history=self.return_history, use_original_text=False)

                    # Combine: user messages + summary
                    formatted_memories.append(
                        f"Conversation on {timestamp_str}:\n"
                        f"User messages: {user_messages}\n"
                        f"Summary: {episodic_summary}\n"
                    )
                elif episodic_format == "user_text_only":
                    # User text only: Extract and show only user messages from original
                    original_text = episodic_memory.get_memory_value(return_history=self.return_history, use_original_text=True)
                    user_messages = self._extract_user_messages(original_text)

                    formatted_memories.append(
                        f"Conversation on {timestamp_str} (with only user messages):\n{user_messages}\n"
                    )
                elif episodic_format == "extracted":
                    # Extracted: Use only the extracted episodic summary
                    episodic_summary = episodic_memory.get_memory_value(return_history=self.return_history, use_original_text=False)
                    formatted_memories.append(
                        f"Conversation on {timestamp_str}:\n{episodic_summary}\n"
                    )
                else:  # episodic_format == "original_text" (default)
                    # Original text: Use full original conversation (user + assistant)
                    original_text = episodic_memory.get_memory_value(return_history=self.return_history, use_original_text=True)
                    formatted_memories.append(
                        f"Conversation on {timestamp_str}:\n{original_text}\n"
                    )

        # Section 2: Factual memories (deduplicated and sorted by relevance)
        if enable_factual and memories:
            formatted_memories.append("\n=== Factual Details ===\n")

            # Deduplicate factual memories by (timestamp, value) and keep highest score
            factual_data = {}  # (timestamp, value) -> max_score
            for memory in memories:
                key = (memory['timestamp'], memory['value'])
                if key not in factual_data:
                    factual_data[key] = memory['score']
                else:
                    factual_data[key] = max(factual_data[key], memory['score'])

            # Sort by score (relevance) in descending order
            sorted_factuals = sorted(
                [(timestamp, value, score) for (timestamp, value), score in factual_data.items()],
                key=lambda x: x[2],
                reverse=True
            )

            for timestamp, value, score in sorted_factuals:
                formatted_memories.append(f"{timestamp}: {value}")

        return formatted_memories
