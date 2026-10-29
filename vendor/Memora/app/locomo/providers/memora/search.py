# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import logging

from omegaconf import DictConfig
from utils import (
    conversation_id, generate_debug_data, load_data, memora_system_name, select_conversations, select_questions,
)
from memora.memora_client import MemoraClient
from memora.utils.latency import LatencyTracker, count_memories_tokens
from answerer import build_answer_req, parse_answer
from memobench import runtime
from memobench.context import set_meta
from memobench.llm import BudgetExceeded, CostOverrun, NeedsApproval
from tqdm import tqdm

# Initialize logger
logger = logging.getLogger(__name__)


class MemoraSearch:

    def __init__(self, cfg: DictConfig, output_path="results.json", top_k=30, retrieval_strategy="semantic"):
        self.cfg = cfg
        self.top_k = top_k
        self.results = defaultdict(list)
        self.output_path = output_path
        self.retrieval_strategy = retrieval_strategy
        # Check if we should use combined user mode
        self.use_combined_user = self.cfg.eval.get("use_combined_user", False)
        
        # Every system answers through the one fixed answerer (answerer.py: ANSWER_PROMPT_COMBINED, the
        # combined-user "mem0" template); other templates / split-user answering are not part of this port.
        if not self.use_combined_user or self.cfg.eval.get("prompt_template", "mem0") != "mem0":
            raise ValueError("The fixed answerer needs eval.use_combined_user=true and eval.prompt_template=mem0")
        self.system = memora_system_name(cfg, retrieval_strategy)
        self._lock = threading.Lock()
        self._client_lock = threading.Lock()

        self.return_history = self.cfg.memory.get("return_history", False)

        # Read hybrid search setting from config
        self.enable_hybrid_search = self.cfg.memory.get("enable_hybrid_search", False)

        # Read multimodal support setting from config, default to True
        self.multimodal_support = self.cfg.memory.get("multimodal_support", True)
        if self.multimodal_support:
            raise ValueError("Multimodal answering is not ported (set memory.multimodal_support=false)")

        # Use only the latest episode to avoid redundancy in episodic memory presentation
        # Set to True to use only the latest episode for each factual memory
        self.use_latest_episode_only = False

        # cached clients
        self.cached_clients = {}

    def get_memory_client(self, user_id: str) -> MemoraClient:
        # conversations run in parallel threads: create each client once
        with self._client_lock:
            if user_id not in self.cached_clients:
                self.cached_clients[user_id] = MemoraClient(self.cfg, user_id=user_id)
            return self.cached_clients[user_id]

    def search_memory(self, user_id, query, top_k=None, max_retries=3, retry_delay=1, latency_tracker=None):
        start_time = time.time()
        retries = 0

        client = self.get_memory_client(user_id)

        if top_k is None:
            top_k = self.top_k
        else:
            top_k = top_k

        while retries < max_retries:
            try:
                memories = client.advance_query(
                    query,
                    top_k=top_k,
                    query_type=self.retrieval_strategy, # semantic or prompt
                    latency_tracker=latency_tracker,
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
            if self.multimodal_support and memory.image_urls:
                memory_dict["image_urls"] = memory.image_urls
            # Include episodic_memory_ids if they exist
            if memory.episodic_memory_ids:
                memory_dict["episodic_memory_ids"] = memory.episodic_memory_ids
            semantic_memories.append(memory_dict)

        return semantic_memories, end_time - start_time

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
                if self.use_latest_episode_only:
                    # Only use the latest episode (last in the list)
                    all_episodic_ids.add(episodic_ids_list[-1])
                else:
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
                if self.use_latest_episode_only:
                    # Only use the latest episode (last in the list)
                    cluster_key = (episodic_ids_list[-1],)
                else:
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

    def _format_speaker_memories(
        self, memories, user_id, enable_episodic
    ):
        """
        Format memories for a speaker, with or without episodic context.
        All necessary information (including episodic_memory_ids) is already in the memories.
        
        Args:
            memories: List of memory dictionaries for this speaker (already has episodic_memory_ids if present)
            user_id: User ID to get the memory client (for fetching episodic memories)
            enable_episodic: Whether to format with episodic memory context
            
        Returns:
            List of formatted memory strings
        """
        
        # Check if we're using segments directly as episodic memories
        use_segments_as_episodic = self.cfg.memory.get("use_segments_as_episodic", False)
        
        if use_segments_as_episodic:
            # In baseline mode: format with episodes and optionally factual memories
            # 1. Episodes (original conversation segments)
            # 2. Factual memories (in relevance order) - optional
            client = self.get_memory_client(user_id)
            include_factual = self.cfg.memory.get("use_segments_as_episodic_include_factual", True)
            
            episodic_ids_ordered = []
            seen_episodic_ids = set()
            
            for memory in memories:
                # Collect episodic IDs from factual memories in order
                episodic_ids_list = memory.get('episodic_memory_ids', [])
                if episodic_ids_list:
                    for episodic_id in episodic_ids_list:
                        if episodic_id not in seen_episodic_ids:
                            episodic_ids_ordered.append(episodic_id)
                            seen_episodic_ids.add(episodic_id)
            
            formatted_memories = []
            
            # Section 1: Episodes
            if episodic_ids_ordered:
                formatted_memories.append("=== Episodes ===\n")
                for episodic_id in episodic_ids_ordered:
                    try:
                        episodic_memory = client.get(episodic_id)
                        if episodic_memory:
                            timestamp_str = episodic_memory.timestamp if episodic_memory.timestamp is not None else "Unknown date"
                            formatted_memories.append(
                                f"Conversation on {timestamp_str}:\n{episodic_memory.get_memory_value(return_history=self.return_history)}\n"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to retrieve episodic memory {episodic_id}: {e}")
            
            # Section 2: Factual memories in relevance order (optional)
            if include_factual and memories:
                formatted_memories.append("\n=== Factual Details ===\n")
                for memory in memories:
                    formatted_memories.append(f"{memory['timestamp']}: {memory['value']}")
            
            return formatted_memories
        
        if not enable_episodic:
            # Format without episodic context
            formatted_memories = []
            for item in memories:
                formatted_memories.append(f"{item['timestamp']}: {item['value']}")
            
            return formatted_memories
        
        # Retrieve and cluster episodic memories
        episodic_clusters, orphan_memories, episodic_memories_dict = \
            self._retrieve_and_cluster_episodic_memories(memories, user_id)
        
        # Format output
        formatted_memories = []
        
        # Add episodic clusters ordered by max score
        for episodic_ids_tuple, cluster_data in episodic_clusters:
            # Get all episodic memories for this cluster
            valid_episodic_entries = []
            for episodic_id in episodic_ids_tuple:
                episodic_entry = episodic_memories_dict.get(episodic_id)
                if episodic_entry:
                    valid_episodic_entries.append(episodic_entry)
            
            if valid_episodic_entries:
                # Add all episodic memories for this cluster
                for episodic_entry in valid_episodic_entries:
                    # extract date from timestamp if possible
                    if episodic_entry.timestamp is not None:
                        date = episodic_entry.timestamp.split(" on ")[-1] if " on " in episodic_entry.timestamp else episodic_entry.timestamp
                    else:
                        date = "Unknown date"
                        logger.warning(f"Episodic memory {episodic_entry.index} has no timestamp.")
                    formatted_memories.append(
                        f"Conversation on {date}: {episodic_entry.value}"
                    )
                
                formatted_memories.append("Details:")  # Blank line after episodic context
                # Add all factual memories for this cluster (already sorted by score)
                for factual_mem in cluster_data['memories']:
                    formatted_memories.append(
                        f"{factual_mem['timestamp']}: {factual_mem['value']}"
                    )
                
                # Add blank line separator between clusters
                formatted_memories.append("\n")
            else:
                # No valid episodic memories found, treat factual memories as orphans
                orphan_memories.extend(cluster_data['memories'])
        
        # Add orphan memories at the end
        for item in orphan_memories:
            formatted_memories.append(
                f"{item['timestamp']}: {item['value']}"
            )
        
        return formatted_memories

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question):
        # Create latency tracker for this question
        tracker = LatencyTracker()
        tracker.start_overall_search()

        enable_episodic = self.cfg.memory.get("enable_episodic_memory", False)

        if self.use_combined_user:
            # Combined user mode: single query (speaker_1_user_id is the combined ID)
            # Memories are already truncated and ordered by the retriever
            speaker_1_memories, speaker_1_memory_time = self.search_memory(
                speaker_1_user_id, question, latency_tracker=tracker
            )

            # Format memories
            with tracker.track("format_memories"):
                search_1_memory = self._format_speaker_memories(
                    speaker_1_memories, speaker_1_user_id, enable_episodic
                )

            # No separate speaker 2 data in combined mode
            speaker_2_memories = []
            speaker_2_memory_time = 0.0
            search_2_memory = []

        else:
            # Original mode: separate queries for each speaker
            # Memories are already truncated and ordered by the retriever
            speaker_1_memories, time_1 = self.search_memory(
                speaker_1_user_id, question, top_k=self.top_k//2, latency_tracker=tracker
            )
            speaker_2_memories, time_2 = self.search_memory(
                speaker_2_user_id, question, top_k=self.top_k//2, latency_tracker=tracker
            )

            # Format memories for both speakers
            with tracker.track("format_memories"):
                search_1_memory = self._format_speaker_memories(
                    speaker_1_memories, speaker_1_user_id, enable_episodic
                )
                search_2_memory = self._format_speaker_memories(
                    speaker_2_memories, speaker_2_user_id, enable_episodic
                )

            speaker_1_name = speaker_1_user_id.split("_")[0]
            speaker_2_name = speaker_2_user_id.split("_")[0]
            speaker_1_memory_time = time_1
            speaker_2_memory_time = time_2

        # End overall search timing
        tracker.end_overall_search()
        
        # Collect image URLs (common for both modes)
        all_image_urls = set()
        for memory in speaker_1_memories + speaker_2_memories:
            if memory.get('image_urls'):
                all_image_urls.update(memory['image_urls'])

        all_images = [
            {"type": "image_url", "image_url": {"url": url}}
            for url in all_image_urls
        ][:500]  # Limit to 500 images

        # Render the fixed answer prompt (answerer.py), exactly as upstream rendered it in combined-user mode
        req = build_answer_req(json.dumps(search_1_memory, indent=4), question, cache_memories=False)

        # Count tokens in memories (not the full prompt, just the memories)
        all_memory_strings = search_1_memory + search_2_memory
        try:
            token_stats = count_memories_tokens(
                all_memory_strings,
                model=self.cfg.llm.get("model", "gpt-4o")
            )
            token_stats["num_images"] = len(all_images)
            tracker.set_prompt_stats(token_stats)
        except Exception as e:
            logger.warning(f"Failed to count tokens: {e}")

        # Answer in real time through the shared LLM layer (it retries 429/5xx itself)
        try:
            with tracker.track("llm_generation"):
                response = runtime.get_llm().call(req)
            if response.error:
                raise RuntimeError(response.error)
        except (BudgetExceeded, NeedsApproval, CostOverrun):
            raise
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            response_time = tracker.get_timing("llm_generation")
            # Get latency summary before returning
            latency_summary = tracker.get_summary()
            # Return empty result with error indicator
            return (
                f"ERROR: LLM call failed.",
                speaker_1_memories,
                speaker_2_memories,
                speaker_1_memory_time,
                speaker_2_memory_time,
                response_time,
                search_1_memory,
                search_2_memory,
                latency_summary,
            )

        response_time = tracker.get_timing("llm_generation")

        # Extract final answer
        result = parse_answer(response.text)
        if result.startswith("ERROR: Empty response"):
            logger.error("Empty response from LLM.")

        # Get comprehensive latency summary
        latency_summary = tracker.get_summary()

        return (
            result,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            response_time,
            search_1_memory,
            search_2_memory,
            latency_summary,
        )

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            response_time,
            formatted_speaker_1_memories,
            formatted_speaker_2_memories,
            latency_summary,
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "response_time": response_time,
            "formatted_speaker_1_memories": formatted_speaker_1_memories,
            "formatted_speaker_2_memories": formatted_speaker_2_memories,
            "latency_breakdown": latency_summary,
        }

        return result

    def process_data_file(self, file_path):

        data = load_data(file_path, subset_idx=self.cfg.eval.subset_idx)

        if self.cfg.general.debug:
            data = generate_debug_data(data)

        # Conversations in parallel (thread pool), questions sequential within a conversation
        conversations = select_conversations(data, self.cfg)
        with ThreadPoolExecutor(max_workers=self.cfg.eval.max_workers) as executor:
            futures = [executor.submit(self._process_conversation, idx, item) for idx, item in conversations]
            for future in tqdm(futures, desc="Processing conversations"):
                future.result()

        # Final save at the end
        self._save()

    def _process_conversation(self, idx, item):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        if self.use_combined_user:
            # Combined mode: use single combined user ID
            speaker_a_user_id = f"{speaker_a}_{speaker_b}_{idx}"
            speaker_b_user_id = None  # Not used in combined mode
        else:
            # Original mode: separate user IDs
            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

        conv_id = conversation_id(item, idx)
        for q_idx, question_item in select_questions(item["qa"], self.cfg):  # category 5 is never answered
            question_id = f"{conv_id}/q{q_idx}"
            set_meta(system=self.system, conv_id=conv_id, question_id=question_id)
            result = self.process_question(question_item, speaker_a_user_id, speaker_b_user_id)
            result.update(conv_id=conv_id, question_id=question_id)
            with self._lock:
                self.results[idx].append(result)

            # Save results after each question is processed
            self._save()

    def _save(self):
        with self._lock:
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)

    def process_questions_parallel(self, qa_list, speaker_a_user_id, speaker_b_user_id, max_workers=1):
        def process_single_question(val):
            result = self.process_question(val, speaker_a_user_id, speaker_b_user_id)
            # Save results after each question is processed
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(executor.map(process_single_question, qa_list), total=len(qa_list), desc="Answering Questions")
            )

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return results
