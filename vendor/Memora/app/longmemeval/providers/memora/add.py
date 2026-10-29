# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import time
from datetime import datetime
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from importlib_metadata import metadata
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from memora.memora_client import MemoraClient
from memobench.context import set_meta
from openai import ContentFilterFinishReasonError
from utils import memora_system_name

logger = logging.getLogger(__name__)

class MemoraADD:

    def __init__(
        self,
        cfg: DictConfig,
        data_path=None,
        batch_size=32,
        data_config=None,
        skip_existing_memories=False,
    ):
        self.cfg = cfg
        self.data_config = data_config
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        self.skip_existing_memories = skip_existing_memories

        if data_path:
            self.load_data()

        # cache of memory clients
        self.cached_clients = {}
        self._client_lock = threading.Lock()

    def get_memory_client(self, user_id: str):
        # Check if client exists (with lock for thread safety)
        with self._client_lock:
            if user_id in self.cached_clients:
                return self.cached_clients[user_id]

        # Create new client (outside lock to avoid blocking)
        client = MemoraClient(self.cfg, user_id=user_id)

        # Add to cache (with lock, double-check pattern)
        with self._client_lock:
            if user_id not in self.cached_clients:
                self.cached_clients[user_id] = client
            return self.cached_clients[user_id]

    def check_question_has_memories(self, question_id: str) -> bool:
        """Check if a question already has memories stored in ChromaDB.

        memo-bench port: each user has two collections, retrieval `<name>` and consolidation `<name>__consol`
        (vendor/Memora/src/memora/db_clients/base.py: CONSOLIDATION_SUFFIX). A store counts as built only if the
        retrieval collection is non-empty AND the consolidation collection exists; a single-collection store
        (built before the port) is rebuilt."""
        import chromadb
        from memora.db_clients.base import CONSOLIDATION_SUFFIX

        user_id = f"question_{question_id}"
        try:
            persist_path = self.cfg.memory.persist_path
            base_collection_name = self.cfg.memory.collection_name
            collection_name = f"{base_collection_name}_{user_id}"

            chroma_client = chromadb.PersistentClient(path=persist_path)

            try:
                collection = chroma_client.get_collection(name=collection_name)
                count = collection.count()
                chroma_client.get_collection(name=f"{collection_name}{CONSOLIDATION_SUFFIX}")
                return count > 0
            except Exception:
                return False

        except Exception:
            return False

    def load_data(self):
        with open(self.data_path, "r") as f:
            data = json.load(f)

        subset_idx = self.cfg.eval.subset_idx
        logger.info(f"subset_idx value: {subset_idx} (type: {type(subset_idx).__name__})")

        # Parse subset_idx: supports -1 (all), single number, range "1:10", or comma-separated "1,5,10"
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
                logger.info(f"Loading {len(data)} specific questions: {question_indices}")
            else:
                # Single number as string
                idx_val = int(subset_idx)
                if idx_val > 0:
                    data = data[idx_val - 1 : idx_val]
                    logger.info(f"Loading single question {idx_val}")
        elif isinstance(subset_idx, int) and subset_idx > 0:
            data = data[subset_idx - 1 : subset_idx]
            logger.info(f"Loading single question {subset_idx}")
        else:
            logger.info(f"Loading all {len(data)} questions")

        # Filter out questions that already have memories if skip_existing_memories is enabled
        if self.skip_existing_memories:
            logger.info("Checking for existing memories in ChromaDB...")
            original_count = len(data)
            filtered_data = []
            skipped_questions = []

            for item in data:
                question_id = item.get('question_id')
                if self.check_question_has_memories(question_id):
                    skipped_questions.append(question_id)
                else:
                    filtered_data.append(item)

            data = filtered_data
            filtered_count = original_count - len(data)
            logger.info(f"Skipped {filtered_count} questions with existing memories: {skipped_questions}")
            logger.info(f"{len(data)} questions remaining to process")

        self.data = data

        if self.cfg.general.debug:
            self.data = self.data[:1]

        logger.info(f"Loaded {len(self.data)} questions for processing.")

        return self.data

    def add_memory(self, client, user_id: str, chunks, retries=3):
        for messages, metadata in chunks:
            for attempt in range(retries):
                try:
                    _ = client.add(messages, type="chat", metadata=metadata)
                    return True
                
                except ContentFilterFinishReasonError as e:
                    logger.warning(
                        f"Content filter rejected memory for user {user_id}. Skipping."
                    )
                    return False
                
                except Exception as e:
                    if attempt < retries - 1:
                        time.sleep(1)
                        continue
                    else:
                        raise e
                
        return False

    def process_conversation(self, item, idx):
        """Process conversation for longmemeval format using SESSION-BASED segmentation."""
        
        question_id = item.get("question_id", "")
        user_id = f"question_{question_id}"
        set_meta(system=memora_system_name(self.cfg), conv_id=question_id, question_id=question_id)

        thread_id = threading.current_thread().name
        start_time = time.time()
        logger.info(f"\n[Thread {thread_id}] START Processing question ({idx+1}/{len(self.data)}): {question_id}")

        if "haystack_sessions" not in item:
            logger.warning(f"Item {idx} doesn't have 'haystack_sessions' - skipping")
            return
            
        
        client = self.get_memory_client(user_id)

        # Skip clear() if we're building fresh memories (force_rebuild or skip_existing)
        if not (self.cfg.memory.get('force_rebuild', False) or self.skip_existing_memories):
            client.clear()
        else:
            logger.info(f"[Thread {thread_id}] Skipping clear() - building fresh collection")
        
        haystack_sessions = item.get("haystack_sessions", [])
        haystack_dates = item.get("haystack_dates", [])
        haystack_session_ids = item.get("haystack_session_ids", [])
        
        num_sessions = len(haystack_sessions)
        total_messages = 0
        
        # Process each session as ONE segment
        for session_idx, (session, date, session_id) in enumerate(
            zip(haystack_sessions, haystack_dates, haystack_session_ids)
        ):
            logger.info(f"  Processing session {session_idx + 1}/{len(haystack_sessions)}: {session_id}")
            
            # Convert session to messages format
            messages = []

            # Parse date string to desired format
            try:
                timestamp = datetime.strptime(date, '%Y/%m/%d (%a) %H:%M')

                output_format = "%-I:%M %p on %-d %B, %Y"
                formatted_string = timestamp.strftime(output_format).lower()
                parts = formatted_string.split(' ')
                parts[4] = parts[4].capitalize()

                timestamp = ' '.join(parts)

            except Exception as e:
                print(f"    Warning: Failed to parse date '{date}': {e}, using original timestamp.")
                timestamp = date

            for turn_idx, turn in enumerate(session):
                role = turn['role']
                content = turn['content']
                
            
                # Add message
                message = {"role": role, "content": content}
                messages.append(message)

            print(f"Total messages in session {session_idx + 1}: {len(messages)}")

            chunks = []

            # Chunk messages if they exceed batch_size
            if len(messages) > self.batch_size:
                for i in range(0, len(messages), self.batch_size):
                    chunk = messages[i : i + self.batch_size]
                    metadata = {
                        "timestamp": timestamp,
                        "segment_index": str(session_id) + "_" + str(i),
                    }
                    chunks.append((chunk, metadata))
            else:
                metadata = {
                    "timestamp": timestamp,
                    "segment_index": str(session_id),
                }
                chunks.append((messages, metadata))
    

            # Add memory for this session
            try:
                self.add_memory(client, user_id, chunks)

            except Exception as e:
                logger.error(f"Error adding session {session_idx} for question {question_id}: {e}")
                continue
        
        # Print summary
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"[Thread {thread_id}] Question {question_id} Summary:")
        print(f"  Total sessions (segments): {num_sessions}")
        print(f"  Total messages: {total_messages}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"{'='*60}\n")
                
    def process_all_conversations(self):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        
        print(f"Total questions to process: {len(self.data)}")
        
        # Process questions in parallel with progress tracking
        with ThreadPoolExecutor(max_workers=self.cfg.eval.max_workers) as executor:
            futures = {
                executor.submit(self.process_conversation, item, idx): (item, idx)
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

            # Summary
            print(f"\n{'='*60}")
            print(f"Processing Complete:")
            print(f"  Successful: {completed}/{len(self.data)}")
            print(f"  Failed: {len(failed)}")
            if failed:
                print(f"  Failed questions: {[qid for qid, _ in failed]}")
            print(f"{'='*60}\n")
