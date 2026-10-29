# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from omegaconf import DictConfig
from tqdm import tqdm

from memora.memora_client import MemoraClient
from memobench.context import get_meta, set_meta
from utils import (
    conversation_id, generate_debug_data, is_valid_image_url, load_data, memora_system_name, select_conversations,
    session_count,
)
from providers.memora.segmentation import ConversationSegmenter

logger = logging.getLogger(__name__)

class MemoraADD:

    def __init__(
        self,
        cfg: DictConfig,
        data_path=None,
        batch_size=2,
        data_config=None,
    ):

        self.cfg = cfg
        self.data_config = data_config

        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        if data_path:
            self.load_data()

        # cache of memory clients
        self.cached_clients = {}
        
        # Always create segmenter - it will handle both LLM and batch-based modes
        self.segmenter = ConversationSegmenter(cfg)

        # Per-segment build stats log
        self.build_log = []

    def get_memory_client(self, user_id: str):
        if user_id in self.cached_clients:
            return self.cached_clients[user_id]

        client = MemoraClient(self.cfg, user_id=user_id)

        # add to cache
        self.cached_clients[user_id] = client

        return client

    def load_data(self):

        self.data = load_data(self.data_path, subset_idx=self.cfg.eval.subset_idx)

        if self.data_config:
            self.data = generate_debug_data(
                self.data,
                self.data_config["conversion_idx"],
                self.data_config["session_idx"],
                self.data_config["num_sessions"],
            )
        elif self.cfg.general.debug:
            self.data = generate_debug_data(self.data)

        return self.data

    def add_memory(self, user_id, message, metadata, retries=3):

        client = self.get_memory_client(user_id)
        count_before = client.count()
        start_time = time.time()
        for attempt in range(retries):
            try:
                _ = client.add(message, type="chat", metadata=metadata)
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)  # Wait before retrying
                    continue
                else:
                    raise e
        elapsed = time.time() - start_time
        count_after = client.count()
        build_stats = client.get_last_build_stats() or {}

        self.build_log.append({
            "user_id": user_id,
            "timestamp": metadata.get("timestamp", ""),
            "segment_topic": metadata.get("segment_topic", ""),
            "segment_index": metadata.get("segment_index", -1),
            "wall_time": round(elapsed, 3),
            "memory_count_before": count_before,
            "memory_count_after": count_after,
            **build_stats,
        })

    def add_memories_for_speaker(self, speaker, chunks, desc):
        """Add memories from chunks (either segments or batches).
        
        Args:
            speaker: User ID for the speaker
            chunks: List of (messages, metadata) tuples
            desc: Description for progress bar
        """
        for messages, metadata in tqdm(chunks, desc=desc):
            self.add_memory(speaker, messages, metadata=metadata)

    def process_conversation(self, item, idx):
        set_meta(system=memora_system_name(self.cfg), conv_id=conversation_id(item, idx), question_id=None)

        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        # Check if we should use combined user mode
        use_combined_user = self.cfg.eval.get("use_combined_user", False)

        if use_combined_user:
            # Combined user mode: single user ID for both speakers
            combined_user_id = f"{speaker_a}_{speaker_b}_{idx}"
            client_combined = self.get_memory_client(combined_user_id)
            client_combined.clear()
        else:
            # Original mode: separate user IDs for each speaker
            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            client_speaker_a = self.get_memory_client(speaker_a_user_id)
            client_speaker_b = self.get_memory_client(speaker_b_user_id)

            # delete all memories for the two users
            client_speaker_a.clear()
            client_speaker_b.clear()

        num_sessions = session_count(conversation, self.cfg)  # memory.max_sessions caps it for smoke tests
        for idx in range(1, num_sessions + 1):

            print("=" * 60, f" Processing session {idx}/{num_sessions} ", "=" * 60)
            key = f"session_{idx}"

            date_time_key = key + "_date_time"
            timestamp = conversation[date_time_key]
            if timestamp is None or timestamp == "":
                print("No timestamp found, using current time")
            chats = conversation[key]

            messages = []
            messages_reverse = [] if not use_combined_user else None
            for chat in chats:
                text_content = chat["text"]
                speaker = chat["speaker"]

                if "blip_caption" in chat:
                    image_description = chat["blip_caption"]
                    if "query" in chat:
                        image_description += f", {chat['query']}"
                    text_content = text_content + f" [Image Description: {image_description}]"

                if "img_url" in chat:
                    image_url = chat["img_url"]
                    if isinstance(image_url, list):
                        if len(image_url) > 0:
                            image_url = image_url[0]
                        else:
                            image_url = None

                    # Validate the image URL before adding it
                    if image_url and is_valid_image_url(image_url):
                        message = [
                            {"type": "text", "text": f"{speaker}: {text_content}"},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    else:
                        # Fall back to text-only message if URL is invalid or missing
                        message = f"{speaker}: {text_content}"
                else:
                    message = f"{speaker}: {text_content}"

                if use_combined_user:
                    # Combined mode: all messages are user messages
                    messages.append({"role": "user", "content": message})
                else:
                    # Original mode: alternate roles based on speaker
                    if chat["speaker"] == speaker_a:
                        messages.append({"role": "user", "content": message})
                        messages_reverse.append({"role": "assistant", "content": message})
                    elif chat["speaker"] == speaker_b:
                        messages.append({"role": "assistant", "content": message})
                        messages_reverse.append({"role": "user", "content": message})
                    else:
                        raise ValueError(f"Unknown speaker: {chat['speaker']}")

            # Create chunks using segmenter (handles both LLM and batch modes with fallback)
            logger.info(f"Processing session {idx} with {len(messages)} messages")
            segments = self.segmenter.segment_conversation(messages)
            logger.info(f"Created {len(segments)} segments")
            
            # Create chunks from segments
            if use_combined_user:
                # Combined mode: single set of chunks
                chunks_combined = []
                for seg_idx, segment in enumerate(segments):
                    segment_messages = [messages[i] for i in segment["indices"]]
                    
                    metadata = {
                        "timestamp": timestamp,
                        "segment_topic": segment.get("topic", "conversation"),
                        "segment_index": seg_idx
                    }
                    
                    chunks_combined.append((segment_messages, metadata))
                
                # Add memories for combined user
                self.add_memories_for_speaker(
                    combined_user_id, chunks_combined, "Adding Memories for Combined User"
                )
            else:
                # Original mode: separate chunks for each speaker
                chunks_a = []
                chunks_b = []
                for seg_idx, segment in enumerate(segments):
                    segment_messages = [messages[i] for i in segment["indices"]]
                    segment_messages_reverse = [messages_reverse[i] for i in segment["indices"]]
                    
                    metadata = {
                        "timestamp": timestamp,
                        "segment_topic": segment.get("topic", "conversation"),
                        "segment_index": seg_idx
                    }
                    
                    chunks_a.append((segment_messages, metadata))
                    chunks_b.append((segment_messages_reverse, metadata))
                
                # Add memories using chunks with threading (call metadata is per-thread: hand it over)
                meta = get_meta()

                def add_with_meta(*args):
                    set_meta(**meta)
                    self.add_memories_for_speaker(*args)

                thread_a = threading.Thread(
                    target=add_with_meta,
                    args=(speaker_a_user_id, chunks_a, "Adding Memories for Speaker A"),
                )
                thread_b = threading.Thread(
                    target=add_with_meta,
                    args=(speaker_b_user_id, chunks_b, "Adding Memories for Speaker B"),
                )

                thread_a.start()
                thread_b.start()
                thread_a.join()
                thread_b.join()

        print("Messages added successfully")

    def process_all_conversations(self):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        with ThreadPoolExecutor(max_workers=self.cfg.eval.max_workers) as executor:
            futures = [
                executor.submit(self.process_conversation, item, idx)
                for idx, item in select_conversations(self.data, self.cfg)  # eval.conversations (1-based)
            ]

            for future in futures:
                future.result()
