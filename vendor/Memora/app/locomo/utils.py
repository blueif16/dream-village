# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
import json

import requests
from requests.exceptions import RequestException
from omegaconf import DictConfig

TECHNIQUES = ["mem0", "rag", "langmem", "zep", "openai"]

METHODS = ["add", "search"]


def init_mem0_client(cfg: DictConfig) -> "Memory":
    """
    Initialize a mem0 Memory client using Azure OpenAI configuration.
    Args:
        cfg: OmegaConf DictConfig with OpenAI settings
    Returns:
        Memory client instance
    """
    from mem0 import Memory  # lazy: mem0 is not installed for the Claude port

    azure_endpoint = cfg.openai.llm_api_base
    managed_identity = cfg.openai.managed_identity
    model = cfg.llm.model
    embedding_model = cfg.openai.embedding_model

    # Build config for LLM and embedder using Azure OpenAI
    config = {
        "llm": {
            "provider": "azure_openai",
            "config": {
                "model": model,
                "temperature": 0.1,
                "max_tokens": 2000,
                # All settings inline, no env vars
                "azure_kwargs": {
                    "azure_endpoint": azure_endpoint,
                    "api_version": "2024-12-01-preview",
                    # Only if using user-assigned MI (otherwise omit):
                    "azure_client_id": managed_identity,
                },
            },
        },
        "embedder": {
            "provider": "azure_openai",
            "config": {
                "model": embedding_model,
                "azure_kwargs": {
                    "azure_endpoint": azure_endpoint,
                    "api_version": "2024-12-01-preview",
                    # Same note: include client_id only for user-assigned MI
                    "azure_client_id": managed_identity,
                },
            },
        },
        # vector store
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "agent_memory",
                "path": cfg.memory.persist_path,
                # Optional: ChromaDB Cloud configuration
                # "api_key": "your-chroma-cloud-api-key",
                # "tenant": "your-chroma-cloud-tenant-id",
            },
        },
    }

    client = Memory.from_config(config)
    return client


def load_data(data_path: str, subset_idx: int = -1) -> list:

    with open(data_path, "r") as f:
        data = json.load(f)

    if subset_idx > 0:
        data = data[subset_idx - 1 : subset_idx]

    return data


def select_conversations(data: list, cfg: DictConfig) -> list:
    """[(idx, item)] to run. eval.conversations: 1-based conversation numbers (like eval.subset_idx), null = all.
    idx stays the position in `data`, so memory-store user ids do not depend on the selection."""
    wanted = cfg.eval.get("conversations", None)
    pairs = list(enumerate(data))
    if wanted:
        wanted = {int(c) for c in wanted}
        pairs = [(idx, item) for idx, item in pairs if idx + 1 in wanted]
    return pairs


def select_questions(qa: list, cfg: DictConfig) -> list:
    """[(q_idx, qa_item)] to answer: category 5 is never answered (evals.py drops it at judging anyway);
    eval.max_questions caps the rest per conversation (null = all). q_idx is the position in the original qa list."""
    pairs = [(q_idx, q) for q_idx, q in enumerate(qa) if str(q.get("category")) != "5"]
    max_questions = cfg.eval.get("max_questions", None)
    if max_questions is not None:
        pairs = pairs[: int(max_questions)]
    return pairs


def session_count(conversation: dict, cfg: DictConfig) -> int:
    """Number of sessions to use from a conversation, capped by memory.max_sessions (null = all)."""
    num_sessions = get_session_num(conversation)
    max_sessions = cfg.memory.get("max_sessions", None)
    if max_sessions is not None:
        num_sessions = min(num_sessions, int(max_sessions))
    return num_sessions


def conversation_id(item: dict, idx: int) -> str:
    return item.get("sample_id", f"conversation_{idx}")


def memora_system_name(cfg: DictConfig, retrieval_strategy: str = None) -> str:
    """Ledger/log system name: memora-noabs for the no-abstraction ablation, else memora-s / memora-p."""
    if cfg.memory.get("embed_field", "index") == "value":
        return "memora-noabs"
    strategy = retrieval_strategy or cfg.retrieval.strategy
    return {"semantic": "memora-s", "prompt": "memora-p"}.get(strategy, f"memora-{strategy}")


def generate_debug_data(data, conversion_idx=0, session_idx=1, num_sessions=3) -> list:
    """
    Generate a smaller subset of the data for debugging purposes.

    Args:
        data (list): Original dataset
        conversion_idx (int, optional). Defaults to 0.
        session_idx (int, optional). Defaults to 1.
        num_sessions (int, optional). Defaults to 1.

    Returns:
        list: Subset of the original data for debugging
    """
    data = data[: conversion_idx + 1]

    # first 10 questions
    data[0]["qa"] = data[0]["qa"][:10]

    # first num_sessions conversations
    conversation = data[0]["conversation"]
    new_conversation = {
        "speaker_a": conversation["speaker_a"],
        "speaker_b": conversation["speaker_b"],
    }

    # parse the indices
    if isinstance(session_idx, int):
        indices = [session_idx]
    elif isinstance(session_idx, str):
        indices = [int(idx) for idx in session_idx.split(",")]
    else:
        raise ValueError("session_idx must be int or comma-separated string")

    for idx in indices:
        for i in range(idx, idx + num_sessions):
            key = f"session_{i}"
            if key in conversation:
                new_conversation[key] = conversation[key]
                new_conversation[f"{key}_date_time"] = conversation[f"{key}_date_time"]
    data[0]["conversation"] = new_conversation
    return data


def measure_execution_time(func, *args, **kwargs):
    """
    Utility function to measure execution time of a function.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        tuple: (result, duration_in_seconds)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    duration = end_time - start_time
    return result, duration


def format_duration(duration: float) -> str:
    """
    Format duration in a simple format showing only seconds.
    
    Args:
        duration: Duration in seconds
        
    Returns:
        Formatted duration string in seconds
    """
    return f"{duration:.2f} seconds"


def get_session_num(conversation):
    num_sessions = 0
    for idx in range(1, 100):
        key = f"session_{idx}"
        if key not in conversation:
            break
        num_sessions += 1
    return num_sessions


def is_valid_image_url(url: str, timeout: int = 5) -> bool:
    """
    Check if an image URL is accessible.
    
    Args:
        url: The image URL to validate
        timeout: Request timeout in seconds
        
    Returns:
        bool: True if the URL is accessible, False otherwise
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Try HEAD request first (faster)
        response = requests.head(url, timeout=timeout, allow_redirects=True, headers=headers)
        
        # Some servers don't support HEAD, try GET if HEAD fails
        if response.status_code == 405 or response.status_code == 403:
            response = requests.get(url, timeout=timeout, allow_redirects=True, headers=headers, stream=True)
            # Close connection immediately after getting headers
            response.close()
        
        # Check if status code is successful
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            # Check if content type is an image or if URL ends with common image extensions
            is_image_type = content_type.startswith('image/')
            is_image_extension = url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))
            return is_image_type or is_image_extension
        return False
    except RequestException as e:
        print(f"Error validating URL {url}: {e}")
        return False
