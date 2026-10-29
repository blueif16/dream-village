# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
import json

import requests
from requests.exceptions import RequestException
from omegaconf import DictConfig

METHODS = ["add", "search"]



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


def load_data(data_path: str, subset_idx: int = -1) -> list:

    with open(data_path, "r") as f:
        data = json.load(f)

    if subset_idx > 0:
        data = data[subset_idx - 1 : subset_idx]

    return data


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


def get_session_num(conversation):
    num_sessions = 0
    for idx in range(1, 100):
        key = f"session_{idx}"
        if key not in conversation:
            break
        num_sessions += 1
    return num_sessions


def memora_system_name(cfg: DictConfig, retrieval_strategy: str = None) -> str:
    """Ledger/log system name: memora-noabs for the no-abstraction ablation, else memora-s / memora-p."""
    if cfg.memory.get("embed_field", "index") == "value":
        return "memora-noabs"
    strategy = retrieval_strategy or cfg.retrieval.strategy
    return {"semantic": "memora-s", "prompt": "memora-p"}.get(strategy, f"memora-{strategy}")


def select_questions(data: list, cfg: DictConfig) -> list:
    """eval.subset_idx / general.debug selection, as providers/memora/search.py (load_data) applies it:
    -1 (all), a 1-based index, a 1-based inclusive range "1:10", or a comma-separated list "1,5,10"."""
    subset_idx = cfg.eval.subset_idx
    if isinstance(subset_idx, str):
        if ":" in subset_idx:
            start_str, end_str = subset_idx.split(":")
            start_idx, end_idx = int(start_str), int(end_str)
            if start_idx > 0 and end_idx > 0:
                data = data[start_idx - 1 : end_idx]
        elif "," in subset_idx:
            question_indices = [int(idx.strip()) for idx in subset_idx.split(",")]
            data = [data[idx - 1] for idx in question_indices if 0 < idx <= len(data)]
        else:
            idx_val = int(subset_idx)
            if idx_val > 0:
                data = data[idx_val - 1 : idx_val]
    elif isinstance(subset_idx, int) and subset_idx > 0:
        data = data[subset_idx - 1 : subset_idx]
    if cfg.general.debug:
        data = data[:1]
    return data


def render_history(item: dict) -> str:
    """The whole haystack of one question for the baselines: every session in dataset order, one line per turn,
    "{session date} | {role}: {content}\\n" (the LoCoMo baselines' clean_chat_history format with the session date
    as the timestamp, app/locomo/providers/full_context.py). Turn labels such as has_answer are not rendered."""
    return "".join(
        f"{date} | {turn['role']}: {turn['content']}\n"
        for session, date in zip(item["haystack_sessions"], item["haystack_dates"])
        for turn in session
    )
