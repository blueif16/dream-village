# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Memora Quick Start Example

Demonstrates basic usage of MemoraClient for adding and querying memory.
"""

import os
from omegaconf import OmegaConf

from memora.memora_client import MemoraClient


def make_config():
    """Build a minimal config for local MemoraClient usage.
    
    Supports two modes via OPENAI_API_TYPE env var:
      - "azure" (default): Uses Azure OpenAI with Managed Identity
      - "openai": Uses standard OpenAI API with OPENAI_API_KEY
    """
    api_type = os.getenv("OPENAI_API_TYPE", "azure")

    cfg = OmegaConf.create({
        "llm": {
            "model": "gpt-4.1-mini",
            "seed": 42,
        },
        "openai": {
            "api_type": api_type,
            # Azure OpenAI settings (used when api_type="azure")
            "llm_api_base": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            "llm_api_version": "2024-12-01-preview",
            "embedding_api_base": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            "embedding_api_version": "2024-12-01-preview",
            "embedding_deployment_name": "text-embedding-3-small",
            "managed_identity": os.getenv("AZURE_MANAGED_IDENTITY_CLIENT_ID", None),
            # Standard OpenAI settings (used when api_type="openai")
            "api_key": os.getenv("OPENAI_API_KEY", None),
            # Shared settings
            "embedding_model": "text-embedding-3-small",
            "model": "gpt-4.1-mini",
        },
        "memory": {
            "memory_store": "example_store",
            "persist_path": "./memory_store/example_store",
            "collection_name": "agent_memory",
            "distance": "cosine",
            "query_score_threshold": 0.4,
            "update_score_threshold": 0.8,
            "force_rebuild": False,
            "enhance_query": False,
            "return_history": True,
            "multimodal_support": False,
            "top_k": 10,
            "cue_top_k": 10,
            "enable_hybrid_search": False,
            "enable_segmentation": False,
            "enable_episodic_memory": False,
            "use_segments_as_episodic": False,
            "enable_cue_index": True,
        },
        "retrieval": {
            "strategy": "semantic",
        },
        "eval": {
            "max_workers": 5,
        },
    })
    return cfg


def main():
    cfg = make_config()
    user_id = "demo_user"

    # Initialize client
    memory_client = MemoraClient(cfg=cfg, user_id=user_id)

    # --- Add memories ---
    context = (
        "Alice mentioned she's moving to Seattle next month for a new job at Microsoft. "
        "She's excited about the role but nervous about the rain."
    )
    print("Adding memory...")
    entries = memory_client.add(context, type="doc")
    print(f"  Stored {len(entries)} memory entries")

    # Add more context
    memory_client.add(
        "Alice said her favorite programming language is Python and she loves hiking on weekends.",
        type="doc",
    )

    # --- Query memories (semantic) ---
    question = "Where is Alice moving and why?"
    print(f"\nQuerying: {question}")
    results = memory_client.query(question, top_k=5)
    for i, entry in enumerate(results, 1):
        print(f"  [{i}] {entry.index}: {entry.value}")

    # --- Advanced query with prompted policy ---
    print(f"\nAdvanced query (prompted policy): {question}")
    results = memory_client.advance_query(question, query_type="prompt", top_k=5)
    for i, entry in enumerate(results, 1):
        print(f"  [{i}] {entry.index}: {entry.value}")

    # --- List stored memories ---
    print("\nAll stored memories:")
    all_memories = memory_client.list_memories(limit=20)
    for i, mem in enumerate(all_memories, 1):
        print(f"  [{i}] {mem.index}: {mem.value}")


if __name__ == "__main__":
    main()
