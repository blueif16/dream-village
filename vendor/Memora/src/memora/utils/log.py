# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
from typing import Any, Dict, Optional

from memora.core.memory_entry import MemoryEntry
from memora.core.segment import Segment

# Initialize module logger
logger = logging.getLogger(__name__)

def log_segments(segments: Segment) -> None:
    """
    Log details of each segment for debugging and monitoring.

    Args:
        segments: List of segments to log
    """
    logger.info("### Logging segments:")
    content = ""
    for segment in segments:
        content += (segment.metadata["heading_path"]) + "\n"
    logger.info(f"\n{content}")
    
def configure_logging(log_level: str = "INFO"):
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # Force reconfiguration even if already configured
    )
    # Also explicitly set the root logger level
    logging.getLogger().setLevel(numeric_level)
    
    # Keep your own logs; silence noisy libs
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    logging.getLogger("azure.identity").setLevel(logging.WARNING)
    logging.getLogger("azure.identity._internal").setLevel(logging.WARNING)

    # httpx stack (used by many SDKs)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # ChromaDB logging
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.WARNING)


def log_memory_building(context: str, user_id: str) -> None:

    logger.info(
        f"\n" + "-" * 60 + "\n"
        f"Building Memory - [{user_id}]\n{context}"
        f"\n" + "-" * 60
    )


def log_memory_operation(
    operation_type: str,
    entry: MemoryEntry,
    user_id: str,
) -> None:
    """
    Log memory operations with consistent formatting for monitoring and debugging.

    This helper method centralizes logging logic to ensure consistent formatting
    across all memory operations. It provides detailed information about each
    operation including timestamps, user context, and operation details.

    The log format includes:
    - Decorative borders for visual separation
    - Operation type (Add, Update, etc.)
    - Timestamp and user identification
    - Memory index and value information
    - Optional additional context

    Args:
        operation_type: Type of operation being performed (Add, Update, Delete, etc.)
        entry: Memory entry object being processed
    """
    # Build structured log message with consistent formatting
    log_message = (
        "\n" + "-" * 60 + "\n"
        f"MEMORY STORE: {operation_type}|{entry.creation_time}|{user_id}\n"
        f"Index: {entry.index}\n"
        f"Value: {entry.value}\n"
        f"Timestamp: {entry.timestamp}\n"
        f"cue indices: {entry.cue_indices}\n"
    )

    log_message += "-" * 60

    logger.info(log_message)


# def memory_store_log_add(logger, index: str, value: str, metadata: dict):
#     logger = logging.getLogger(__name__)
#     logger.info(
#         "\n" + "=" * 60 + "\n"
#         f"MEMORY STORE: Add|{metadata.get('timestamp', 'N/A')}|{metadata.get('user_id', 'N/A')}\n"
#         f"Index: {index}\n"
#         f"Value: {value}\n" + "=" * 60
#     )


# def memory_store_log_update(
#     logger, old_index: str, new_index: str, new_value: str, metadata: dict
# ):
#     logger = logging.getLogger(__name__)
#     logger.info(
#         "\n" + "=" * 60 + "\n"
#         f"MEMORY STORE: Update|{metadata.get('timestamp', 'N/A')}|{metadata.get('user_id', 'N/A')}\n"
#         f"Index: {old_index} -> {new_index}\n"
#         f"Update Value: {new_value}\n" + "=" * 60
#     )
