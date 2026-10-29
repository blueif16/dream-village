# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Latency tracking utilities for measuring memory system performance.

This module provides tools for tracking and analyzing latency across different
components of the memory retrieval and answer generation pipeline.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import tiktoken

logger = logging.getLogger(__name__)


class LatencyTracker:
    """
    Tracks latency measurements across the memory retrieval pipeline.

    This class provides a hierarchical structure for tracking timing information
    at different stages of memory retrieval and answer generation:

    - Search latency (broken down by component)
    - Retrieval steps (for policy-based retrievers)
    - Memory formatting time
    - LLM generation time
    - Prompt statistics

    Example:
        tracker = LatencyTracker()
        with tracker.track("search_primary"):
            # perform primary search
            pass
        with tracker.track("search_cue"):
            # perform cue search
            pass

        results = tracker.get_summary()
        print(f"Total search time: {results['total_search_time']:.3f}s")
    """

    def __init__(self):
        """Initialize the latency tracker."""
        # Raw timing records
        self._timings: Dict[str, List[float]] = {}

        # Hierarchical structure for retrieval steps
        self._retrieval_steps: List[Dict[str, Any]] = []

        # Prompt statistics
        self._prompt_stats: Dict[str, Any] = {}

        # Overall search time
        self._overall_search_start: Optional[float] = None
        self._overall_search_time: Optional[float] = None

    @contextmanager
    def track(self, operation: str):
        """
        Context manager for tracking operation timing.

        Args:
            operation: Name of the operation to track

        Example:
            with tracker.track("search_primary"):
                results = memory.query(...)
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            if operation not in self._timings:
                self._timings[operation] = []
            self._timings[operation].append(elapsed)
            logger.debug(f"[LatencyTracker] {operation}: {elapsed:.4f}s")

    def start_overall_search(self):
        """Mark the start of overall search timing."""
        self._overall_search_start = time.time()

    def end_overall_search(self):
        """Mark the end of overall search timing."""
        if self._overall_search_start is not None:
            self._overall_search_time = time.time() - self._overall_search_start
            logger.debug(f"[LatencyTracker] Overall search: {self._overall_search_time:.4f}s")

    def add_timing(self, operation: str, duration: float):
        """
        Manually add a timing record.

        Args:
            operation: Name of the operation
            duration: Duration in seconds
        """
        if operation not in self._timings:
            self._timings[operation] = []
        self._timings[operation].append(duration)

    def add_retrieval_step(self, step_data: Dict[str, Any]):
        """
        Add a retrieval step record (for policy-based retrievers).

        Args:
            step_data: Dictionary containing step information:
                - step: step number
                - action: action type (INIT_RETRIEVE, EXPAND, RE_QUERY, STOP)
                - duration: time taken for this step
                - query: query string (optional)
                - memories_count: number of memories retrieved (optional)
                - other metadata
        """
        self._retrieval_steps.append(step_data)

    def set_prompt_stats(self, stats: Dict[str, Any]):
        """
        Set prompt statistics.

        Args:
            stats: Dictionary containing prompt statistics:
                - total_tokens: total token count of all memories
                - num_memories: number of memories in prompt
                - avg_tokens: average tokens per memory
                - num_images: number of images in prompt (if multimodal)
        """
        self._prompt_stats = stats

    def get_timing(self, operation: str) -> float:
        """
        Get the total time for an operation.

        Args:
            operation: Name of the operation

        Returns:
            Total time in seconds (sum of all occurrences)
        """
        return sum(self._timings.get(operation, []))

    def get_timing_list(self, operation: str) -> List[float]:
        """
        Get all timing records for an operation.

        Args:
            operation: Name of the operation

        Returns:
            List of timing measurements in seconds
        """
        return self._timings.get(operation, [])

    def get_search_breakdown(self) -> Dict[str, float]:
        """
        Get detailed breakdown of search timing components.

        Returns:
            Dictionary with search timing breakdown:
                - search_primary: time for primary index search
                - search_cue: time for cue index search
                - search_hybrid: time for hybrid (BM25/keyword) search
                - search_rrf_merge: time for RRF merging
                - search_llm_filter: time for LLM filtering
                - search_other: other search-related timing
        """
        breakdown = {}

        # Extract search-related timings
        search_keys = [
            "search_primary",
            "search_cue",
            "search_hybrid",
            "search_rrf_merge",
            "search_llm_filter",
            "search_keyword_extract",
            "search_bm25_index_build",
        ]

        for key in search_keys:
            if key in self._timings:
                breakdown[key] = sum(self._timings[key])

        return breakdown

    def get_retrieval_steps_summary(self) -> Dict[str, Any]:
        """
        Get summary of retrieval steps (for policy-based retrievers).

        Returns:
            Dictionary with retrieval step information:
                - num_steps: total number of steps
                - steps: list of step records
                - total_step_time: sum of all step durations
        """
        if not self._retrieval_steps:
            return {}

        total_step_time = sum(
            step.get("duration", 0)
            for step in self._retrieval_steps
        )

        return {
            "num_steps": len(self._retrieval_steps),
            "steps": self._retrieval_steps,
            "total_step_time": round(total_step_time, 4),
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive latency summary.

        Returns:
            Dictionary with complete latency breakdown:
                - total_search_time: overall search time
                - search_breakdown: detailed search component times
                - retrieval_steps: step-by-step retrieval info (if applicable)
                - format_time: memory formatting time
                - llm_time: LLM generation time
                - prompt_stats: prompt statistics
                - total_time: end-to-end time
        """
        summary = {}

        # Overall search time
        if self._overall_search_time is not None:
            summary["total_search_time"] = round(self._overall_search_time, 4)

        # Search breakdown
        search_breakdown = self.get_search_breakdown()
        if search_breakdown:
            summary["search_breakdown"] = {
                k: round(v, 4) for k, v in search_breakdown.items()
            }

            # Calculate percentage breakdown if we have overall time
            if self._overall_search_time and self._overall_search_time > 0:
                total_accounted = sum(search_breakdown.values())
                summary["search_breakdown_pct"] = {
                    k: round(v / self._overall_search_time * 100, 2)
                    for k, v in search_breakdown.items()
                }

                # Add unaccounted time
                unaccounted = self._overall_search_time - total_accounted
                if unaccounted > 0.001:  # more than 1ms
                    summary["search_breakdown"]["search_unaccounted"] = round(unaccounted, 4)
                    summary["search_breakdown_pct"]["search_unaccounted"] = round(
                        unaccounted / self._overall_search_time * 100, 2
                    )

        # Retrieval steps (for policy retrievers)
        steps_summary = self.get_retrieval_steps_summary()
        if steps_summary:
            summary["retrieval_steps"] = steps_summary

        # Memory formatting time
        format_time = self.get_timing("format_memories")
        if format_time > 0:
            summary["format_time"] = round(format_time, 4)

        # LLM generation time
        llm_time = self.get_timing("llm_generation")
        if llm_time > 0:
            summary["llm_time"] = round(llm_time, 4)

        # Prompt statistics
        if self._prompt_stats:
            summary["prompt_stats"] = self._prompt_stats

        # Calculate total time
        total_time = 0.0
        if self._overall_search_time:
            total_time += self._overall_search_time
        if format_time > 0:
            total_time += format_time
        if llm_time > 0:
            total_time += llm_time

        if total_time > 0:
            summary["total_time"] = round(total_time, 4)

        return summary

    def log_summary(self, level: int = logging.INFO):
        """
        Log the latency summary.

        Args:
            level: Logging level (default: INFO)
        """
        summary = self.get_summary()

        logger.log(level, "=" * 60)
        logger.log(level, "Latency Summary")
        logger.log(level, "=" * 60)

        if "total_search_time" in summary:
            logger.log(level, f"Total Search Time: {summary['total_search_time']:.4f}s")

        if "search_breakdown" in summary:
            logger.log(level, "\nSearch Breakdown:")
            for key, value in summary["search_breakdown"].items():
                pct = summary.get("search_breakdown_pct", {}).get(key, 0)
                logger.log(level, f"  {key}: {value:.4f}s ({pct:.1f}%)")

        if "retrieval_steps" in summary:
            steps_info = summary["retrieval_steps"]
            logger.log(level, f"\nRetrieval Steps: {steps_info['num_steps']}")
            logger.log(level, f"Total Step Time: {steps_info['total_step_time']:.4f}s")
            for step in steps_info["steps"]:
                action = step.get("action", "UNKNOWN")
                duration = step.get("duration", 0)
                step_num = step.get("step", "?")
                logger.log(level, f"  Step {step_num} ({action}): {duration:.4f}s")

        if "format_time" in summary:
            logger.log(level, f"\nFormat Time: {summary['format_time']:.4f}s")

        if "llm_time" in summary:
            logger.log(level, f"LLM Generation Time: {summary['llm_time']:.4f}s")

        if "prompt_stats" in summary:
            logger.log(level, "\nPrompt Statistics:")
            for key, value in summary["prompt_stats"].items():
                logger.log(level, f"  {key}: {value}")

        if "total_time" in summary:
            logger.log(level, f"\nTotal Time: {summary['total_time']:.4f}s")

        logger.log(level, "=" * 60)


# Cache tokenizer encoding
_ENCODING_CACHE = {}


def get_encoding(model: str = "gpt-4o"):
    """
    Get or create cached tokenizer encoding.

    Args:
        model: Model name for tokenizer (default: gpt-4o)

    Returns:
        tiktoken Encoding object
    """
    if model not in _ENCODING_CACHE:
        try:
            _ENCODING_CACHE[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base encoding (used by GPT-4, GPT-3.5-turbo)
            _ENCODING_CACHE[model] = tiktoken.get_encoding("cl100k_base")

    return _ENCODING_CACHE[model]


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count tokens in a text string using tiktoken.

    Args:
        text: Input text
        model: Model name for tokenizer (default: gpt-4o)

    Returns:
        Exact token count
    """
    encoding = get_encoding(model)
    return len(encoding.encode(text))


def count_memories_tokens(memories: List[str], model: str = "gpt-4o") -> Dict[str, int]:
    """
    Count tokens in a list of memory strings.

    Args:
        memories: List of memory strings
        model: Model name for tokenizer (default: gpt-4o)

    Returns:
        Dictionary with token counts:
            - total_tokens: total tokens across all memories
            - num_memories: number of memories
            - avg_tokens: average tokens per memory
    """
    encoding = get_encoding(model)
    total_tokens = 0

    for memory in memories:
        total_tokens += len(encoding.encode(memory))

    avg_tokens = total_tokens / len(memories) if memories else 0

    return {
        "total_tokens": total_tokens,
        "num_memories": len(memories),
        "avg_tokens": round(avg_tokens, 2),
    }
