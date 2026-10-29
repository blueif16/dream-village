# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Shared components for policy-based memory retrieval.

This module contains shared prompts, formatting functions, and utilities
used by both the prompted policy retriever and the RL-trained policy retriever.
"""

from typing import Dict, List, Any
import json
import logging
import os
from datetime import datetime

from memora.core.memory_entry import MemoryEntry

logger = logging.getLogger(__name__)

# ============================================================================
# POLICY ISSUE TRACKER
# ============================================================================

class PolicyIssueTracker:
    """Track and log policy decision issues for debugging."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.issues = {
            "expand_no_valid_ids": [],
            "requery_no_new_query": [],
            "unknown_action": [],
            "json_parse_error": [],
        }
        self._initialized = True
    
    def log_issue(self, issue_type: str, step: int = None, details: str = ""):
        """Log an issue with timestamp and optional step."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "details": details,
        }
        if issue_type in self.issues:
            self.issues[issue_type].append(entry)
    
    def save_to_file(self, output_dir: str):
        """Save issues to a JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, "policy_issues.json")
        
        summary = {
            "counts": {k: len(v) for k, v in self.issues.items()},
            "details": self.issues,
        }
        
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Policy issues saved to {filepath}")
    
    def get_summary(self) -> Dict[str, int]:
        """Get issue counts."""
        return {k: len(v) for k, v in self.issues.items()}
    
    def reset(self):
        """Reset all counters."""
        for key in self.issues:
            self.issues[key] = []


# Global tracker instance
policy_tracker = PolicyIssueTracker()


# ============================================================================
# SHARED PROMPT TEMPLATES
# ============================================================================

POLICY_SYSTEM_MESSAGE = """You are Memora-Control, a decision policy for an iterative memory system.

Initial retrieval has ALREADY been performed, yielding a Working Set (W) of memories and a Frontier (F) of expansion candidates.
Your job is to decide how to proceed using ONLY these actions:
- EXPAND: grow memory from expansion candidates in Frontier if they add useful information to answering the user question.
- RE_QUERY: regenerate a better query and retrieve again, if current memories are insufficient to answer the question, or important facts are missing.
- STOP: enough information has been collected in the Working Set (W) to answer the user question.

Definitions:
- Working Set (W): memories already collected
- Frontier (F): expansion candidates reachable from W
- Expansion is cheaper than re-querying the full corpus

Decision rules:
1) Choose STOP if W is sufficient to answer the user question with high confidence.
2) Choose EXPAND if frontier contains high-novelty items likely to fill remaining gaps.
3) Choose RE_QUERY if important facts are missing AND expansion is unlikely to find them.
4) Here are some typical scenarios for RE_QUERY:
    (a) Query refinement: The previous query failed to return relevant results, or the question requires a different semantic angle.
    (b) Relative answers: If W provides a relative answer (a "pointer") rather than a direct value. You need to RE-QUERY for the specific target.
        - *Example:*
            - Query: "Where did Mike go to college?"
            - W: "Mike went to the same college as Sarah"  (relative answer, no specific college - gap identified)
            - Action: RE_QUERY -> `new_query`: "Where did Sarah go to college?" 
        - *Example:*
            - Query: "When is the deadline for Project X?"
            - W: "The deadline is three months after the project kickoff." (relative answer, no specific date - gap identified)
            - Action: RE_QUERY -> `new_query`: "When was the kickoff date for Project X?"
5) Prefer EXPAND over RE_QUERY when both are viable.
6) Minimize redundancy and unnecessary steps.

Output STRICT JSON only. No extra text

JSON schema:
{
  "action": "EXPAND | RE_QUERY | STOP",
  "reason": "one sentence",
  "confidence": 0.0-1.0,
  "frontier_ids": ["id1","id2"],  
  "new_query": "string"           
}

Critical Field rules:
- frontier_ids: Required for EXPAND. Use the EXACT memory index strings shown in brackets in FrontierSummary (e.g., "Jolene's hobby" not "1" or "11"). Empty [] for RE_QUERY and STOP.
- new_query: Required for RE_QUERY (the reformulated search query), empty "" for EXPAND and STOP

Example - if FrontierSummary shows:
  - [User's favorite color]: The user likes blue...
  - [User's pet name]: The user has a cat named Whiskers...
Then frontier_ids should be: ["User's favorite color", "User's pet name"]
NOT: ["1", "2"] or ["11", "12"]
"""

POLICY_USER_MESSAGE_TEMPLATE = """UserQuestion:
{user_question}

CurrentQuery:
{current_query}

Step:
{step}/{max_steps}

WorkingSetSummary:
{W_summary}

FrontierSummary:
{F_summary}

Constraints:
- Prefer EXPAND over RE_QUERY when possible
- Avoid repeating information already covered in W
- RE_QUERY should be concise and retrieval-optimize
- Avoid RE_QUERY with the same query as before"""


# ============================================================================
# SHARED FORMATTING FUNCTIONS
# ============================================================================

def format_working_set(
    working_set: List[MemoryEntry], 
    max_value_length: int = 150
) -> str:
    """
    Format working set memories for prompt.
    
    Args:
        working_set: List of MemoryEntry objects
        max_value_length: Maximum characters to show for each memory value
    
    Returns:
        Formatted string representation
    """
    if not working_set:
        return "(empty)"
    
    lines = []
    for i, mem in enumerate(working_set):
        value = (mem.value or "")[:max_value_length]
        suffix = "..." if len(mem.value or "") > max_value_length else ""
        lines.append(f"[{i+1}] {mem.index}: {value}{suffix}")
    
    return "\n".join(lines)


def format_frontier(
    frontier: Dict[str, MemoryEntry], 
    max_value_length: int = 100
) -> str:
    """
    Format frontier candidates for prompt.
    
    Args:
        frontier: Dictionary mapping memory IDs to MemoryEntry objects
        max_value_length: Maximum characters to show for each memory value
    
    Returns:
        Formatted string representation
    """
    if not frontier:
        return "(empty - no expansion candidates)"
    
    lines = []
    for mem_id, mem in frontier.items():
        value = (mem.value or "")[:max_value_length]#
        suffix = "..." if len(mem.value or "") > max_value_length else ""
        lines.append(f"- [{mem_id}]: {value}{suffix}")
    
    return "\n".join(lines)


def format_user_message(
    user_question: str,
    current_query: str,
    working_set: List[MemoryEntry],
    frontier: Dict[str, MemoryEntry],
    step: int,
    max_steps: int,
) -> str:
    """
    Format the user message for policy prompt.
    
    Args:
        user_question: Original user question
        current_query: Current search query
        working_set: Current working set of memories
        frontier: Current frontier candidates
        step: Current step number
        max_steps: Maximum steps allowed
    
    Returns:
        Formatted user message string
    """
    return POLICY_USER_MESSAGE_TEMPLATE.format(
        user_question=user_question,
        current_query=current_query,
        step=step,
        max_steps=max_steps,
        W_summary=format_working_set(working_set),
        F_summary=format_frontier(frontier),
    )


# ============================================================================
# SHARED UTILITY FUNCTIONS
# ============================================================================

def select_from_frontier(
    frontier: Dict[str, MemoryEntry],
    frontier_ids: List[str],
) -> List[MemoryEntry]:
    """
    Select memories from frontier by IDs.
    
    Args:
        frontier: Dictionary mapping memory IDs to MemoryEntry objects
        frontier_ids: List of IDs to select
    
    Returns:
        List of selected MemoryEntry objects
    """
    if not frontier_ids:
        return []
    
    return [frontier[fid] for fid in frontier_ids if fid in frontier]


def parse_json_response(response_text: str, step: int = None) -> Dict[str, Any]:
    """
    Parse JSON from LLM response, handling markdown code blocks.
    
    Args:
        response_text: Raw response text from LLM
        step: Current training step (for logging)
    
    Returns:
        Parsed dictionary, or error dict if parsing fails
    """
    text = response_text.strip()
    
    # Handle markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    
    # Handle "JSON" prefix (model literally says "JSON" before the object)
    if text.upper().startswith("JSON"):
        text = text[4:]
    
    text = text.strip()
    
    # Find the JSON object boundaries
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx + 1]
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}. Response: {text[:150]}")
        policy_tracker.log_issue(
            "json_parse_error", 
            step=step, 
            details=f"Error: {e}, Response snippet: {text[:100]}"
        )
        return {
            "action": "STOP",
            "reason": "Failed to parse response",
            "confidence": 0.0,
            "_parse_error": str(e),
        }


def validate_policy_decision(
    decision: Dict[str, Any],
    frontier: Dict[str, MemoryEntry],
    step: int = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Validate and normalize a policy decision.
    
    Args:
        decision: Raw decision dict from LLM
        frontier: Current frontier for validation
        step: Current training step (for logging)
        verbose: Whether to print debug info
    
    Returns:
        Validated decision dict with normalized fields
    """
    action = decision.get("action", "STOP").upper()
    reason = decision.get("reason", "")
    confidence = float(decision.get("confidence", 0.0))
    
    result = {
        "action": action,
        "reason": reason,
        "confidence": confidence,
        "frontier_ids": [],
        "new_query": "",
    }
    
    if action == "EXPAND":
        frontier_ids = decision.get("frontier_ids", [])
        # Validate IDs exist in frontier
        valid_ids = [fid for fid in frontier_ids if fid in frontier]
        if not valid_ids and frontier:
            # Fallback to first frontier item
            logger.warning("EXPAND action but no valid ids provided")
            policy_tracker.log_issue(
                "expand_no_valid_ids",
                step=step,
                details=f"Provided IDs: {frontier_ids}, Available: {list(frontier.keys())[:5]}"
            )
            valid_ids = [list(frontier.keys())[0]]

        if verbose:
            print("=====Chosen action EXPAND======")
            print("Chosen frontier ids for EXPAND:", valid_ids)
            print("================================\n")
        result["frontier_ids"] = valid_ids
        
    elif action == "RE_QUERY":
        new_query = decision.get("new_query", "")
        if not new_query:
            logger.warning("RE_QUERY action but no new_query provided")
            policy_tracker.log_issue(
                "requery_no_new_query",
                step=step,
                details=f"Reason given: {reason}"
            )
            new_query = "related information"
        
        if verbose:
            print("=====Chosen action RE_QUERY======")
            print("Chosen new query for RE_QUERY:", new_query)
            print("================================\n")
        result["new_query"] = new_query
        
    elif action not in ("STOP", "EXPAND", "RE_QUERY"):
        logger.warning(f"Unknown action: {action}, defaulting to STOP")
        policy_tracker.log_issue(
            "unknown_action",
            step=step,
            details=f"Unknown action: {action}"
        )
        result["action"] = "STOP"
        result["reason"] = f"Unknown action: {action}"
    
    return result