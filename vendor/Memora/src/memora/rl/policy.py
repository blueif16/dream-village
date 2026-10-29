# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Policy interfaces for RL-based retrieval.

This module defines the abstract base class and data structures for 
retrieval policies used in GRPO training.
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, List
from dataclasses import dataclass, field

from memora.core.memory_entry import MemoryEntry
from memora.retriever.policy_utils import (
    POLICY_SYSTEM_MESSAGE,
    format_user_message,
    format_working_set,
    format_frontier,
    parse_json_response,
    validate_policy_decision,
)


@dataclass
class PolicyState:
    """State representation for policy input."""
    query: str
    current_query: str
    working_set: List[MemoryEntry]
    frontier: Dict[str, MemoryEntry]
    step: int
    max_steps: int


@dataclass 
class PolicyOutput:
    """Output from policy action selection."""
    action: str
    action_text: str
    frontier_ids: List[str] = field(default_factory=list)
    new_query: str = ""
    reason: str = ""
    confidence: float = 0.0
    log_prob: float = 0.0


class BasePolicy(ABC):
    """Abstract base class for retrieval policies."""
    
    def format_state(self, state: PolicyState) -> str:
        """Format state into user message content."""
        return format_user_message(
            user_question=state.query,
            current_query=state.current_query,
            working_set=state.working_set,
            frontier=state.frontier,
            step=state.step,
            max_steps=state.max_steps,
        )
    
    def get_system_message(self) -> str:
        """Get the system message for the policy."""
        return POLICY_SYSTEM_MESSAGE
    
    def build_messages(self, state: PolicyState) -> List[Dict[str, str]]:
        """Build chat messages from state."""
        return [
            {"role": "system", "content": self.get_system_message()},
            {"role": "user", "content": self.format_state(state)},
        ]
    
    def build_messages_with_response(
        self, 
        state: PolicyState, 
        action_text: str,
    ) -> List[Dict[str, str]]:
        """Build chat messages including the assistant response."""
        messages = self.build_messages(state)
        messages.append({"role": "assistant", "content": action_text})
        return messages
    
    def _parse_action(
        self, 
        response_text: str, 
        frontier: Dict[str, MemoryEntry],
        training_step: int = None,
    ) -> PolicyOutput:
        """Parse LLM response into PolicyOutput."""
        decision = parse_json_response(response_text, step=training_step)
        validated = validate_policy_decision(decision, frontier, step=training_step)
        
        return PolicyOutput(
            action=validated["action"],
            action_text=response_text,
            frontier_ids=validated["frontier_ids"],
            new_query=validated["new_query"],
            reason=validated["reason"],
            confidence=validated["confidence"],
        )
    
    @abstractmethod
    def select_action(
        self, 
        state: PolicyState, 
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> PolicyOutput:
        """Select action given current state."""
        pass
    
    @abstractmethod
    def compute_log_prob(
        self, 
        state: PolicyState, 
        action_text: str,
    ) -> torch.Tensor:
        """Compute log probability of action (differentiable for training)."""
        pass