# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Trajectory Collector for GRPO-based Retrieval Learning

Collects retrieval trajectories that model the Policy-Guided Sequential Retrieval

Actions:
    a) QUERY_CUE_INDEX: Query the cue index
    b) QUERY_PRIMARY_INDEX: Query the primary index  
    c) PROPAGATE_PRIMARY_TO_CUE: From primary memory, follow to linked cue indices
    d) PROPAGATE_CUE_TO_CUE: From one cue index to another
    e) REFORMULATE_QUERY: Reformulate the query based on retrieved memories
    f) STOP: Stop retrieval when sufficient memories are retrieved

State: (query, retrieved_memories, frontier, budget)
"""

import json 
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np  
from omegaconf import DictConfig
import random
import math
import logging

from memora.memora_client import MemoraClient
from memora.core.memory_entry import MemoryEntry
from memora.core.memory import QueryMode

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Possible retrieval actions in the MDP"""
    ## Currently only using QUERY actions and STOP
    QUERY_PRIMARY_INDEX = "query_primary"
    QUERY_CUE_INDEX = "query_cue"

    # PROPAGATE_PRIMARY_TO_CUE = "propagate_primary_to_cue"
    # PROPAGATE_CUE_TO_PRIMARY = "propagate_cue_to_primary"
    # REFORMULATE_QUERY = "reformulate"
    STOP = "stop"

# To add a disabling option for budget as of now -- WILL SEE LATER
# Action costs (for budget tracking)
ACTION_COSTS = {
    ActionType.QUERY_PRIMARY_INDEX: 1.0,
    ActionType.QUERY_CUE_INDEX: 1.0,
    
    # ActionType.PROPAGATE_PRIMARY_TO_CUE: 0.5,
    # ActionType.PROPAGATE_CUE_TO_PRIMARY: 0.5,
    # ActionType.REFORMULATE_QUERY: 0.2,
    ActionType.STOP: 0.0,
}

@dataclass
class RetrievalState:
    """State s_t = (q_t, W_t, F_t, b_t) in the MDP"""
    query: str                              # Current query q_t
    retrieved_memories: List[str]           # W_t: Retrieved memory indices
    frontier: List[str]                     # F_t: Frontier of candidate memories
    budget: float                           # b_t: Remaining budget
    step: int = 0


@dataclass
class RetrievalAction:
    """An action taken at step t"""
    action_type: ActionType
    target_memory_index: Optional[str] = None  # Memory selected (if applicable)
    new_query: Optional[str] = None            # New query (if reformulate)
    score: float = 0.0                         # Selection score

@dataclass
class RetrievalStep:
    """A single step (s_t, a_t) in the trajectory"""
    step_idx: int
    state: Dict  # Serialized state
    action: Dict  # Serialized action
    reward: float = 0.0  # Step reward (may be sparse)


@dataclass
class Trajectory:
    """A complete retrieval trajectory for a single query"""
    query: str
    user_id: str
    steps: List[RetrievalStep] = field(default_factory=list)
    retrieved_memories: List[Dict] = field(default_factory=list)
    final_answer: Optional[str] = None
    ground_truth: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    trajectory_score: Optional[float] = None
    
    # Trajectory-level metrics for scoring
    groundedness: Optional[float] = None
    redundancy: Optional[float] = None
    cost: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "user_id": self.user_id,
            "steps": [asdict(step) for step in self.steps],
            "retrieved_memories": self.retrieved_memories,
            "final_answer": self.final_answer,
            "ground_truth": self.ground_truth,
            "evidence": self.evidence,
            "trajectory_score": self.trajectory_score,
            "groundedness": self.groundedness,
            "redundancy": self.redundancy,
            "cost": self.cost,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Trajectory':
        steps = [RetrievalStep(**s) for s in data.get("steps", [])]
        return cls(
            query=data["query"],
            user_id=data["user_id"],
            steps=steps,
            retrieved_memories=data.get("retrieved_memories", []),
            final_answer=data.get("final_answer"),
            ground_truth=data.get("ground_truth"),
            evidence=data.get("evidence", []),
            trajectory_score=data.get("trajectory_score"),
            groundedness=data.get("groundedness"),
            redundancy=data.get("redundancy"),
            cost=data.get("cost"),
        )


class TrajectoryCollector:
    """
    Collects retrieval trajectories following the Policy-Guided Sequential Retrieval
    algorithm
    
    Current implementation uses the existing Memora retrieval as a behavior policy
    with controlled stochasticity for exploration.
    """

    def __init__(
            self, 
            cfg: DictConfig, 
            top_k: int = 20, 
            budget: float = 10.0,
            max_steps: int = 15
    ):
        self.cfg = cfg
        self.top_k = top_k
        self.initial_budget = budget
        self.max_steps = max_steps
        self.client_cache: Dict[str, MemoraClient] = {}

    def get_client(self, user_id: str) -> MemoraClient:
        """Get or create a MemoraClient for the given user_id"""
        if user_id not in self.client_cache:
            self.client_cache[user_id] = MemoraClient(self.cfg, user_id=user_id)
        return self.client_cache[user_id]

    def _init_frontier(
            self,
            query: str,
            client: MemoraClient
    ) -> Tuple[List[MemoryEntry], List[MemoryEntry]]:
        """
        Initialize frontier F_0 with initial retrieval results
        Returns (primary_candidates, cue_candidates)
        """
         # Query both primary and cue indices
        primary_results = client.query(
            query,
            top_k=self.top_k,
            where={"memory_type": {"$eq": "factual"}},
        )
        
        cue_results = client.query(
            query,
            top_k=self.top_k // 2,    
            where={"linked_memory": {"$ne": ""}},
        )

        return primary_results, cue_results

    def _select_action(
        self,
        state: RetrievalState,
        primary_candidates: List[MemoryEntry],
        cue_candidates: List[MemoryEntry],
        temperature: float = 0.0,
        policy_network = None,
    ) -> RetrievalAction:
        """
        Policy pi(a_t | s_t) to select the next action
        Currently uses a simple heuristic + optional stochasticity
        To replace or augment with learned policy after GRPO training
        
        Returns: RetrievalAction object
        """

        if policy_network is not None:
            # Use the learned policy network to select action
            # (Not implemented - TO DO)
            pass

        # deduplication -> don't store a memory that has already been retrieved
        available_primary = [
            m for m in primary_candidates if m.index not in state.retrieved_memories
        ]
        available_cue = [
            m for m in cue_candidates if m.index not in state.retrieved_memories
        ]

        ## Check stopping conditions
        if state.budget <= 0 or (not available_primary and not available_cue):
            return RetrievalAction(action_type=ActionType.STOP)
        
        all_candidates = []

        for m in available_primary:
            all_candidates.append((m, ActionType.QUERY_PRIMARY_INDEX, m.score))

        for m in available_cue:
            all_candidates.append((m, ActionType.QUERY_CUE_INDEX, m.score))
        
        if not all_candidates:
            return RetrievalAction(action_type=ActionType.STOP)

        # Select based on temperature
        if temperature > 0 and len(all_candidates) > 1:
            scores = [c[2] for c in all_candidates]
            probs = self._softmax(scores, temperature)
            selected_idx = random.choices(range(len(all_candidates)), weights=probs)[0]
        else:
            # Greedy: pick highest score
            selected_idx = max(range(len(all_candidates)), key=lambda i: all_candidates[i][2])
        
        selected = all_candidates[selected_idx]
        return RetrievalAction(
            action_type=selected[1],
            target_memory_index=selected[0].index,  ## m.index
            score=selected[2],
        )
        
    def _apply_action(
        self,
        action: RetrievalAction,
        state: RetrievalState,
        client: MemoraClient,
        primary_candidates: List[MemoryEntry],
        cue_candidates: List[MemoryEntry],
    ) -> Tuple[Set[str], MemoryEntry, float]:
        """
        Apply action a_t to get new retrieved memories and updated frontier

        Returns:
            (new_memories, retrieved_entry, cost)
        """
        new_memories = set()
        retrieved_entry = None   ## MemoryEntry
        cost = ACTION_COSTS[action.action_type]

        if action.action_type == ActionType.STOP:
            return new_memories, retrieved_entry, cost
        
        if action.action_type in [ActionType.QUERY_PRIMARY_INDEX, ActionType.QUERY_CUE_INDEX]:
            # Find memory entry
            # Iterate the candidates and search for the memory with the target index
            for m in primary_candidates + cue_candidates:
                if m.index == action.target_memory_index:
                    retrieved_entry = m
                    break

            if retrieved_entry:
                if retrieved_entry.is_cue_index():
                    # get_linked_memories(): points to list of primary memory
                    # indices it points to
                    for primary_idx in retrieved_entry.get_linked_memories():
                        # Only add if not already retrieved
                        if primary_idx not in state.retrieved_memories:
                            primary_entry = client._client._agent_memory.get(primary_idx)
                            if primary_entry:
                                new_memories.add(primary_idx)
                else:
                    new_memories.add(retrieved_entry.index)

        return new_memories, retrieved_entry, cost
    
    def collect_single_trajectory(
            self,
            query: str,
            user_id: str,
            ground_truth: str = None,
            evidence: List[str] = None,
            temperature: float = 0.0,
            policy_network = None,
    ) -> Trajectory:
        """
        Collect one trajectory for the given query

        Args:
            query: The question q
            user_id: User ID for memory lookup
            ground_truth: Expected Answer (for evaluation)
            evidence: Ground truth evidence memories
            temperature: Sampling temperature (0=greedy, >0=stochastic)
            
        Returns:
            Trajectory with all steps recorded
        """
        client = self.get_client(user_id)
        
        state = RetrievalState(
            query = query,
            retrieved_memories = [],
            frontier = [],
            budget = self.initial_budget,
            step = 0,
        )

        primary_candidates, cue_candidates = self._init_frontier(query, client)

        trajectory = Trajectory(
            query=query,
            user_id=user_id,
            ground_truth=ground_truth,
            evidence=evidence or [],
        )

        total_cost = 0.0

        for t in range(self.max_steps):
            state.step = t

            # Select action
            action = self._select_action(state, primary_candidates,
                                        cue_candidates, temperature, policy_network)
            
            if action.action_type == ActionType.STOP or state.budget <= 0:
                # Stop condition
                step = RetrievalStep(
                    step_idx=t,
                    state = {
                        "query": state.query,
                        "retrieved_count": len(state.retrieved_memories),
                        "budget": state.budget,
                    },
                    action={
                        "action_type": ActionType.STOP.value,
                    }
                )
                trajectory.steps.append(step)
                break

            # Apply action
            new_memories, retrieved_entry, cost = self._apply_action(
                 action,
                 state,
                 client,
                 primary_candidates,
                 cue_candidates
            )

            step = RetrievalStep(
                step_idx=t,
                state = {
                    "query": state.query,
                    "retrieved_count": len(state.retrieved_memories),
                    "budget": state.budget,
                },
                action={
                    "action_type": action.action_type.value,
                    "target_memory_index": action.target_memory_index,
                    "score": action.score,
                }
            )
            trajectory.steps.append(step)
            state.retrieved_memories.extend(new_memories)

            if retrieved_entry: 
                if retrieved_entry.is_cue_index():
                    # For cue index, store the linked primary memories
                    for primary_idx in retrieved_entry.get_linked_memories():
                        primary_entry = client._client._agent_memory.get(primary_idx)
                        if primary_entry:
                            trajectory.retrieved_memories.append({
                                "index": primary_entry.index,
                                "value": primary_entry.value,
                                "score": action.score,
                                "via_cue": retrieved_entry.index,
                            })
                else:
                    trajectory.retrieved_memories.append({
                        "index": retrieved_entry.index,
                        "value": retrieved_entry.value,
                        "score": action.score,
                    })
            
            ## Update budget
            state.budget -= cost
            total_cost += cost

        trajectory.cost = total_cost
        return trajectory
    
    def collect_trajectory_group(
        self,
        query: str,
        user_id: str,
        ground_truth: str = None,
        evidence: List[str] = None,
        G: int = 4,
        temperatures: List[float] = None,
    ) -> List[Trajectory]:
        """
        Collect G trajectories for a single query for GRPO.
        
        Uses different temperatures to generate diverse trajectories:
        - τ(1): Greedy (temperature=0)
        - τ(2..G): Stochastic (temperature>0)
        """
        if temperatures is None:
            temperatures = [0.0] + [0.3 + 0.2 * i for i in range(G - 1)]

        trajectories = []
        for g, temp in enumerate(temperatures[:G]):
            traj = self.collect_single_trajectory(
                query=query,
                user_id=user_id,
                ground_truth=ground_truth,
                evidence=evidence,
                temperature=temp,
                policy_network=None,
            )
            trajectories.append(traj)
        
        return trajectories

    def _softmax(self, scores: List[float], temperature: float) -> List[float]:
        """Compute softmax probabilities with temperature"""
        if not scores:
            return []
        scaled = [s / max(temperature, 1e-8) for s in scores]
        max_s = max(scaled)
        exps = [math.exp(s - max_s) for s in scaled]
        sum_exps = sum(exps)
        return [e / sum_exps for e in exps]


    '''
    # Might Use in Future Implementation (if going with dynamic frontier)
    # TRAVERSE_A→C: From a retrieved primary, add its cues to frontier
    def _traverse_primary_to_cue(self, primary_entry, cue_candidates, state):
        """Expand frontier with primary's cue indices"""
        new_cues = []
        for cue_idx in primary_entry.get_cue_indices():
            if cue_idx not in [c.index for c in cue_candidates]:
                if cue_idx not in state.retrieved_memories:
                    cue_entry = self.client._client._agent_memory.get(cue_idx)
                    if cue_entry:
                        new_cues.append(cue_entry)
        cue_candidates.extend(new_cues)
        return new_cues  # ΔF (frontier expansion, no retrieval)

    # TRAVERSE_C→C: From a cue, find semantically related cues
    # This would require additional infrastructure (cue-to-cue links)
    # NOT currently supported in the codebase
    '''