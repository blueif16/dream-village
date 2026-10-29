# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Trajectory Collector for GRPO-based Retrieval Learning
"""

import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from omegaconf import DictConfig

from memora.core.memory_entry import MemoryEntry
from memora.core.memory import AgentMemory, QueryMode
from memora.core.memory_expander import MemoryExpander
from memora.utils.memory import dedup_memories
from memora.retriever.policy_utils import select_from_frontier  # SHARED

from .policy import BasePolicy, PolicyState, PolicyOutput

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryStep:
    """A single step in the trajectory."""
    step_idx: int
    action: str
    action_text: str
    log_prob: float
    frontier_ids: List[str] = field(default_factory=list)
    new_query: str = ""
    reason: str = ""
    confidence: float = 0.0
    working_set_size: int = 0
    frontier_size: int = 0
    current_query: str = ""
    state_text: str = ""
    policy_state_dict: Optional[Dict] = None  # Serialized PolicyState
    
    def to_dict(self, include_state_text: bool = False, for_training: bool = False) -> Dict:
        if for_training:
            return {
                "step_idx": self.step_idx,
                "action_text": self.action_text,
                "log_prob": self.log_prob,
                "state_text": self.state_text,
            }
        
        d = {
            "step_idx": self.step_idx,
            "action": self.action,
            "action_text": self.action_text,
            "log_prob": self.log_prob,
            "frontier_ids": self.frontier_ids,
            "new_query": self.new_query,
            "reason": self.reason,
            "confidence": self.confidence,
            "working_set_size": self.working_set_size,
            "frontier_size": self.frontier_size,
            "current_query": self.current_query,
        }
        if include_state_text:
            d["state_text"] = self.state_text
        return d


@dataclass
class Trajectory:
    """A single trajectory (one rollout for a query)."""
    trajectory_id: int
    steps: List[TrajectoryStep] = field(default_factory=list)
    retrieved_memories: List[Dict] = field(default_factory=list)
    total_log_prob: float = 0.0
    trajectory_score: Optional[float] = None
    groundedness: Optional[float] = None
    redundancy: Optional[float] = None
    cost: Optional[float] = None
    generated_answer: Optional[str] = None
    
    def to_dict(self, include_state_text: bool = False, include_retrieved_memories: bool = True, for_training: bool = False) -> Dict:
        d = {
            "trajectory_id": self.trajectory_id,
            "steps": [s.to_dict(include_state_text, for_training) for s in self.steps],
            "total_log_prob": self.total_log_prob,
            "trajectory_score": self.trajectory_score,
        }
        if not for_training:
            d.update({
                "groundedness": self.groundedness,
                "redundancy": self.redundancy,
                "cost": self.cost,
                "generated_answer": self.generated_answer,
            })
        if include_retrieved_memories and not for_training:
            d["retrieved_memories"] = self.retrieved_memories
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Trajectory':
        steps = [TrajectoryStep(**s) for s in data.get("steps", [])]
        return cls(
            trajectory_id=data.get("trajectory_id", 0),
            steps=steps,
            retrieved_memories=data.get("retrieved_memories", []),
            total_log_prob=data.get("total_log_prob", 0.0),
            trajectory_score=data.get("trajectory_score"),
            groundedness=data.get("groundedness"),
            redundancy=data.get("redundancy"),
            cost=data.get("cost"),
            generated_answer=data.get("generated_answer"),
        )


@dataclass
class QueryTrajectories:
    """All trajectories for a single query (grouped)."""
    query: str
    user_id: str
    ground_truth: Optional[str] = None
    evidence: List[Dict] = field(default_factory=list)
    trajectories: List[Trajectory] = field(default_factory=list)
    # Scoring results (populated after scoring)
    rewards: List[float] = field(default_factory=list)
    advantages: List[float] = field(default_factory=list)
    
    def to_dict(self, include_state_text: bool = False, include_retrieved_memories: bool = True, for_training: bool = False) -> Dict:
        d = {
            "query": self.query,
            "user_id": self.user_id,
            "ground_truth": self.ground_truth,
            "evidence": self.evidence,
            "trajectories": [t.to_dict(include_state_text, include_retrieved_memories, for_training) for t in self.trajectories],
        }
        # Include rewards/advantages if populated
        if self.rewards:
            d["rewards"] = self.rewards
        if self.advantages:
            d["advantages"] = self.advantages
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'QueryTrajectories':
        trajectories = [Trajectory.from_dict(t) for t in data.get("trajectories", [])]
        return cls(
            query=data["query"],
            user_id=data["user_id"],
            ground_truth=data.get("ground_truth"),
            evidence=data.get("evidence", []),
            trajectories=trajectories,
            rewards=data.get("rewards", []),
            advantages=data.get("advantages", []),
        )


def memory_entry_to_dict(entry: MemoryEntry) -> Dict:
    """Convert MemoryEntry to dict for serialization."""
    return {
        "index": entry.index,
        "value": entry.value,
        "score": entry.score,
        "memory_type": entry.memory_type,
        "linked_memory": entry.linked_memory,
        "cue_indices": entry.cue_indices,
    }


class TrajectoryCollector:
    """Collects retrieval trajectories using a policy model."""
    
    def __init__(
        self,
        cfg: DictConfig,
        memory_client: AgentMemory,
        policy: BasePolicy,
        top_k: int = 10,
        max_steps: int = 5,
        enable_hybrid_search: bool = False,
    ):
        self.cfg = cfg
        self.memory_client = memory_client
        self.policy = policy
        self.top_k = top_k
        self.max_steps = max_steps
        self.enable_hybrid_search = enable_hybrid_search
        self.expander = MemoryExpander(memory_client)

        if cfg.memory.get("enable_cue_index", False):
            self.query_mode = QueryMode.BOTH
        else:
            self.query_mode = QueryMode.PRIMARY_ONLY

    def collect_single_trajectory(
        self,
        query: str,
        trajectory_id: int = 0,
        temperature: float = 1.0,
        do_sample: bool = True,
        training_step: int = None,  # NEW
    ) -> Trajectory:
        """Collect one trajectory for a given query."""
        self.expander.reset()
        trajectory = Trajectory(trajectory_id=trajectory_id)
        
        # Initial retrieval
        memory_entries = self.memory_client.query(
            query,
            top_k=self.top_k,
            enable_hybrid_search=self.enable_hybrid_search,
            query_mode=self.query_mode,
        )
        current_query = query
        frontier = self.expander.build_frontier({}, memory_entries)
        
        # Iterative retrieval
        for step_idx in range(1, self.max_steps + 1):
            state = PolicyState(
                query=query,
                current_query=current_query,
                working_set=memory_entries,
                frontier=frontier,
                step=step_idx,
                max_steps=self.max_steps,
            )
            
            output = self.policy.select_action(
                state, 
                temperature=temperature, 
                do_sample=do_sample,
                training_step=training_step,  # NEW
            )
            
            # Record step
            step = TrajectoryStep(
                step_idx=step_idx,
                state_text=self.policy.format_state(state), 
                action=output.action,
                action_text=output.action_text,
                log_prob=output.log_prob,
                frontier_ids=output.frontier_ids,
                new_query=output.new_query,
                reason=output.reason,
                confidence=output.confidence,
                working_set_size=len(memory_entries),
                frontier_size=len(frontier),
                current_query=current_query,
            )

            step.policy_state_dict = {
                "query": state.query,
                "current_query": state.current_query,
                "working_set": [mem.model_dump() for mem in state.working_set],
                "frontier": {k : mem.model_dump() for k, mem in state.frontier.items()},
                "step": state.step,
                "max_steps": state.max_steps,
            }
            
            trajectory.steps.append(step)
            trajectory.total_log_prob += output.log_prob
            
            # Execute action
            if output.action == "STOP":
                break
            elif output.action == "EXPAND":
                chosen = select_from_frontier(frontier, output.frontier_ids)  # SHARED
                if chosen:
                    memory_entries = dedup_memories(memory_entries + chosen)
                    for mem in chosen:
                        frontier.pop(mem.index, None)
                    frontier = self.expander.build_frontier(frontier, chosen)
            elif output.action == "RE_QUERY":
                current_query = output.new_query
                new_entries = self.memory_client.query(
                    current_query,
                    top_k=self.top_k,
                    enable_hybrid_search=self.enable_hybrid_search,
                    query_mode=self.query_mode,
                )
                memory_entries = dedup_memories(memory_entries + new_entries)
                frontier = self.expander.build_frontier(frontier, new_entries)
            else:
                break
        
        trajectory.retrieved_memories = [memory_entry_to_dict(m) for m in memory_entries]
        return trajectory

    def collect_trajectories_for_query(
        self,
        query: str,
        user_id: str,
        ground_truth: str = None,
        evidence: List[Dict] = None,
        num_trajectories: int = 4,
        temperature: float = 1.0,
        do_sample: bool = True,
        training_step: int = None,  # NEW
    ) -> QueryTrajectories:
        """Collect G trajectories for a single query."""
        query_trajs = QueryTrajectories(
            query=query, user_id=user_id,
            ground_truth=ground_truth, evidence=evidence or [],
        )
        for traj_id in range(num_trajectories):
            traj = self.collect_single_trajectory(
                query=query,
                trajectory_id=traj_id,
                temperature=temperature,
                do_sample=do_sample,
                training_step=training_step,  # NEW
            )
            query_trajs.trajectories.append(traj)
        return query_trajs


# I/O functions
def save_trajectories(query_trajectories: List[QueryTrajectories], path: str, **kwargs):
    """Save trajectories to JSON file."""
    data = [qt.to_dict(**kwargs) for qt in query_trajectories]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(query_trajectories)} queries to {path}")


def load_trajectories(path: str) -> List[QueryTrajectories]:
    """Load trajectories from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return [QueryTrajectories.from_dict(d) for d in data]