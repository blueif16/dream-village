# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import time
import math 
import signal
import logging
import traceback
import json
import numpy as np
from typing import Dict, List, Any, Optional 
from dataclasses import dataclass
import torch
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from memora.rl.policy import PolicyState
from memora.rl.policy_qwen import QwenPolicy
from memora.rl.trajectory_collector import (
    TrajectoryCollector,
    QueryTrajectories,
    Trajectory,
)

from memora.rl.trajectory_scorer import TrajectoryScorer
from memora.retriever.policy_utils import policy_tracker

logger = logging.getLogger(__name__)

# tensorboard --logdir=grpo_retrieval_output/tensorboard --port=6006

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    # Training
    output_dir: str = "./grpo_output"
    num_train_epochs: int = 4
    per_device_train_batch_size: int = 4
    group_size: int = 4                   
    temperature: float = 1.0 
    learning_rate: float = 1e-5
    clip_range: float = 0.2       
    max_grad_norm: float = 0.5

    use_kl: bool = True
    kl_coef: float = 0.05             
    target_kl: float = 0.1     

    w_groundedness: float = 1.0     
    w_redundancy: float = 0.4       
    w_cost: float = 0.4        

    use_soft_groundedness: bool = False
    soft_weight: float = 0.3 

    use_scaled_groundedness: bool = False
    checkpoint_every: int = 5


    reward_scale_strategy: str = "batch"  # "group" or "batch"


class RetrievalGRPOTrainer:
    """
    GRPO trainer for retrieval policy.
    
    Following router_trainer pattern:
    1. collect_training_batch(): collect K rollouts per query, flatten to steps
    2. compute_grpo_loss(): compute PPO-clip loss with advantages
    3. training_step(): backward + optimizer step
    4. train(): main loop
    """
    
    def __init__(
        self,
        config: GRPOConfig,
        cfg: DictConfig,           
        policy: QwenPolicy,
        query_data: List[Dict],
        memory_client_factory,
        scorer: Optional[TrajectoryScorer] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            config: GRPO training config
            cfg: Memora DictConfig (for memory client, LLM, etc.)
            policy: QwenPolicy instance
            query_data: Training queries with ground truth
            memory_client_factory: Factory to create AgentMemory per user
            scorer: Optional TrajectoryScorer (created if not provided)
        """
        self.config = config
        self.cfg = cfg
        self.policy = policy
        self.query_data = query_data
        self.memory_client_factory = memory_client_factory
        
        # Initialize scorer with soft groundedness config
        self.scorer = scorer or TrajectoryScorer(
            cfg=cfg,
            w_groundedness=config.w_groundedness,
            w_redundancy=config.w_redundancy,
            w_cost=config.w_cost,
            use_soft_groundedness=config.use_soft_groundedness,
            soft_weight=config.soft_weight,
            use_scaled_groundedness=config.use_scaled_groundedness,  # NEW
        )
        
        # Ensure model is loaded
        self.policy._load_model()
        self.device = self.policy.model.device
        
        # Setup optimizer (only trainable params - LoRA if enabled)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.policy.model.parameters() if p.requires_grad],
            lr=config.learning_rate,
        )
        
        # Training state
        self.global_step = 0
        self.query_index = 0  # Cycles through queries
        self.training_stats = []
        
        # Memory client cache
        self._memory_clients: Dict[str, Any] = {}
        
        # Setup output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Setup TensorBoard logging
        self.writer = SummaryWriter(log_dir=os.path.join(config.output_dir, "tensorboard"))

        logger.info(f"Trainer initialized with {len(query_data)} queries")

    def _get_memory_client(self, user_id: str):
        """Get or create memory client for user."""
        if user_id not in self._memory_clients:
            self._memory_clients[user_id] = self.memory_client_factory(user_id)
        return self._memory_clients[user_id]

    def _rebuild_state_from_dict(self, state_dict: Dict) -> PolicyState:
        """Rebuild PolicyState from serialized dict."""
        from memora.core.memory_entry import MemoryEntry
        
        # Reconstruct MemoryEntry objects from dicts
        working_set = [MemoryEntry(**m) for m in state_dict.get("working_set", [])]
        frontier = {k: MemoryEntry(**m) for k, m in state_dict.get("frontier", {}).items()}
        
        return PolicyState(
            query=state_dict["query"],
            current_query=state_dict["current_query"],
            working_set=working_set,
            frontier=frontier,
            step=state_dict["step"],
            max_steps=state_dict["max_steps"],
        )

    def compute_advantages(
        self, 
        rewards_per_group: List[List[float]]
    ) -> torch.Tensor:
        """
        Compute advantages using the configured scaling strategy.
        Args:
            rewards_per_group: List of reward lists, one per query/group. [[r1, r2, ...], [...], ...]
        Returns: 
            Flattened tensor of advantages
        """
        
        all_advantages = []
        all_rewards_flat = []

        # Compute group relative advantages (subtract local mean)
        for group_rewards in rewards_per_group:
            if len(group_rewards) == 0:
                continue
            group_mean = sum(group_rewards) / len(group_rewards)
            group_advantages = [r - group_mean for r in group_rewards]  # mean centered group advantages
            all_advantages.extend(group_advantages)
            all_rewards_flat.extend(group_rewards)
        
        advantages = torch.tensor(all_advantages, device=self.device)

        # Scale by global std (if using batch scaling) as mentioned in HF TRL
        if self.config.reward_scale_strategy == "batch" and len(advantages) > 1:
            global_std = advantages.std()
            if global_std > 1e-8:
                advantages = advantages / (global_std + 1e-8)
                print(f"    [Scaling] Divided advantages by global std: {global_std:.4f}")
                logger.debug(f"Batch scaling: global std = {global_std:.4f}")

        
        raw_adv_std = torch.tensor(all_advantages).std().item()
        logger.info(f"Raw advantage std: {raw_adv_std:.4f}, Scaled std: {advantages.std().item():.4f}")

        # Warn if too small
        if raw_adv_std < 0.05:
            logger.warning("⚠️ [Advantage var]iance too low - rewards too similar within groups")

        return advantages
    
    def collect_training_batch(
        self,
        batch_size: int,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Collect a batch of rollout groups for training.
        
        Following router_trainer pattern:
        - Sample 'batch_size' queries
        - Collect K trajectories per query
        - Score trajectories (rewards)
        - Compute group relative advantages
        - Flatten to step-level: (state, action_text, log_prob, advantage)

        Args:
            batch_size: Number of queries to collect
            temperature: Sampling temperature

        Returns: 
            Dict with step-level training data
        """

        self.policy.eval_mode()

        all_states: List[PolicyState] = []
        all_action_texts: List[str] = []
        all_old_logprobs: List[float] = []
        # all_advantages: List[float] = []
        all_rewards: List[float] = []

        """
        rewards_per_group = [[0.8, 0.6, 0.9, 0.7], [0.5, 0.4, 0.6, 0.3]]  # 2 groups, 4 trajs each
        steps_per_group = [[3, 2, 4, 2], [5, 3, 2, 4]]  # steps per trajectory
        """
        rewards_per_group: List[List[float]] = []
        steps_per_group: List[List[int]] = []

        # Sample queries 
        sampled_queries = []
        for _ in range(batch_size):
            sampled_queries.append(self.query_data[self.query_index])
            self.query_index = (self.query_index + 1) % len(self.query_data)

        # NEW: Track all scores for logging
        all_score_details = []

        # refer: data_utils.py
        queries_processed = 0
        for q_data in sampled_queries:
            try:
                query = q_data["question"]  # query keyword doesnt exist, check data_utils!
                user_id = q_data["user_id"] 
                ground_truth = q_data.get("ground_truth")  ## adverserial answer?

                # create collector for this user
                memory_client = self._get_memory_client(user_id)
                collector = TrajectoryCollector(
                    cfg=self.cfg,
                    memory_client=memory_client,
                    policy=self.policy,
                    max_steps=self.cfg.get("retrieval", {}).get("local_policy", {}).get("max_steps", 5),
                    top_k=self.cfg.memory.get("top_k", 10),
                )
                # Collect K trajectories (group)
                # collect_trajectories_for_query() internally calls policy.select_action() and computes logprob
                query_trajs: QueryTrajectories = collector.collect_trajectories_for_query(
                    query=query,
                    user_id=user_id,
                    ground_truth=ground_truth,
                    num_trajectories=self.config.group_size,
                    temperature=temperature,
                    do_sample=True,
                    training_step=self.global_step,  # NEW
                )

                if len(query_trajs.trajectories) == 0:
                    logger.warning(f"No trajectories collected for query: {query[:50]}...")
                    continue

                # Score each trajectory (rewards)
                group_rewards = []
                group_step_counts = []
                for traj in query_trajs.trajectories:
                    score_result = self.scorer.score_trajectory(
                        traj, query=query, ground_truth=ground_truth,
                    )
                    total_score = float(score_result.total_score)
                    group_rewards.append(total_score)
                    group_step_counts.append(len(traj.steps))
                    
                    # NEW: Collect score details
                    all_score_details.append({
                        "groundedness": score_result.groundedness,
                        "cost": score_result.cost,
                        "total": total_score,
                        "num_steps": len(traj.steps),
                    })

                all_rewards.extend(group_rewards)
                rewards_per_group.append(group_rewards)
                steps_per_group.append(group_step_counts)  # [2,3,4,...] total steps per group

                # Flatten steps
                for traj_idx, traj in enumerate(query_trajs.trajectories):
                    for step in traj.steps:
                        # step is a TrajectoryStep object
                        state = self._rebuild_state_from_dict(step.policy_state_dict)
                        all_states.append(state)  
                        all_action_texts.append(step.action_text)
                        all_old_logprobs.append(step.log_prob)

                queries_processed += 1
        
            except Exception as e:
                logger.error(f"Error collecting for query: {e}")
        
        if queries_processed == 0:
            raise RuntimeError("Failed to collect any queries - aborting")

        # NEW: Log groundedness distribution
        if all_score_details:
            self._log_groundedness_scores(all_score_details)

        # Compute advantages as per the configuration
        advantages = self.compute_advantages(rewards_per_group)

        # Expand advantages to step level (each step gets its trajectory's advantage)
        step_advantages = []
        group_idx = 0
        

        """
        rewards_per_group = [[0.8, 0.6, 0.9, 0.7], [0.5, 0.4, 0.6, 0.3]]  # 2 groups, 4 trajs each
        steps_per_group = [[3, 2, 4, 2], [5, 3, 2, 4]]  # steps per trajectory

        advantages (flat) = [a0, a1, a2, a3, a4, a5, a6, a7]  # 8 trajectory advantages

        Group 0 (adv_offset=0):
        traj 0: advantages[0] repeated 3 times
        traj 1: advantages[1] repeated 2 times
        traj 2: advantages[2] repeated 4 times
        traj 3: advantages[3] repeated 2 times

        Group 1 (adv_offset=4):
        traj 0: advantages[4] repeated 5 times
        traj 1: advantages[5] repeated 3 times
        ........
        """
        for group_step_counts in steps_per_group:       # ex: [3, 2, 4, 2]
            # rewards_per_group[group_idx] has k rewards for k trajectories
            # group_step_counts has k step counts
            # adv_offset: where this group's trajectories start in the flat advantages tensor
            adv_offset = sum(len(g) for g in rewards_per_group[:group_idx])
            for traj_idx, num_steps in enumerate(group_step_counts):
                traj_advantage = advantages[adv_offset + traj_idx].item()
                step_advantages.extend([traj_advantage] * num_steps)
            group_idx += 1

        return {
            'states': all_states,
            'action_texts': all_action_texts,
            'old_logprobs': torch.tensor(all_old_logprobs, device=self.device),
            'advantages': torch.tensor(step_advantages, device=self.device),
            'mean_reward': sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            'num_steps': len(all_states),
            'reward_std': float(np.std(all_rewards)) if all_rewards else 0.0,  # For logging
        }

    def _log_advantages_to_file(self, advantages_list: List[float], normalized: bool = False):
        """Log advantages to a separate file for analysis."""
        log_file = os.path.join(self.config.output_dir, "advantages_log.jsonl")
        entry = {
            "step": self.global_step,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "normalized": normalized,
            "advantages": advantages_list,
            "count": len(advantages_list),
            "mean": sum(advantages_list) / len(advantages_list) if advantages_list else 0,
            "std": float(np.std(advantages_list)) if advantages_list else 0,
            "min": min(advantages_list) if advantages_list else 0,
            "max": max(advantages_list) if advantages_list else 0,
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _log_groundedness_scores(self, scores: List[dict]):
        """Log groundedness scores to file for distribution analysis."""
        log_file = os.path.join(self.config.output_dir, "groundedness_log.jsonl")
        g_scores = [s["groundedness"] for s in scores]
        entry = {
            "step": self.global_step,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scores": scores,  # List of {groundedness, cost, total, num_steps}
            "groundedness_values": g_scores,
            "g_mean": sum(g_scores) / len(g_scores) if g_scores else 0,
            "g_std": float(np.std(g_scores)) if len(g_scores) > 1 else 0,
            "g_unique": len(set(g_scores)),  # Number of unique values
            "g_ones": sum(1 for g in g_scores if g == 1.0),  # Count of 1s (correct)
            "g_zeros": sum(1 for g in g_scores if g == 0.0),  # Count of 0s (incorrect)
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def compute_grpo_loss(
        self, 
        batch: Dict[str, Any],
        chunk_size: int = 4,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute GRPO loss for a batch
        Args: 
            batch: Batch dict from collect_training_batch
            chunk_size: Process this many steps at a time

        Returns: 
            Dict with loss and metrics
        """

        states = batch['states']  
        action_texts = batch['action_texts']
        old_logprobs = batch['old_logprobs']
        advantages = batch['advantages']

        # log advantages (for debugging)
        adv_mean = advantages.mean().item()
        adv_std = advantages.std().item()
        self._log_advantages_to_file(advantages.tolist(), normalized=True)

        # compute new logprobs
        all_new_logprobs = []
        for i in range(0, len(states), chunk_size):
            chunk_states = states[i:i+chunk_size]
            chunk_actions = action_texts[i:i+chunk_size] 
            chunk_new_logprobs = self.policy.compute_log_prob(chunk_states, chunk_actions)
            all_new_logprobs.append(chunk_new_logprobs)  
        
        new_logprobs = torch.cat(all_new_logprobs)

        # Compute ratio: r = exp(log pi_new - log pi_old)
        ratios = torch.exp(new_logprobs - old_logprobs)

        ## PPO-clip objective
        clip_range = self.config.clip_range
        clipped_ratios = torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range)

        ## Maximizing: max(min(r*A, clip(r)*A))
        ## Minimizing: min(-min(r*A, clip(r)*A)) = min(max(-r*A, -clip(r)*A)) 
        ## Hence, Policy loss: L = -min(r* A, clip(r) * A)
        policy_loss_unclipped = -advantages * ratios         # -r*A
        policy_loss_clipped = -advantages * clipped_ratios   # -clip(r)*A
        policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()

        loss = policy_loss
        kl_penalty = 0.0
        approx_kl = 0.0

        if self.config.use_kl:
            kl_div = (old_logprobs - new_logprobs).mean()
            kl_penalty = self.config.kl_coef * torch.clamp(kl_div - self.config.target_kl, min=0.0)
            loss = loss + kl_penalty
            approx_kl = kl_div.item()

        with torch.no_grad():
            clip_fraction = (torch.abs(ratios - 1.0) > clip_range).float().mean().item()

        
        return {
            'loss': loss,
            'policy_loss': policy_loss.detach(),
            'kl_penalty': kl_penalty.detach() if isinstance(kl_penalty, torch.Tensor) else kl_penalty,
            'clip_fraction': clip_fraction,
            'approx_kl': approx_kl,
            'mean_ratio': ratios.mean().detach(),
            'mean_advantage': adv_mean,  
            'adv_std': adv_std,
        }
    
    def training_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """ 
        Perform one training step 
        
        Args: 
            batch: Training batch from collect_training_batch
        Returns:
            Dict of metrics
        """
        self.policy.train_mode()

        # Clear cuda cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Compute loss
        loss_dict = self.compute_grpo_loss(batch)
        loss = loss_dict['loss']

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.policy.model.parameters() if p.requires_grad],
            self.config.max_grad_norm,
        )

        # Optimizer step
        self.optimizer.step()

        # Clear cache 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
        
        return metrics

    
    def train(self):
        """Main training loop"""
        print("=" * 70)
        print("Starting GRPO Training")
        print("=" * 70)

        # Try to resume from checkpoint
        latest_checkpoint = os.path.join(self.config.output_dir, "latest")
        if os.path.exists(latest_checkpoint):
            self.load_checkpoint(latest_checkpoint)

        num_queries = len(self.query_data)
        steps_per_epoch = math.ceil(num_queries / self.config.per_device_train_batch_size)
        total_steps = steps_per_epoch * self.config.num_train_epochs

        print(f"\nTraining Configuration:")
        print(f"  Queries: {num_queries}")
        print(f"  Epochs: {self.config.num_train_epochs}")
        print(f"  Batch size: {self.config.per_device_train_batch_size}")
        print(f"  Group size (K): {self.config.group_size}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Total steps: {total_steps}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Resume from step: {self.global_step}")
        print()

        # Training loop
        for epoch in range(self.config.num_train_epochs):
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch + 1}/{self.config.num_train_epochs}")
            print(f"{'=' * 70}")

            for step_in_epoch in range(steps_per_epoch):
                # Skip steps if resuming
                current_step = epoch * steps_per_epoch + step_in_epoch + 1
                if current_step <= self.global_step:
                    continue
                    
                self.global_step = current_step
                step_start = time.time()

                try:
                    # Collect fresh trajectories (on-policy)
                    batch = self.collect_training_batch(
                        batch_size=self.config.per_device_train_batch_size,
                        temperature=self.config.temperature,
                    )

                    # Training step
                    metrics = self.training_step(batch)
                    step_time = time.time() - step_start

                    # Log
                    print(f"\n[Step {self.global_step}/{total_steps}] "
                          f"loss={metrics['loss']:.4f}, "
                          f"kl={metrics['approx_kl']:.4f}, "
                          f"clip={metrics['clip_fraction']:.3f}, "
                          f"reward={batch['mean_reward']:.3f}, "
                          f"ratio={metrics['mean_ratio']:.3f}, "  # ADD
                          f"steps={batch['num_steps']}, "
                          f"time={step_time:.1f}s")
                    
                    # TensorBoard logging
                    self.writer.add_scalar('train/loss', metrics['loss'], self.global_step)
                    self.writer.add_scalar('train/reward', batch['mean_reward'], self.global_step)
                    self.writer.add_scalar('train/kl', metrics['approx_kl'], self.global_step)
                    self.writer.add_scalar('train/clip_fraction', metrics['clip_fraction'], self.global_step)
                    self.writer.add_scalar('train/ratio', metrics['mean_ratio'], self.global_step)
                    self.writer.add_scalar('train/num_steps', batch['num_steps'], self.global_step)
                    self.writer.flush()

                    # ADD: Track reward trend
                    if len(self.training_stats) >= 5:
                        recent_rewards = [s['mean_reward'] for s in self.training_stats[-5:]]
                        reward_trend = recent_rewards[-1] - recent_rewards[0]
                        print(f"    [Trend] reward_delta={reward_trend:+.3f} over last 5 steps")

                    self.training_stats.append({
                        'step': self.global_step,
                        'epoch': epoch + 1,
                        **metrics,
                        'mean_reward': batch['mean_reward'],
                        'num_steps': batch['num_steps'],
                    })

                    # Checkpoint
                    if self.global_step % self.config.checkpoint_every == 0:
                        self.save_checkpoint("latest")
                        self.save_checkpoint(f"step_{self.global_step}")

                except Exception as e:
                    logger.error(f"Error in training step {self.global_step}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # === END OF EPOCH: Print summary statistics ===
            self._log_epoch_stats(epoch + 1)

        # Save final
        self.save_checkpoint("final")
        self._save_training_charts()  # Final charts
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"Final model saved to: {self.config.output_dir}/final")

        self.writer.close()

    def _log_epoch_stats(self, epoch: int):
        """Log summary statistics for a completed epoch."""
        import numpy as np
        
        epoch_stats = [s for s in self.training_stats if s["epoch"] == epoch]
        
        if len(epoch_stats) == 0:
            return
        
        print(f"\n{'─' * 70}")
        print(f"EPOCH {epoch} SUMMARY ({len(epoch_stats)} steps)")
        print(f"{'─' * 70}")
        
        # Compute statistics
        metrics_to_log = {
            'Loss': ('loss', '.4f'),
            'Policy Loss': ('policy_loss', '.4f'),
            'KL Penalty': ('kl_penalty', '.4f'),
            'Clip Fraction': ('clip_fraction', '.3f'),
            'Approx KL': ('approx_kl', '.3f'),
            'Mean Ratio': ('mean_ratio', '.3f'),
            'Mean Reward': ('mean_reward', '.3f'),
            '|Advantage|': ('mean_advantage', '.4f'),
        }
        
        for display_name, (key, fmt) in metrics_to_log.items():
            if key == 'mean_advantage':
                values = [abs(s[key]) for s in epoch_stats if key in s]
            else:
                values = [s[key] for s in epoch_stats if key in s]
            
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {display_name:15s}: {mean_val:{fmt}} ± {std_val:{fmt}}")
        
        # Log to TensorBoard as epoch-level metrics
        epoch_metrics = {
            'loss': np.mean([s['loss'] for s in epoch_stats]),
            'clip_fraction': np.mean([s['clip_fraction'] for s in epoch_stats]),
            'approx_kl': np.mean([s['approx_kl'] for s in epoch_stats]),
            'mean_ratio': np.mean([s['mean_ratio'] for s in epoch_stats]),
            'mean_reward': np.mean([s['mean_reward'] for s in epoch_stats]),
            'advantage_magnitude': np.mean([abs(s['mean_advantage']) for s in epoch_stats]),
        }
        
        for key, val in epoch_metrics.items():
            self.writer.add_scalar(f'epoch/{key}', val, epoch)
        
        self.writer.flush()
        print(f"{'─' * 70}\n")
        
        # Generate and save charts
        self._save_training_charts()
    
    def _save_training_charts(self):
        """Generate and save training metric charts to output_dir."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for server
        import matplotlib.pyplot as plt
        
        if len(self.training_stats) < 2:
            return
        
        stats = self.training_stats
        steps = [s["step"] for s in stats]
        epochs = [s["epoch"] for s in stats]
        
        # Extract metrics
        loss = [s["loss"] for s in stats]
        policy_loss = [s["policy_loss"] for s in stats]
        clip_fraction = [s["clip_fraction"] for s in stats]
        approx_kl = [s["approx_kl"] for s in stats]
        mean_ratio = [s["mean_ratio"] for s in stats]
        mean_advantage = [s["mean_advantage"] for s in stats]
        mean_reward = [s["mean_reward"] for s in stats]
        
        # Find epoch boundaries
        epoch_boundaries = []
        for i in range(1, len(epochs)):
            if epochs[i] != epochs[i-1]:
                epoch_boundaries.append(steps[i-1])
        
        def add_epoch_lines(ax):
            for eb in epoch_boundaries:
                ax.axvline(x=eb, color='gray', linestyle='--', alpha=0.5)
        
        def moving_average(data, window=5):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='same')
        
        # Create figure with subplots (5 rows, 2 cols)
        fig, axes = plt.subplots(5, 2, figsize=(14, 20))
        current_epoch = max(epochs)
        fig.suptitle(f"GRPO Training Metrics (Step {self.global_step}, Epoch {current_epoch})", 
                     fontsize=14, fontweight='bold')
        
        # 1. Loss
        ax = axes[0, 0]
        ax.plot(steps, loss, 'b-', alpha=0.5, linewidth=1)
        ax.plot(steps, moving_average(loss), 'b-', linewidth=2, label='MA(5)')
        add_epoch_lines(ax)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Total Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. Policy Loss
        ax = axes[0, 1]
        ax.plot(steps, policy_loss, 'g-', alpha=0.5, linewidth=1)
        ax.plot(steps, moving_average(policy_loss), 'g-', linewidth=2, label='MA(5)')
        add_epoch_lines(ax)
        ax.set_xlabel("Step")
        ax.set_ylabel("Policy Loss")
        ax.set_title("Policy Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 3. Clip Fraction
        ax = axes[1, 0]
        ax.plot(steps, clip_fraction, 'r-', alpha=0.5, linewidth=1)
        ax.plot(steps, moving_average(clip_fraction), 'r-', linewidth=2, label='MA(5)')
        add_epoch_lines(ax)
        ax.axhline(y=0.2, color='orange', linestyle=':', label='Target (0.2)')
        ax.set_xlabel("Step")
        ax.set_ylabel("Clip Fraction")
        ax.set_title("Clip Fraction")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 4. Approx KL
        ax = axes[1, 1]
        ax.plot(steps, approx_kl, 'purple', alpha=0.5, linewidth=1)
        ax.plot(steps, moving_average(approx_kl), 'purple', linewidth=2, label='MA(5)')
        add_epoch_lines(ax)
        ax.set_xlabel("Step")
        ax.set_ylabel("Approx KL")
        ax.set_title("Approximate KL Divergence")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 5. Mean Ratio
        ax = axes[2, 0]
        ax.plot(steps, mean_ratio, 'orange', alpha=0.5, linewidth=1)
        ax.plot(steps, moving_average(mean_ratio), 'orange', linewidth=2, label='MA(5)')
        add_epoch_lines(ax)
        ax.axhline(y=1.0, color='black', linestyle=':', label='Ideal (1.0)')
        ax.axhline(y=1-self.config.clip_range, color='red', linestyle=':', alpha=0.5)
        ax.axhline(y=1+self.config.clip_range, color='red', linestyle=':', alpha=0.5, label='Clip bounds')
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Ratio")
        ax.set_title("Mean Ratio π_new/π_old")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 6. Mean Advantage
        ax = axes[2, 1]
        ax.plot(steps, mean_advantage, 'cyan', alpha=0.5, linewidth=1)
        ax.plot(steps, moving_average(mean_advantage), 'cyan', linewidth=2, label='MA(5)')
        add_epoch_lines(ax)
        ax.axhline(y=0, color='black', linestyle=':')
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Advantage")
        ax.set_title("Mean Advantage")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 7. Mean Reward (per step)
        ax = axes[3, 0]
        ax.plot(steps, mean_reward, 'magenta', alpha=0.5, linewidth=1)
        ax.plot(steps, moving_average(mean_reward), 'magenta', linewidth=2, label='MA(5)')
        add_epoch_lines(ax)
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Mean Reward (per step)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 8. Advantage Magnitude (log scale)
        ax = axes[3, 1]
        adv_magnitude = [abs(a) for a in mean_advantage]
        ax.semilogy(steps, adv_magnitude, 'brown', alpha=0.5, linewidth=1)
        ax.semilogy(steps, moving_average(adv_magnitude), 'brown', linewidth=2, label='MA(5)')
        add_epoch_lines(ax)
        ax.set_xlabel("Step")
        ax.set_ylabel("|Mean Advantage|")
        ax.set_title("Advantage Magnitude (log scale)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 9. Reward vs Epoch (aggregated)
        ax = axes[4, 0]
        unique_epochs = sorted(set(epochs))
        epoch_rewards_mean = []
        epoch_rewards_std = []
        for ep in unique_epochs:
            ep_rewards = [s["mean_reward"] for s in stats if s["epoch"] == ep]
            epoch_rewards_mean.append(np.mean(ep_rewards))
            epoch_rewards_std.append(np.std(ep_rewards))
        
        ax.errorbar(unique_epochs, epoch_rewards_mean, yerr=epoch_rewards_std, 
                    fmt='o-', color='magenta', capsize=5, linewidth=2, markersize=8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Reward vs Epoch")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(unique_epochs)
        
        # 10. Loss vs Epoch (aggregated)
        ax = axes[4, 1]
        epoch_loss_mean = []
        epoch_loss_std = []
        for ep in unique_epochs:
            ep_loss = [s["loss"] for s in stats if s["epoch"] == ep]
            epoch_loss_mean.append(np.mean(ep_loss))
            epoch_loss_std.append(np.std(ep_loss))
        
        ax.errorbar(unique_epochs, epoch_loss_mean, yerr=epoch_loss_std,
                    fmt='o-', color='blue', capsize=5, linewidth=2, markersize=8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss vs Epoch")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(unique_epochs)
        
        plt.tight_layout()
        
        # Save to output_dir
        chart_path = os.path.join(self.config.output_dir, "training_metrics.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Training charts saved to {chart_path}")

    def save_checkpoint(self, name: str):
        """Save training checkpoint."""
        import json
        
        self.writer.flush()
        
        save_path = os.path.join(self.config.output_dir, name)
        os.makedirs(save_path, exist_ok=True)
        
        # Save model (LoRA weights if using LoRA)
        self.policy.save_checkpoint(save_path)
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "query_index": self.query_index,
            "training_stats": self.training_stats,
        }
        state_path = os.path.join(save_path, "training_state.json")
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
        
        # Save optimizer state
        opt_path = os.path.join(save_path, "optimizer.pt")
        torch.save(self.optimizer.state_dict(), opt_path)
        
        # Save policy issues log
        policy_tracker.save_to_file(self.config.output_dir)
        
        logger.info(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training checkpoint. Returns True if successful."""
        import json
        
        state_path = os.path.join(checkpoint_path, "training_state.json")
        opt_path = os.path.join(checkpoint_path, "optimizer.pt")
        
        if not os.path.exists(state_path):
            return False
        
        # Load training state
        with open(state_path, "r") as f:
            state = json.load(f)
        
        self.global_step = state["global_step"]
        self.query_index = state["query_index"]
        self.training_stats = state["training_stats"]
        
        # Load optimizer state
        if os.path.exists(opt_path):
            self.optimizer.load_state_dict(torch.load(opt_path))
        
        logger.info(f"Resumed from checkpoint: step {self.global_step}")
        return True


def main():
    """Training script for retrieval policy GRPO."""
    import argparse
    from pathlib import Path
    from omegaconf import OmegaConf
    from memora.core.memory import AgentMemory
    from memora.rl.data_utils import load_locomo_data, split_locomo_stratified

    parser = argparse.ArgumentParser(description="Train retrieval policy with GRPO")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml (e.g. app/locomo/conf/config.yaml)")
    parser.add_argument("--output_dir", type=str, default="./grpo_retrieval_output")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2, help="Queries per batch")
    parser.add_argument("--group_size", type=int, default=4, help="Trajectories per query (K)")

    parser.add_argument("--use_kl", action="store_true", default=False,
                        help="Enable KL penalty in loss (default: True)")

    parser.add_argument("--scale_strategy", type=str, default="batch",
                        help="Reward scaling strategy: 'batch' or 'group'")
    
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for rollouts")
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--max_train_queries", type=int, default=None, help="Limit training queries (for debugging)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--w_groundedness", type=float, default=1.0, help="Weight for groundedness reward")
    parser.add_argument("--w_redundancy", type=float, default=0.4, help="Weight for redundancy penalty")
    parser.add_argument("--w_cost", type=float, default=0.4, help="Weight for cost penalty")

    parser.add_argument("--use_soft_groundedness", action="store_true", default=False,
                        help="Enable soft groundedness scoring (binary + semantic similarity)")
    parser.add_argument("--soft_weight", type=float, default=0.3,
                        help="Weight for soft component when use_soft_groundedness=True (default: 0.3 = 70%% binary + 30%% soft)")
    # NEW: Scaled groundedness flag
    parser.add_argument("--use_scaled_groundedness", action="store_true", default=False,
                        help="Use 1-5 scale groundedness instead of binary (provides more gradient variance)")
    parser.add_argument("--data_split_file", type=str, default=None,
                        help="Path to custom data split JSON file")

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load config
    cfg_path = Path(args.config)
    cfg = OmegaConf.load(cfg_path)

    # Load and split data
    print("=" * 70)
    print("Loading LoCoMo Data")
    print("=" * 70)
    data_path = os.path.join(cfg.general.data_path, "locomo10.json")
    data, qa_items = load_locomo_data(data_path)
    print(f"Total QA pairs (all categories): {len(qa_items)}")
    
    # Load or create data split
    if args.data_split_file and os.path.exists(args.data_split_file):
        # Use custom split
        print(f"\nUsing custom split: {args.data_split_file}")
        with open(args.data_split_file) as f:
            split_data = json.load(f)
        
        train_ids = set(split_data["train_question_ids"])
        val_ids = set(split_data.get("val_question_ids", []))
        test_ids = set(split_data.get("test_question_ids", []))
        
        train_qa = [q for q in qa_items if q.get("question_id") in train_ids]
        val_qa = [q for q in qa_items if q.get("question_id") in val_ids]
        test_qa = [q for q in qa_items if q.get("question_id") in test_ids]
        
        # Copy split to output dir
        import shutil
        os.makedirs(args.output_dir, exist_ok=True)
        shutil.copy(args.data_split_file, os.path.join(args.output_dir, "data_split.json"))
    else:
        # Use stratified split (existing logic)
        split_save_path = os.path.join(args.output_dir, "data_split.json")
        train_qa, val_qa, test_qa = split_locomo_stratified(
            data, qa_items,
            train_ratio=0.10,
            val_ratio=0.10,
            seed=args.seed,
            exclude_adversarial=True,
            save_path=split_save_path,
        )
    
    print(f"Split: {len(train_qa)} train, {len(val_qa)} val, {len(test_qa)} test")

    # Prepare query data
    query_data = [
        {
            "question": item["question"],
            "user_id": item["user_id"],
            "ground_truth": item["answer"],
            "evidence": item.get("evidence_list", []),
            "category": item.get("category"),
        }
        for item in train_qa
    ]
    
    print(f"\n✓ Training with {len(query_data)} queries")

    # Initialize policy
    print("\n" + "=" * 70)
    print("Initializing Policy")
    print("=" * 70)
    policy = QwenPolicy(
        model_name=args.model_name,
        use_lora=args.use_lora,
    )
    print(f"✓ Policy: {args.model_name}, LoRA={args.use_lora}")

    # Memory client factory
    def get_memory_client(user_id: str) -> AgentMemory:
        return AgentMemory(cfg=cfg, user_id=user_id)

    # Training config
    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        use_kl=args.use_kl,
        reward_scale_strategy=args.scale_strategy,

        group_size=args.group_size,
        learning_rate=args.learning_rate,
        checkpoint_every=args.checkpoint_every,
        temperature=args.temperature,
        w_groundedness=args.w_groundedness,
        w_redundancy=args.w_redundancy,
        w_cost=args.w_cost,
        use_soft_groundedness=args.use_soft_groundedness,
        soft_weight=args.soft_weight,
        use_scaled_groundedness=args.use_scaled_groundedness,
    )

    # Initialize trainer
    trainer = RetrievalGRPOTrainer(
        config=config,
        cfg=cfg,
        policy=policy,
        query_data=query_data,
        memory_client_factory=get_memory_client,
    )

    # Train
    trainer.train()

    # Save final model
    print("\n" + "=" * 70)
    print("Saving Final Model")
    print("=" * 70)
    final_path = os.path.join(args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    policy.save_checkpoint(final_path)
    print(f"✓ Model saved to: {final_path}")


if __name__ == "__main__":
    main()




