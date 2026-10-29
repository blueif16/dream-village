# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Collect trajectories for GRPO training with scoring and advantage computation.
"""
import logging
from pathlib import Path
from omegaconf import OmegaConf

from memora.rl.data_utils import load_locomo_data, split_by_conversation
from memora.rl.trajectory_collector import TrajectoryCollector, save_trajectories, QueryTrajectories
from memora.rl.trajectory_scorer import TrajectoryScorer
from memora.rl.policy_qwen import QwenPolicy
from memora.core.memory import AgentMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_advantages(rewards: list[float]) -> list[float]:
    """Compute group-relative advantages: A_i = R_i - mean(R)"""
    if len(rewards) == 0:
        return []
    mean_r = sum(rewards) / len(rewards)
    return [r - mean_r for r in rewards]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Collect trajectories for GRPO training")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml (e.g. app/locomo/conf/config.yaml)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to locomo10.json dataset")
    parser.add_argument("--output", type=str, default="trajectories_epoch_0.json", help="Output file path")
    args = parser.parse_args()

    # ========== Configuration ==========
    cfg_path = Path(args.config)
    cfg = OmegaConf.load(cfg_path)
    
    # Dataset path
    data_path = args.data_path
    
    # Output path
    output_path = args.output
    
    # ========== Settings ==========
    USE_GPU = True           # Set True when GPU available
    USE_INSTRUCT = True      # False = base model for RL training
    NUM_TEST_QUERIES = 10    # Number of queries to test
    G = 6                    # Trajectories per query
    INCLUDE_STATE_TEXT = False  # Set True for debugging
    
    # Scoring weights (same as grpo_trainer)
    W_GROUNDEDNESS = 1.0
    W_REDUNDANCY = 0.4
    W_COST = 0.2
    
    # ========== Load Data ==========
    logger.info(f"Loading data from {data_path}")
    data, qa_items = load_locomo_data(data_path)
    train_qa, val_qa, test_qa = split_by_conversation(data, qa_items)
    logger.info(f"Loaded {len(train_qa)} training QA pairs")
    
    # Format for trajectory collection (evidence not needed for scoring)
    queries = [
        {
            "query": item["question"],
            "user_id": item["user_id"],
            "ground_truth": item["answer"],
        }
        for item in train_qa
    ]
    
    # ========== Initialize Policy ==========
    if USE_GPU:
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        logger.info(f"Using QwenPolicy with {model_name}")
        policy = QwenPolicy(model_name=model_name)
    
    # ========== Initialize Scorer ==========
    scorer = TrajectoryScorer(
        cfg=cfg,
        w_groundedness=W_GROUNDEDNESS,
        w_redundancy=W_REDUNDANCY,
        w_cost=W_COST,
    )
    logger.info(f"Scorer initialized: w_g={W_GROUNDEDNESS}, w_r={W_REDUNDANCY}, w_c={W_COST}")
    
    # ========== Collect Trajectories ==========
    test_queries = queries[:NUM_TEST_QUERIES]
    logger.info(f"Collecting {G} trajectories for {len(test_queries)} queries...")
    
    all_query_trajectories: list[QueryTrajectories] = []
    memory_clients = {}  # Cache per user
    
    # Track scoring statistics
    all_rewards = []
    all_advantages = []
    zero_advantage_count = 0
    
    for q_idx, q_data in enumerate(test_queries):
        user_id = q_data["user_id"]
        query = q_data["query"]
        ground_truth = q_data.get("ground_truth")
        
        # Get or create memory client for this user
        if user_id not in memory_clients:
            logger.info(f"Creating memory client for user: {user_id}")
            memory_clients[user_id] = AgentMemory(cfg, user_id)
        
        collector = TrajectoryCollector(
            cfg=cfg,
            memory_client=memory_clients[user_id],
            policy=policy,
            max_steps=5,
            top_k=10,
        )
        
        logger.info(f"\nQuery {q_idx+1}/{len(test_queries)}: {query[:50]}...")
        
        # Collect G trajectories for this query (grouped)
        query_trajs = collector.collect_trajectories_for_query(
            query=query,
            user_id=user_id,
            ground_truth=ground_truth,
            evidence=q_data.get("evidence"),
            num_trajectories=G,
            temperature=1.2,
            do_sample=True,
        )
        
        # ========== Score each trajectory ==========
        rewards = []
        for traj in query_trajs.trajectories:
            score_result = scorer.score_trajectory(
                traj,
                query=query,
                ground_truth=ground_truth,
            )
            
            # Store scores in trajectory
            traj.trajectory_score = score_result.total_score
            traj.groundedness = score_result.groundedness
            traj.redundancy = score_result.redundancy
            traj.cost = score_result.cost
            traj.generated_answer = score_result.generated_answer
            
            rewards.append(score_result.total_score)
        
        # ========== Compute advantages ==========
        advantages = compute_advantages(rewards)
        
        # Check if all advantages are zero (no learning signal)
        if all(abs(a) < 1e-6 for a in advantages):
            zero_advantage_count += 1
        
        all_rewards.extend(rewards)
        all_advantages.extend(advantages)
        
        # Store in query_trajs for later analysis
        query_trajs.rewards = rewards
        query_trajs.advantages = advantages
        
        all_query_trajectories.append(query_trajs)
        
        # Print per-query scoring summary
        print(f"\n  Scoring for Query {q_idx+1}:")
        print(f"  {'Traj':<5} {'Score':<8} {'Adv':<10} {'Ground':<8} {'Redund':<8} {'Cost':<8} {'Steps':<6}")
        print(f"  {'-'*60}")
        for i, traj in enumerate(query_trajs.trajectories):
            print(f"  {i:<5} {traj.trajectory_score:<8.3f} {advantages[i]:<10.4f} "
                  f"{traj.groundedness:<8.2f} {traj.redundancy:<8.2f} {traj.cost:<8.2f} "
                  f"{len(traj.steps):<6}")
        
        print(f"  Mean reward: {sum(rewards)/len(rewards):.4f}, "
              f"Reward std: {(sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards))**0.5:.4f}")
    
    # ========== Save ==========
    save_trajectories(all_query_trajectories, "trajectories_full.json", 
                      include_state_text=True, include_retrieved_memories=True)
    
    logger.info(f"Saved to trajectories_full.json")
    
    # ========== Overall Summary ==========
    print("\n" + "="*70)
    print("TRAJECTORY COLLECTION & SCORING SUMMARY")
    print("="*70)
    
    print(f"\nQueries processed: {len(test_queries)}")
    print(f"Trajectories per query: {G}")
    print(f"Total trajectories: {len(all_rewards)}")
    
    print(f"\n--- Reward Statistics ---")
    if all_rewards:
        mean_reward = sum(all_rewards) / len(all_rewards)
        reward_std = (sum((r - mean_reward)**2 for r in all_rewards) / len(all_rewards))**0.5
        print(f"  Mean reward: {mean_reward:.4f}")
        print(f"  Std reward:  {reward_std:.4f}")
        print(f"  Min reward:  {min(all_rewards):.4f}")
        print(f"  Max reward:  {max(all_rewards):.4f}")
        print(f"  Reward range: {max(all_rewards) - min(all_rewards):.4f}")
    
    print(f"\n--- Advantage Statistics ---")
    if all_advantages:
        nonzero_adv = [a for a in all_advantages if abs(a) > 1e-6]
        print(f"  Zero-advantage queries: {zero_advantage_count}/{len(test_queries)} "
              f"({100*zero_advantage_count/len(test_queries):.1f}%)")
        print(f"  Non-zero advantages: {len(nonzero_adv)}/{len(all_advantages)}")
        if nonzero_adv:
            print(f"  Mean |advantage|: {sum(abs(a) for a in nonzero_adv)/len(nonzero_adv):.4f}")
    
    print(f"\n--- Diagnosis ---")
    if zero_advantage_count > len(test_queries) * 0.5:
        print("  ⚠️  WARNING: >50% of queries have zero advantage!")
        print("     Possible causes:")
        print("     1. All trajectories get same reward (scoring too coarse)")
        print("     2. Group size too small (try G=8)")
        print("     3. Temperature too low (trajectories too similar)")
    else:
        print("  ✓ Sufficient reward variance for learning")
    
    if all_rewards and reward_std < 0.1:
        print("  ⚠️  WARNING: Low reward variance (std < 0.1)")
        print("     Consider adjusting scoring weights or adding noise")
    
    # ========== Per-Query Details ==========
    print("\n" + "="*70)
    print("PER-QUERY DETAILS")
    print("="*70)
    
    for qt in all_query_trajectories:
        print(f"\nQuery: {qt.query[:60]}...")
        print(f"  User: {qt.user_id}")
        print(f"  Ground Truth: {qt.ground_truth[:80] if qt.ground_truth else 'N/A'}...")
        print(f"  Rewards: {[f'{r:.3f}' for r in qt.rewards]}")
        print(f"  Advantages: {[f'{a:.4f}' for a in qt.advantages]}")
        
        for traj in qt.trajectories:
            actions = [s.action for s in traj.steps]
            print(f"    [{traj.trajectory_id}] score={traj.trajectory_score:.3f}, "
                  f"steps={len(traj.steps)}, "
                  f"actions={actions}")


if __name__ == "__main__":
    main()