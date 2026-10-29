# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from typing import Dict, List, Tuple, Optional
import random
from pathlib import Path
from collections import defaultdict


def resolve_evidence_ids(
    evidence_ids: List[str],
    conversation_data: Dict,
) -> List[Dict]:
    """
    Resolve evidence IDs like ["D1:3", "D1:4"] to actual dialogue text with metadata.
    
    Note: Currently unused in GRPO scoring, but useful for debugging/analysis.
    """
    evidence_list = []
    
    # Build lookup: dia_id -> (text, session_num) from all sessions
    dia_id_to_info = {}
    
    for key, value in conversation_data.items():
        if key.startswith("session_") and isinstance(value, list):
            session_num = int(key.split("_")[1])
            for turn in value:
                if isinstance(turn, dict) and "dia_id" in turn:
                    dia_id_to_info[turn["dia_id"]] = {
                        "text": turn.get("text", ""),
                        "speaker": turn.get("speaker", ""),
                        "session_num": session_num,
                    }
    
    # Get session timestamps
    session_timestamps = {}
    for key, value in conversation_data.items():
        if key.endswith("_date_time"):
            session_num = int(key.split("_")[1])
            session_timestamps[session_num] = value
    
    # Resolve each evidence ID
    for ev_id in evidence_ids:
        if ev_id in dia_id_to_info:
            info = dia_id_to_info[ev_id]
            session_num = info["session_num"]
            timestamp = session_timestamps.get(session_num, "Unknown")
            evidence_list.append({
                "dia_id": ev_id,
                "text": info["text"],
                "speaker": info["speaker"],
                "timestamp": timestamp,
                "session_num": session_num,
            })
    
    return evidence_list


def load_locomo_data(
    data_path: str,
    resolve_evidence: bool = False,
    use_combined_user: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load LoCoMo dataset.

    Args:
        data_path: Path to locomo JSON file
        resolve_evidence: Whether to resolve evidence IDs to text (default False,
                         only needed for debugging/analysis, not for GRPO scoring)
        use_combined_user: If True, user_id = {speaker_a}_{speaker_b}_{idx}
    
    Returns:
        (conversations, qa_items) - raw conversations and flattened QA pairs
    """
    with open(data_path, "r") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = list(data.values())
    
    qa_items = []
    
    for idx, item in enumerate(data):
        conversation_data = item.get("conversation", {})
        speaker_a = conversation_data.get("speaker_a", "speaker_a")
        speaker_b = conversation_data.get("speaker_b", "speaker_b")
        
        if use_combined_user:
            user_id = f"{speaker_a}_{speaker_b}_{idx}"
        else:
            user_id = None
        
        for qa in item.get("qa", []):
            # Create question_id matching evaluation format: "{user_idx}_{category}_{question_prefix}"
            question_prefix = qa["question"][:30]
            question_id = f"{idx}_{qa.get('category', 0)}_{question_prefix}"
            
            qa_item = {
                "question": qa["question"],
                "answer": str(qa.get("answer", qa.get("adversarial_answer", ""))),
                "category": qa.get("category", 0),
                "conv_idx": idx,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "question_id": question_id,
            }
            
            if use_combined_user:
                qa_item["user_id"] = user_id
            else:
                qa_item["user_id"] = None
                qa_item["speaker_a_user_id"] = f"{speaker_a}_{idx}"
                qa_item["speaker_b_user_id"] = f"{speaker_b}_{idx}"

            if resolve_evidence:
                evidence_ids = qa.get("evidence", [])
                qa_item["evidence"] = evidence_ids
                qa_item["evidence_list"] = resolve_evidence_ids(evidence_ids, conversation_data)
            
            if qa_item["answer"]:
                qa_items.append(qa_item)
    
    return data, qa_items


def split_locomo_stratified(
    data: List[Dict],
    qa_items: List[Dict],
    train_ratio: float = 0.10,
    val_ratio: float = 0.10,
    seed: int = 42,
    exclude_adversarial: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split LoCoMo QA items with stratified sampling across categories.
    
    Ensures each split (train/val/test) has proportional representation from
    each category (1-4), excluding adversarial (category 5) by default.
    
    Split ratio: 1:1:8 (train:val:test) = 10%:10%:80%
    
    Args:
        data: Raw conversation data (unused but kept for API consistency)
        qa_items: All QA items from load_locomo_data()
        train_ratio: Fraction for training (default 0.10 = 10%)
        val_ratio: Fraction for validation (default 0.10 = 10%)
        seed: Random seed for reproducibility
        exclude_adversarial: Whether to exclude category 5 (default True)
        save_path: Optional path to save split info JSON for reproducibility
        
    Returns:
        (train_qa, val_qa, test_qa) - stratified splits
    """
    random.seed(seed)
    
    # Filter out adversarial (category 5) if requested
    if exclude_adversarial:
        qa_items = [q for q in qa_items if int(q.get("category", 0)) != 5]
    
    # Group QA items by category
    category_to_qa: Dict[int, List[Dict]] = defaultdict(list)
    for q in qa_items:
        cat = int(q.get("category", 0))
        category_to_qa[cat].append(q)
    
    # Stratified split: sample proportionally from each category
    train_qa, val_qa, test_qa = [], [], []
    
    print(f"\n{'='*60}")
    print("Stratified Split (1:1:8 ratio)")
    print(f"{'='*60}")
    print(f"Seed: {seed}")
    print(f"Train ratio: {train_ratio:.0%}, Val ratio: {val_ratio:.0%}, Test ratio: {1-train_ratio-val_ratio:.0%}")
    print(f"\nPer-category breakdown:")
    print(f"{'Category':<10} {'Total':<8} {'Train':<8} {'Val':<8} {'Test':<8}")
    print(f"{'-'*42}")
    
    split_info = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "exclude_adversarial": exclude_adversarial,
        "categories": {},
        "train_question_ids": [],
        "val_question_ids": [],
        "test_question_ids": [],
    }
    
    for cat in sorted(category_to_qa.keys()):
        cat_items = category_to_qa[cat]
        random.shuffle(cat_items)
        
        n = len(cat_items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        # Ensure at least 1 sample in train/val if category has enough items
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
        
        cat_train = cat_items[:n_train]
        cat_val = cat_items[n_train:n_train + n_val]
        cat_test = cat_items[n_train + n_val:]
        
        train_qa.extend(cat_train)
        val_qa.extend(cat_val)
        test_qa.extend(cat_test)
        
        print(f"{cat:<10} {n:<8} {len(cat_train):<8} {len(cat_val):<8} {len(cat_test):<8}")
        
        split_info["categories"][str(cat)] = {
            "total": n,
            "train": len(cat_train),
            "val": len(cat_val),
            "test": len(cat_test),
        }
    
    # Shuffle within each split to mix categories
    random.shuffle(train_qa)
    random.shuffle(val_qa)
    random.shuffle(test_qa)
    
    print(f"{'-'*42}")
    print(f"{'TOTAL':<10} {len(qa_items):<8} {len(train_qa):<8} {len(val_qa):<8} {len(test_qa):<8}")
    
    # Store question identifiers for reproducibility verification
    split_info["train_question_ids"] = [
        f"{q['conv_idx']}_{q['category']}_{q['question'][:30]}" for q in train_qa
    ]
    split_info["val_question_ids"] = [
        f"{q['conv_idx']}_{q['category']}_{q['question'][:30]}" for q in val_qa
    ]
    split_info["test_question_ids"] = [
        f"{q['conv_idx']}_{q['category']}_{q['question'][:30]}" for q in test_qa
    ]
    
    # Save split info if path provided
    if save_path:
        with open(save_path, "w") as f:
            json.dump(split_info, f, indent=2)
        print(f"\n✓ Split info saved to: {save_path}")
    
    return train_qa, val_qa, test_qa


def load_split(
    qa_items: List[Dict],
    split_path: str,
    exclude_adversarial: bool = True,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load a previously saved split and reconstruct train/val/test sets.
    
    Args:
        qa_items: All QA items from load_locomo_data()
        split_path: Path to saved split JSON
        exclude_adversarial: Whether to exclude category 5
        
    Returns:
        (train_qa, val_qa, test_qa)
    """
    with open(split_path, "r") as f:
        split_info = json.load(f)
    
    # Verify seed matches
    saved_seed = split_info.get("seed")
    print(f"Loading split from: {split_path}")
    print(f"  Seed: {saved_seed}")
    
    # Recreate the split using the same seed
    return split_locomo_stratified(
        data=[],  # Not used
        qa_items=qa_items,
        train_ratio=split_info["train_ratio"],
        val_ratio=split_info["val_ratio"],
        seed=saved_seed,
        exclude_adversarial=split_info.get("exclude_adversarial", exclude_adversarial),
        save_path=None,  # Don't re-save
    )


# Keep old function for backward compatibility
def split_locomo_for_training(
    data: List[Dict],
    qa_items: List[Dict],
    train_ratio: float = 0.10,
    val_ratio: float = 0.05,
    seed: int = 42,
    exclude_adversarial: bool = True,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    [DEPRECATED] Use split_locomo_stratified() instead for stratified sampling.
    
    Split LoCoMo QA items for GRPO training.
    Splits by conversation to avoid data leakage (but not stratified by category).
    """
    print("\n⚠️  WARNING: Using deprecated split_locomo_for_training()")
    print("   Consider using split_locomo_stratified() for balanced category sampling.\n")
    
    random.seed(seed)
    
    if exclude_adversarial:
        qa_items = [q for q in qa_items if int(q.get("category", 0)) != 5]
    
    # Group by conversation
    conv_to_qa = {}
    for q in qa_items:
        conv_idx = q["conv_idx"]
        if conv_idx not in conv_to_qa:
            conv_to_qa[conv_idx] = []
        conv_to_qa[conv_idx].append(q)
    
    conv_indices = list(conv_to_qa.keys())
    random.shuffle(conv_indices)
    
    total_questions = len(qa_items)
    target_train = int(total_questions * train_ratio)
    target_val = int(total_questions * val_ratio)
    
    train_qa, val_qa, test_qa = [], [], []
    train_count, val_count = 0, 0
    
    for conv_idx in conv_indices:
        conv_questions = conv_to_qa[conv_idx]
        
        if train_count < target_train:
            train_qa.extend(conv_questions)
            train_count += len(conv_questions)
        elif val_count < target_val:
            val_qa.extend(conv_questions)
            val_count += len(conv_questions)
        else:
            test_qa.extend(conv_questions)
    
    print(f"Split (excluding cat 5): Train={len(train_qa)}, Val={len(val_qa)}, Test={len(test_qa)}")
    print(f"Category distribution in train:")
    for cat in [1, 2, 3, 4]:
        count = sum(1 for q in train_qa if int(q.get("category", 0)) == cat)
        print(f"  Category {cat}: {count}")
    
    return train_qa, val_qa, test_qa


if __name__ == "__main__":
    _script_dir = Path(__file__).resolve().parent
    _repo_root = _script_dir.parent.parent.parent  # src/memora/rl -> repo root
    data_path = str(_repo_root / "app" / "locomo" / "data" / "locomo10.json")
    split_save_path = str(_script_dir / "data_split.json")
    
    # Load data
    conversations, qa_items = load_locomo_data(data_path)
    print(f"Loaded {len(conversations)} conversations, {len(qa_items)} QA pairs")
    
    # Create stratified split (1:1:8 ratio)
    train_qa, val_qa, test_qa = split_locomo_stratified(
        conversations, 
        qa_items,
        train_ratio=0.10,  # 10% train
        val_ratio=0.10,    # 10% val  
        seed=42,
        exclude_adversarial=True,
        save_path=split_save_path,
    )
    
    print(f"\nFinal split: Train={len(train_qa)}, Val={len(val_qa)}, Test={len(test_qa)}")
