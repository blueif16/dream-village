# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#!/usr/bin/env python3
"""
Aggregate scores from all splits and calculate overall and per-category averages.

Usage:
    python aggregate_scores.py longmemeval_outputs
    python aggregate_scores.py longmemeval_outputs --num-splits 5
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def load_split_scores(output_dir: Path, num_splits: int = 5) -> List[Dict]:
    """Load score files from all splits (supports any retrieval strategy)."""
    scores = []
    for i in range(num_splits):
        split_dir = output_dir / f"split_{i}"

        # Find score file dynamically
        score_files = list(split_dir.glob("memora_*_scores.json"))

        if score_files:
            score_file = score_files[0]
            with open(score_file, 'r') as f:
                scores.append(json.load(f))
        else:
            print(f"Warning: No score file found in {split_dir}, skipping split {i}")
    return scores


def aggregate_scores(split_scores: List[Dict]) -> Dict:
    """
    Aggregate scores from all splits.
    Calculate weighted averages based on question counts per category.
    """
    all_metrics = set()
    all_categories = set()

    for split in split_scores:
        if "mean_scores_per_question_type" in split:
            per_type = split["mean_scores_per_question_type"]
            all_metrics.update(k for k in per_type.keys() if k != "count")
            if "count" in per_type:
                all_categories.update(per_type["count"].keys())

    category_sums = defaultdict(lambda: defaultdict(float))
    category_counts = defaultdict(int)
    overall_sums = defaultdict(float)
    total_questions = 0

    for split in split_scores:
        if "mean_scores_per_question_type" not in split:
            continue

        per_type = split["mean_scores_per_question_type"]
        counts = per_type.get("count", {})

        for metric in all_metrics:
            if metric in per_type:
                metric_scores = per_type[metric]
                for category in all_categories:
                    if category in metric_scores and category in counts:
                        count = counts[category]
                        score = metric_scores[category]
                        category_sums[metric][category] += score * count

        for category, count in counts.items():
            category_counts[category] += count

        if "overall_mean_scores" in split:
            split_total = split.get("summary", {}).get("total_questions", 0)
            for metric, score in split["overall_mean_scores"].items():
                overall_sums[metric] += score * split_total
            total_questions += split_total

    # Calculate averages
    per_category_averages = {}
    for metric in all_metrics:
        per_category_averages[metric] = {}
        for category in all_categories:
            if category_counts[category] > 0:
                per_category_averages[metric][category] = round(
                    category_sums[metric][category] / category_counts[category], 4
                )

    per_category_averages["count"] = dict(category_counts)

    overall_averages = {}
    if total_questions > 0:
        for metric, total_score in overall_sums.items():
            overall_averages[metric] = round(total_score / total_questions, 4)

    return {
        "mean_scores_per_question_type": per_category_averages,
        "overall_mean_scores": overall_averages,
        "summary": {
            "total_questions": total_questions,
            "num_splits_aggregated": len(split_scores),
            "question_types": sorted(all_categories),
        },
    }


def print_scores(scores: Dict):
    """Pretty-print the aggregated scores."""
    print(f"\n{'='*70}")
    print("AGGREGATED SCORES ACROSS ALL SPLITS")
    print(f"{'='*70}")

    per_type = scores["mean_scores_per_question_type"]
    counts = per_type.get("count", {})
    metrics = [k for k in per_type.keys() if k != "count"]

    # Header
    header = f"{'Question Type':<35} | {'Count':>5}"
    for m in sorted(metrics):
        header += f" | {m:>10}"
    print(header)
    print("-" * len(header))

    # Per-type scores
    for category in sorted(counts.keys()):
        row = f"{category:<35} | {counts[category]:>5}"
        for m in sorted(metrics):
            score = per_type.get(m, {}).get(category, 0)
            row += f" | {score:>10.4f}"
        print(row)

    # Overall
    print("-" * len(header))
    overall = scores["overall_mean_scores"]
    total = scores["summary"]["total_questions"]
    row = f"{'OVERALL':<35} | {total:>5}"
    for m in sorted(metrics):
        row += f" | {overall.get(m, 0):>10.4f}"
    print(row)
    print(f"{'='*70}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python aggregate_scores.py <output_dir> [--num-splits N]")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    num_splits = 5

    if "--num-splits" in sys.argv:
        idx = sys.argv.index("--num-splits")
        num_splits = int(sys.argv[idx + 1])

    if not output_dir.exists():
        print(f"Error: {output_dir} does not exist")
        sys.exit(1)

    print(f"Loading scores from {output_dir} ({num_splits} splits)...")
    split_scores = load_split_scores(output_dir, num_splits)

    if not split_scores:
        print("Error: No score files found")
        sys.exit(1)

    print(f"Found {len(split_scores)} split score files")

    aggregated = aggregate_scores(split_scores)
    print_scores(aggregated)

    # Save aggregated scores
    output_file = output_dir / "aggregated_scores.json"
    with open(output_file, "w") as f:
        json.dump(aggregated, f, indent=4)
    print(f"Aggregated scores saved to: {output_file}")


if __name__ == "__main__":
    main()
