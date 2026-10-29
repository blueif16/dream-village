#!/bin/bash
# gather_results.sh - Aggregate scores from all splits
#
# Usage:
#   ./gather_results.sh                           # Uses default: longmemeval_outputs
#   ./gather_results.sh longmemeval_outputs_v2    # Uses specified directory

OUTPUT_DIR="${1:-longmemeval_outputs}"

echo "=================================================="
echo "Aggregating Results from All Splits"
echo "=================================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: $OUTPUT_DIR directory not found"
    exit 1
fi

# Check for score files
missing=0
for i in {0..4}; do
    split_dir="${OUTPUT_DIR}/split_${i}"
    score_file=$(find "$split_dir" -maxdepth 1 -name "memora_*_scores.json" 2>/dev/null | head -1)

    if [ -z "$score_file" ]; then
        echo "Warning: No score file found in $split_dir"
        missing=$((missing + 1))
    fi
done

if [ $missing -eq 5 ]; then
    echo "Error: No score files found in any splits"
    exit 1
fi

if [ $missing -gt 0 ]; then
    echo ""
    echo "Warning: $missing split(s) missing score files"
    echo "Proceeding with available splits..."
    echo ""
fi

# Run aggregation
python3 aggregate_scores.py "$OUTPUT_DIR"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "Results gathered successfully!"
    echo "See: ${OUTPUT_DIR}/aggregated_scores.json"
else
    echo ""
    echo "Error during aggregation (exit code: $exit_code)"
fi
