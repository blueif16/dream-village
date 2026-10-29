#!/bin/bash
# check_status.sh - Check status of all parallel splits
#
# Usage:
#   ./check_status.sh                           # Uses default: longmemeval_outputs
#   ./check_status.sh longmemeval_outputs_v2    # Uses specified directory

SESSION_PREFIX="longmemeval_split_"
NUM_SPLITS=5
OUTPUT_DIR="${1:-longmemeval_outputs}"

echo "=================================================="
echo "LongMemEval Splits Status"
echo "=================================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check tmux sessions
echo "=== Tmux Sessions ==="
running=0
completed=0

for i in $(seq 0 $((NUM_SPLITS - 1))); do
    session_name="${SESSION_PREFIX}${i}"
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "  Split $i: RUNNING"
        running=$((running + 1))
    else
        echo "  Split $i: COMPLETED or NOT STARTED"
        completed=$((completed + 1))
    fi
done

echo ""
echo "Summary: $running running, $completed completed/not started"
echo ""

# Check memory stores
echo "=== Memory Stores ==="
for i in $(seq 0 $((NUM_SPLITS - 1))); do
    memory_store="memory_store/memora_split_${i}"
    if [ -d "$memory_store" ]; then
        chroma_db="$memory_store/chroma.sqlite3"
        if [ -f "$chroma_db" ]; then
            db_size=$(du -h "$chroma_db" | cut -f1)
            echo "  Split $i: EXISTS ($db_size)"
        else
            echo "  Split $i: EXISTS (no DB file yet)"
        fi
    else
        echo "  Split $i: NOT CREATED"
    fi
done
echo ""

# Check output files
echo "=== Output Files ==="
for i in $(seq 0 $((NUM_SPLITS - 1))); do
    split_dir="${OUTPUT_DIR}/split_${i}"
    if [ -d "$split_dir" ]; then
        # Count output lines (JSONL format)
        output_file=$(find "$split_dir" -maxdepth 1 -name "memora_*_output.json" 2>/dev/null | head -1)
        if [ -n "$output_file" ]; then
            line_count=$(wc -l < "$output_file" 2>/dev/null || echo "0")
            echo "  Split $i: $line_count questions processed"
        else
            echo "  Split $i: output dir exists, no output file yet"
        fi

        # Check for score file
        score_file=$(find "$split_dir" -maxdepth 1 -name "memora_*_scores.json" 2>/dev/null | head -1)
        if [ -n "$score_file" ]; then
            echo "           EVALUATION COMPLETE"
        fi
    else
        echo "  Split $i: NOT STARTED"
    fi
done
echo ""

# Overall status
echo "=== Overall ==="
total_processed=0
total_evaluated=0
for i in $(seq 0 $((NUM_SPLITS - 1))); do
    split_dir="${OUTPUT_DIR}/split_${i}"
    output_file=$(find "$split_dir" -maxdepth 1 -name "memora_*_output.json" 2>/dev/null | head -1)
    if [ -n "$output_file" ]; then
        count=$(wc -l < "$output_file" 2>/dev/null || echo "0")
        total_processed=$((total_processed + count))
    fi
    score_file=$(find "$split_dir" -maxdepth 1 -name "memora_*_scores.json" 2>/dev/null | head -1)
    if [ -n "$score_file" ]; then
        total_evaluated=$((total_evaluated + 1))
    fi
done

echo "Total questions processed: $total_processed / 500"
echo "Splits with evaluation complete: $total_evaluated / $NUM_SPLITS"

if [ $total_evaluated -eq $NUM_SPLITS ]; then
    echo ""
    echo "ALL SPLITS COMPLETE! Run ./gather_results.sh to aggregate scores."
fi
