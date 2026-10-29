#!/bin/bash
# run_parallel_splits.sh - Run 5 parallel splits across different endpoints
#
# Usage:
#   ./run_parallel_splits.sh                                    # Normal run
#   ./run_parallel_splits.sh --force                            # Kill existing sessions
#   ./run_parallel_splits.sh --skip-existing                    # Only rebuild questions without memories
#   ./run_parallel_splits.sh --force-rebuild                    # Force rebuild all memories from scratch
#   ./run_parallel_splits.sh --output-dir longmemeval_outputs_v2  # Use custom output directory
#
# Manual cleanup commands:
#   Kill sessions: for i in {0..4}; do tmux kill-session -t "longmemeval_split_$i" 2>/dev/null || true; done
#   Clean outputs: rm -rf longmemeval_outputs/split_*
#   Clean memory stores: rm -rf memory_store/memora_split_*

# Parse arguments
FORCE=false
SKIP_EXISTING=false
FORCE_REBUILD=false
OUTPUT_DIR="longmemeval_outputs"

i=1
while [ $i -le $# ]; do
    arg="${!i}"
    if [[ "$arg" == "--force" ]] || [[ "$arg" == "-f" ]]; then
        FORCE=true
    elif [[ "$arg" == "--skip-existing" ]] || [[ "$arg" == "-s" ]]; then
        SKIP_EXISTING=true
        FORCE=true
    elif [[ "$arg" == "--force-rebuild" ]]; then
        FORCE_REBUILD=true
        FORCE=true
    elif [[ "$arg" == "--output-dir" ]] || [[ "$arg" == "-o" ]]; then
        i=$((i + 1))
        OUTPUT_DIR="${!i}"
    fi
    i=$((i + 1))
done

# Configuration
SPLITS=("1:100" "101:200" "201:300" "301:400" "401:500")
WORKERS=4

# Azure OpenAI endpoints - configure with your endpoints
# Uses environment variable AZURE_OPENAI_ENDPOINT as default for all splits
# Override individual splits by editing the array below
DEFAULT_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-}"

ENDPOINTS=(
    "${DEFAULT_ENDPOINT}"
    "${DEFAULT_ENDPOINT}"
    "${DEFAULT_ENDPOINT}"
    "${DEFAULT_ENDPOINT}"
    "${DEFAULT_ENDPOINT}"
)

# Uncomment and edit to use multiple endpoints for load balancing:
# ENDPOINTS=(
#     "https://endpoint-1.openai.azure.com/"
#     "https://endpoint-2.openai.azure.com/"
#     "https://endpoint-3.openai.azure.com/"
#     "https://endpoint-4.openai.azure.com/"
#     "https://endpoint-5.openai.azure.com/"
# )

SESSION_PREFIX="longmemeval_split_"

echo "=================================================="
if [ "$FORCE_REBUILD" = true ]; then
    echo "Force Rebuild Mode - Rebuilding All Memories"
elif [ "$SKIP_EXISTING" = true ]; then
    echo "Retrying Failed Questions (skip_existing mode)"
else
    echo "Starting 5 Parallel LongMemEval Splits"
fi
echo "=================================================="
echo "Output directory: $OUTPUT_DIR"
echo "Splits: ${SPLITS[@]}"
echo "Workers per split: $WORKERS"
echo ""

# Increase file descriptor limit (ChromaDB opens many files per client)
ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null || true
echo "File descriptor limit: $(ulimit -n)"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Clean up memory stores and outputs if force rebuild
if [ "$FORCE_REBUILD" = true ]; then
    echo "Force rebuild enabled - cleaning up existing data..."
    for i in {0..4}; do
        memory_store="memory_store/memora_split_${i}"
        if [ -d "$memory_store" ]; then
            echo "  Deleting memory store: $memory_store"
            rm -rf "$memory_store"
        fi
        output_dir="${OUTPUT_DIR}/split_${i}"
        if [ -d "$output_dir" ]; then
            echo "  Deleting output directory: $output_dir"
            rm -rf "$output_dir"
        fi
    done
    echo "Cleanup complete"
    echo ""
fi

# Check for existing sessions
existing_sessions=0
for i in {0..4}; do
    session_name="${SESSION_PREFIX}${i}"
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "Warning: Session $session_name already exists"
        existing_sessions=$((existing_sessions + 1))
    fi
done

if [ $existing_sessions -gt 0 ]; then
    if [ "$FORCE" = true ]; then
        echo "Force mode: Killing existing sessions..."
        for i in {0..4}; do
            tmux kill-session -t "${SESSION_PREFIX}${i}" 2>/dev/null || true
        done
    else
        read -p "Kill existing sessions and restart? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            for i in {0..4}; do
                tmux kill-session -t "${SESSION_PREFIX}${i}" 2>/dev/null || true
            done
        else
            echo "Exiting to avoid conflicts"
            exit 1
        fi
    fi
fi

# Create tmux sessions for each split
for i in {0..4}; do
    subset="${SPLITS[$i]}"
    endpoint="${ENDPOINTS[$i]}"
    session_name="${SESSION_PREFIX}${i}"
    memory_store="memora_split_${i}"
    split_output_dir="${OUTPUT_DIR}/split_${i}"

    echo "Creating session $session_name for questions $subset"
    echo "  Memory store: $memory_store"
    echo "  Endpoint: $endpoint"
    echo "  Workers: $WORKERS"

    tmux new-session -d -s "$session_name"
    tmux send-keys -t "$session_name" "cd $(pwd)" C-m
    tmux send-keys -t "$session_name" "ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null || true" C-m

    # Build command with conditional flags
    cmd="python run_memora.py eval.subset_idx='$subset' eval.max_workers=$WORKERS eval.parallel_search=true memory.memory_store='$memory_store'"
    if [ "$FORCE_REBUILD" = true ]; then
        cmd="$cmd memory.force_rebuild=true"
    elif [ "$SKIP_EXISTING" = true ]; then
        cmd="$cmd memory.skip_existing=true"
    fi
    if [ -n "$endpoint" ]; then
        cmd="$cmd openai.llm_api_base='$endpoint' openai.embedding_api_base='$endpoint'"
    fi
    cmd="$cmd +general.output_dir='$split_output_dir'"

    tmux send-keys -t "$session_name" "$cmd" C-m
    tmux send-keys -t "$session_name" "echo 'Split $i COMPLETE'" C-m

    echo ""
done

echo "=================================================="
echo "All 5 splits launched!"
echo "=================================================="
echo ""
echo "Monitor progress:"
echo "  ./check_status.sh $OUTPUT_DIR"
echo ""
echo "Attach to a session:"
echo "  tmux attach -t ${SESSION_PREFIX}0"
echo ""
echo "After completion, gather results:"
echo "  ./gather_results.sh $OUTPUT_DIR"
