#!/bin/zsh
# gemini-4-argon rerun of the baseline protocol (docs/baseline-protocol.md): same commands, prompts, data and
# threshold; only the backbone changes. Every job starts at once; each Memora-P starts on a copy of its S store
# as soon as that build completes. Gemini key only (ANTHROPIC_API_KEY blanked); if it stops working, jobs fail.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG=$REPO/runs-argon
export ANTHROPIC_API_KEY= MEMOBENCH_MODEL=gemini-4-argon MEMOBENCH_ROOT=$LOG \
  MEMOBENCH_BUDGET_USD=1000000 MEMOBENCH_APPROVAL_USD=1000000 \
  MEMOBENCH_MAX_ATTEMPTS=30 MEMOBENCH_EMBED_MAX_ATTEMPTS=30 MEMOBENCH_GEMINI_CONCURRENCY=128

C=(memory.enable_episodic_memory=True memory.use_segments_as_episodic=True memory.enable_cue_index=True memory.update_score_threshold=0.9291)
SUB=$(python3 -c "import json;print(json.load(open('$REPO/docs/lme_subset24.json'))['subset_idx'])")
L=(memory.update_score_threshold=0.9291 "eval.subset_idx=\"$SUB\"" eval.max_workers=24 eval.parallel_search=true)

job() {  # job <app> <log name> <args...>
  local app=$1 name=$2; shift 2
  (cd $REPO/vendor/Memora/app/$app && MEMOBENCH_RUN_ID=argon-$app uv run python "$@" > $LOG/$name.log 2>&1
   echo "$name exit $?" >> $LOG/$name.log) &
}
p_after_s() {  # p_after_s <app> <S log name> <S store> <P log name> <args...>
  local app=$1 slog=$2 store=$3 name=$4; shift 4
  (until grep -q "Memory building completed" $LOG/$slog.log 2>/dev/null; do sleep 20; done
   cd $REPO/vendor/Memora/app/$app && rm -rf memory_store/$store-P && cp -R memory_store/$store memory_store/$store-P
   MEMOBENCH_RUN_ID=argon-$app uv run python run_memora.py memory.memory_store=$store-P retrieval.strategy=prompt "$@" \
     > $LOG/$name.log 2>&1; echo "$name exit $?" >> $LOG/$name.log) &
}

# LoCoMo: Memora-S, no-abstraction, Memora-P (on the S store), full context, RAG
job locomo locomo-S run_memora.py memory.memory_store=argon-seg-cue retrieval.strategy=semantic $C
job locomo locomo-noabs run_memora.py memory.memory_store=argon-seg-cue-noabs +memory.embed_field=value retrieval.strategy=semantic $C
p_after_s locomo locomo-S argon-seg-cue locomo-P $C
job locomo locomo-fc run_experiments.py memory.type=full_context
job locomo locomo-rag run_experiments.py memory.type=rag

# LongMemEval: full context + RAG on all 500; Memora-S, no-abstraction, Memora-P on the 24-question subset
job longmemeval lme-fc run_experiments.py memory.type=full_context
job longmemeval lme-rag run_experiments.py memory.type=rag
job longmemeval lme-sub24-S run_memora.py memory.memory_store=argon-lme-sub24 retrieval.strategy=semantic $L
job longmemeval lme-sub24-noabs run_memora.py memory.memory_store=argon-lme-sub24-noabs +memory.embed_field=value retrieval.strategy=semantic $L
p_after_s longmemeval lme-sub24-S argon-lme-sub24 lme-sub24-P $L

wait
echo "argon rerun done"
