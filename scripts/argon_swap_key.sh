#!/bin/zsh
# Move the running Argon Memora jobs (scripts/run_argon.sh) onto the GEMINI_API_KEY now in .env without losing work.
# A job is stopped only once its build has completed: run_memora.py skips building whenever the store directory
# exists, so a job killed mid-build would answer from a half-built store. S jobs also wait until their Memora-P
# copy has started. The restart runs the identical command (build skipped, answers already made are cache hits)
# and logs to <name>-2.log.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG=$REPO/runs-argon
export ANTHROPIC_API_KEY= MEMOBENCH_MODEL=gemini-4-argon MEMOBENCH_ROOT=$LOG \
  MEMOBENCH_BUDGET_USD=1000000 MEMOBENCH_APPROVAL_USD=1000000 \
  MEMOBENCH_MAX_ATTEMPTS=30 MEMOBENCH_EMBED_MAX_ATTEMPTS=30 MEMOBENCH_GEMINI_CONCURRENCY=128

C=(memory.enable_episodic_memory=True memory.use_segments_as_episodic=True memory.enable_cue_index=True memory.update_score_threshold=0.9291)
SUB=$(python3 -c "import json;print(json.load(open('$REPO/docs/lme_subset24.json'))['subset_idx'])")
L=(memory.update_score_threshold=0.9291 "eval.subset_idx=\"$SUB\"" eval.max_workers=24 eval.parallel_search=true)

swap() {  # swap <app> <log name> <store> <wait for P: yes|no> <args...>
  local app=$1 name=$2 store=$3 waitp=$4; shift 4
  (until grep -q "Memory building completed" $LOG/$name.log 2>/dev/null; do sleep 15; done
   if [[ $waitp == yes ]]; then until pgrep -f "memory.memory_store=$store-P " >/dev/null; do sleep 5; done; fi
   pkill -f "memory.memory_store=$store "; sleep 5
   echo "$(date '+%F %T') $name stopped after build; restarting on the new key" >> $LOG/swap.log
   cd $REPO/vendor/Memora/app/$app && MEMOBENCH_RUN_ID=argon-$app uv run python run_memora.py memory.memory_store=$store "$@" \
     > $LOG/$name-2.log 2>&1; echo "$name-2 exit $?" >> $LOG/$name-2.log) &
}

swap locomo locomo-S argon-seg-cue yes retrieval.strategy=semantic $C
swap locomo locomo-noabs argon-seg-cue-noabs no +memory.embed_field=value retrieval.strategy=semantic $C
swap longmemeval lme-sub24-S argon-lme-sub24 yes retrieval.strategy=semantic $L
swap longmemeval lme-sub24-noabs argon-lme-sub24-noabs no +memory.embed_field=value retrieval.strategy=semantic $L
wait
