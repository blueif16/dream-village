#!/bin/zsh
# LongMemEval-500 Memora chain, each stage starting as soon as its input exists:
#   [prefetch ABC done] -> main build + answer S  ||  noabs prefetch C -> noabs build + answer
#                       -> (main build done) copy store -> answer P
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
APP=$REPO/vendor/Memora/app/longmemeval
export MEMOBENCH_BUDGET_USD=110 MEMOBENCH_MAX_ATTEMPTS=30
COMMON=(memory.update_score_threshold=0.9291 eval.subset_idx=-1 eval.max_workers=40 eval.parallel_search=true)

# start only after the main prefetch (phases A-C) has finished
while pgrep -f "lme_prefetch.py --phases ABC" >/dev/null; do sleep 30; done

cd $APP
# main store: build (cache-fed) + Memora-S answers + judge
(MEMOBENCH_RUN_ID=lme-full uv run python run_memora.py memory.memory_store=lme-main retrieval.strategy=semantic "${COMMON[@]}" \
   > $REPO/runs/lme-full-S.log 2>&1; echo "S exit $?" >> $REPO/runs/lme-full-S.log) &

# no-abstraction: its own embeddings (values) prefetched, then build + answers + judge
(cd $REPO && MEMOBENCH_RUN_ID=lme-noabs-prefetch uv run python scripts/lme_prefetch.py --phases C eval.subset_idx=-1 \
   memory.update_score_threshold=0.9291 retrieval.strategy=semantic +memory.embed_field=value > $REPO/runs/lme-noabs-prefetch.log 2>&1
 cd $APP && MEMOBENCH_RUN_ID=lme-full uv run python run_memora.py memory.memory_store=lme-noabs +memory.embed_field=value \
   retrieval.strategy=semantic "${COMMON[@]}" > $REPO/runs/lme-full-noabs.log 2>&1; echo "noabs exit $?" >> $REPO/runs/lme-full-noabs.log) &

# Memora-P on a copy of the main store once its build is complete
(until grep -q "Memory building completed" $REPO/runs/lme-full-S.log 2>/dev/null; do sleep 20; done
 rm -rf memory_store/lme-main-P && cp -R memory_store/lme-main memory_store/lme-main-P
 MEMOBENCH_RUN_ID=lme-full uv run python run_memora.py memory.memory_store=lme-main-P retrieval.strategy=prompt "${COMMON[@]}" \
   > $REPO/runs/lme-full-P.log 2>&1; echo "P exit $?" >> $REPO/runs/lme-full-P.log) &

wait
echo "LME Memora chain done"
