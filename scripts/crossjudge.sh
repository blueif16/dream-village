#!/bin/zsh
# Re-judge a finished answers file with the other backbone's judge (same judge prompt, structured label).
#   crossjudge.sh <haiku|argon> <locomo|longmemeval> <label> <path/to/*_output.json>
# haiku: Batch API on the Claude key, Track A ledger (runs/, $110 cap). argon: Gemini key only (Claude blanked).
# Writes runs-argon/crossjudge/<label>__judge-<judge>_{eval,scores}.json. Re-judging with the original judge is
# all cache hits and reproduces the original *_eval.json exactly (checked on LoCoMo + LongMemEval full context).
set -eu
judge=$1 app=$2 label=$3 answers=$4
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT=$REPO/runs-argon/crossjudge
mkdir -p $OUT
export PYTHONPATH=$REPO:$REPO/vendor/Memora/src MEMOBENCH_MAX_ATTEMPTS=30
if [[ $judge == haiku ]]; then
  export MEMOBENCH_ROOT=$REPO/runs MEMOBENCH_RUN_ID=crossjudge-haiku MEMOBENCH_BUDGET_USD=110
else
  export ANTHROPIC_API_KEY= MEMOBENCH_MODEL=gemini-4-argon MEMOBENCH_ROOT=$REPO/runs-argon MEMOBENCH_RUN_ID=crossjudge-argon \
    MEMOBENCH_BUDGET_USD=1000000 MEMOBENCH_APPROVAL_USD=1000000 MEMOBENCH_GEMINI_CONCURRENCY=128
fi
cd $REPO/vendor/Memora/app/$app
uv run python evals.py +eval.result_file=$answers +eval.metrics_file=$OUT/${label}__judge-${judge}_eval.json \
  +eval.score_file=$OUT/${label}__judge-${judge}_scores.json
