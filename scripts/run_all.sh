#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

# Must be run after installing requirements in the active environment.
# Env overrides allow a minimal smoke run: TRAIN_STEPS=2 DATA_MODE=fixture ...
DATA_MODE="${DATA_MODE:-real}"
TRAIN_STEPS="${TRAIN_STEPS:-180}"
EVAL_VAL_SIZE="${EVAL_VAL_SIZE:-80}"
EVAL_TEST_SIZE="${EVAL_TEST_SIZE:-250}"
ALLOW_FIXTURE=()
if [[ "$DATA_MODE" == "fixture" ]]; then ALLOW_FIXTURE=(--allow-fixture); fi

python -m scripts.prepare_data --mode "$DATA_MODE"
for task in amazon_sent agnews_topic amazon_lm yelp_lm; do
  python -m scripts.train_task --task "$task" --max-steps "$TRAIN_STEPS"
done
python -m scripts.extract_vectors
python -m scripts.run_experiments --val-size "$EVAL_VAL_SIZE" --test-size "$EVAL_TEST_SIZE" "${ALLOW_FIXTURE[@]}"
python -m scripts.plot_results
printf '\nFinished. Metrics: results/metrics.csv; plots: results/figures/.\n'
