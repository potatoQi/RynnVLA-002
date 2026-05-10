#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

train_record=${TRAIN_RECORD:-../processed_data/bair_robot_pushing_tokens/his_1_world_model_train/record.json}
test_record=${TEST_RECORD:-../processed_data/bair_robot_pushing_tokens/his_1_world_model_test/record.json}
train_script=${TRAIN_SCRIPT:-train_bair_world_model_from_official_awm_pretokenize.sh}
poll_seconds=${POLL_SECONDS:-60}

echo "Waiting for BAIR pretokenized records:"
echo "  train: $train_record"
echo "  test:  $test_record"

while [[ ! -s "$train_record" || ! -s "$test_record" ]]; do
  date "+%Y-%m-%d %H:%M:%S"
  if [[ -e "$train_record" ]]; then
    ls -lh "$train_record"
  else
    echo "train record not ready"
  fi
  if [[ -e "$test_record" ]]; then
    ls -lh "$test_record"
  else
    echo "test record not ready"
  fi
  sleep "$poll_seconds"
done

echo "Pretokenized records are ready; starting training."
exec bash "$script_dir/$train_script" "$@"
