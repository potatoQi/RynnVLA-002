#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export DATA_CONFIG_TRAIN=${DATA_CONFIG_TRAIN:-../configs/bair_robot_pushing/his_1_awm_nopretokenize_train.yaml}
export DATA_CONFIG_TEST=${DATA_CONFIG_TEST:-../configs/bair_robot_pushing/his_1_awm_nopretokenize_test.yaml}
export OUT_DIR_TRAIN=${OUT_DIR_TRAIN:-../processed_data/bair_robot_pushing_tokens/his_1_awm_train}
export OUT_DIR_TEST=${OUT_DIR_TEST:-../processed_data/bair_robot_pushing_tokens/his_1_awm_test}

exec bash "$script_dir/prepare_bair_pretokenize.sh" "$@"
