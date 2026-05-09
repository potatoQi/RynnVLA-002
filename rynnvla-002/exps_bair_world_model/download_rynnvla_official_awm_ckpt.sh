#!/bin/bash
set -euo pipefail

repo=${HF_REPO:-Alibaba-DAMO-Academy/RynnVLA-002}
endpoint=${HF_ENDPOINT:-https://hf-mirror.com}
ckpt_dir=${CKPT_DIR:-../ckpts}
model_subdir=${MODEL_SUBDIR:-Action_World_model_512/libero_spatial}
aria2_input=${ARIA2_INPUT:-/tmp/rynnvla-official-awm-aria2.txt}

files=(
  args.json
  config.json
  generation_config.json
  model-00001-of-00003.safetensors
  model-00002-of-00003.safetensors
  model-00003-of-00003.safetensors
  model.safetensors.index.json
)

dst_root="$ckpt_dir/$model_subdir"
mkdir -p "$dst_root"

: > "$aria2_input"
for file_name in "${files[@]}"; do
  rel_path="$model_subdir/$file_name"
  dst="$ckpt_dir/$rel_path"
  if [[ -s "$dst" && ! -f "$dst.aria2" ]]; then
    echo "skip $rel_path"
    continue
  fi
  mkdir -p "$(dirname "$dst")"
  {
    echo "$endpoint/$repo/resolve/main/$rel_path"
    echo "  dir=$(cd "$(dirname "$dst")" && pwd)"
    echo "  out=$(basename "$dst")"
  } >> "$aria2_input"
done

if [[ ! -s "$aria2_input" ]]; then
  echo "all official AWM checkpoint files already exist: $dst_root"
else
  aria2c -c -x 16 -s 16 -k 1M -j 3 \
    --retry-wait=5 --max-tries=0 \
    --timeout=60 --connect-timeout=20 \
    --auto-file-renaming=false \
    --allow-overwrite=false \
    --summary-interval=30 \
    -i "$aria2_input"
fi

echo "official AWM checkpoint dir: $dst_root"
