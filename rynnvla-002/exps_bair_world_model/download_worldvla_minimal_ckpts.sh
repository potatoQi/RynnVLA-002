#!/bin/bash
set -euo pipefail

repo=${HF_REPO:-Alibaba-DAMO-Academy/WorldVLA}
endpoint=${HF_ENDPOINT:-https://hf-mirror.com}
ckpt_dir=${CKPT_DIR:-../ckpts}
aria2_input=${ARIA2_INPUT:-/tmp/rynnvla-worldvla-aria2.txt}

files=(
  chameleon/tokenizer/text_tokenizer.json
  chameleon/tokenizer/tokenizers_checklist.chk
  chameleon/tokenizer/vqgan.ckpt
  chameleon/tokenizer/vqgan.yaml
  chameleon/starting_point/config.json
  chameleon/starting_point/generation_config.json
  chameleon/starting_point/model-00001-of-00002.safetensors
  chameleon/starting_point/model-00002-of-00002.safetensors
  chameleon/starting_point/model.safetensors.index.json
  base_model/tokenizer.json
  base_model/tokenizer_config.json
  base_model/special_tokens_map.json
)

: > "$aria2_input"
for rel_path in "${files[@]}"; do
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
  echo "all minimal checkpoint files already exist"
else
  aria2c -c -x 16 -s 16 -k 1M -j 3 \
    --retry-wait=5 --max-tries=0 \
    --timeout=60 --connect-timeout=20 \
    --auto-file-renaming=false \
    --allow-overwrite=false \
    --summary-interval=30 \
    -i "$aria2_input"
fi

ln -sfn chameleon/starting_point "$ckpt_dir/starting_point"
