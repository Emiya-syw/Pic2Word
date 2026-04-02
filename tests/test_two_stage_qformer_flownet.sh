#!/usr/bin/env bash
set -euo pipefail

out_file="$(mktemp)"
DRY_RUN=1 bash two_stage_qformer_flownet.sh >"$out_file"

# Ensure both stages are generated with expected key flags
rg --fixed-strings -- 'CUDA_VISIBLE_DEVICES=0\,1\,2\,3\,4\,5\,6\,7' "$out_file" >/dev/null
rg --fixed-strings -- '--training-stage qformer_pretrain' "$out_file" >/dev/null
rg --fixed-strings -- '--training-stage flow' "$out_file" >/dev/null
rg --fixed-strings -- '--qformer-num-query-tokens 4' "$out_file" >/dev/null
rg --fixed-strings -- '--qformer-prompt a\ photo\ of\ \*' "$out_file" >/dev/null
rg --fixed-strings -- '--name qformer10_flow10_stage1_qformer_pretrain' "$out_file" >/dev/null
rg --fixed-strings -- '--name qformer10_flow10_stage2_flow' "$out_file" >/dev/null
rg --fixed-strings -- '--epochs 10' "$out_file" >/dev/null

echo "two_stage_qformer_flownet.sh dry-run command generation check passed"
