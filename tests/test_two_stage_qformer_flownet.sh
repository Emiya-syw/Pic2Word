#!/usr/bin/env bash
set -euo pipefail

out_file="$(mktemp)"
DRY_RUN=1 bash two_stage_qformer_flownet.sh >"$out_file"

# Ensure both stages are generated with expected key flags
rg --fixed-strings -- 'CUDA_VISIBLE_DEVICES=0\,1\,2\,3\,4\,5\,6\,7' "$out_file" >/dev/null
rg --fixed-strings -- '--freeze-flow-net' "$out_file" >/dev/null
rg --fixed-strings -- '--lambda-fm 0.0' "$out_file" >/dev/null
rg --fixed-strings -- '--name qformer10_jointflow10_stage1_qformer' "$out_file" >/dev/null
rg --fixed-strings -- '--name qformer10_jointflow10_stage2_joint' "$out_file" >/dev/null
rg --fixed-strings -- '--epochs 20' "$out_file" >/dev/null

echo "two_stage_qformer_flownet.sh dry-run command generation check passed"
