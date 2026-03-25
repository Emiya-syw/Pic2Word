#!/usr/bin/env bash
set -euo pipefail

# Usage example:
#   bash eval_flow_cirr_fashioniq.sh
# Optional env overrides:
#   RESUME=/path/to/checkpoint.pt GPU_ID=0 CUDA_VISIBLE_DEVICES=0,1 EVAL_CSV_PATH=./logs/my_eval.csv bash eval_flow_cirr_fashioniq.sh

GPU_ID="${GPU_ID:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
RESUME="${RESUME:-/home/sunyw/CIR/Pic2Word/logs/fm_composed_1e-4/checkpoints/epoch_12.pt}"
MODEL_NAME="${MODEL_NAME:-ViT-L/14}"
LOSS_TYPE="${LOSS_TYPE:-global}"
EVAL_CIRR_TEST="${EVAL_CIRR_TEST:-0}"  # 1 => additionally run CIRR test split
EVAL_CSV_PATH="${EVAL_CSV_PATH:-./logs/eval_flow_metrics.csv}"

# Global flow config (kept aligned with existing fashioniq.sh defaults)
flow_conditioning="${FLOW_CONDITIONING:-disabled}"
flow_start_source="${FLOW_START_SOURCE:-composed}"
flow_condition_source="${FLOW_CONDITION_SOURCE:-image}"
flow_compose_method="${FLOW_COMPOSE_METHOD:-pic2word}"
flow_pic2word_marker="${FLOW_PIC2WORD_MARKER:-*}"
flow_start_text_weight="${FLOW_START_TEXT_WEIGHT:-1.0}"
flow_start_image_weight="${FLOW_START_IMAGE_WEIGHT:-1.0}"
flow_condition_text_weight="${FLOW_CONDITION_TEXT_WEIGHT:-1.0}"
flow_condition_image_weight="${FLOW_CONDITION_IMAGE_WEIGHT:-1.0}"
global_start_noise_std="${GLOBAL_START_NOISE_STD:-0.0}"

extra_flow_args=(
  --global-flow-conditioning "${flow_conditioning}"
  --global-flow-start-source "${flow_start_source}"
  --global-flow-condition-source "${flow_condition_source}"
  --global-flow-compose-method "${flow_compose_method}"
  --global-flow-pic2word-marker "${flow_pic2word_marker}"
  --global-flow-start-text-weight "${flow_start_text_weight}"
  --global-flow-start-image-weight "${flow_start_image_weight}"
  --global-flow-condition-text-weight "${flow_condition_text_weight}"
  --global-flow-condition-image-weight "${flow_condition_image_weight}"
  --global-start-noise-std "${global_start_noise_std}"
)

run_eval() {
  local mode="$1"
  local source="${2:-}"

  echo "=================================================="
  echo "Running eval mode: ${mode} ${source:+(source=${source})}"
  echo "Checkpoint: ${RESUME}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, GPU_ID=${GPU_ID}"
  echo "=================================================="

  local cmd=(
    python src/eval_retrieval_fm.py
    --openai-pretrained
    --resume "${RESUME}"
    --eval-mode "${mode}"
    --gpu "${GPU_ID}"
    --loss-type "${LOSS_TYPE}"
    --model "${MODEL_NAME}"
    --eval-csv "${EVAL_CSV_PATH}"
    "${extra_flow_args[@]}"
  )

  if [[ -n "${source}" ]]; then
    cmd+=(--source-data "${source}")
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}"
}

# 1) CIRR validation split
run_eval "cirr"

# Optional: CIRR test split (will write jsons under res_cirr/)
# if [[ "${EVAL_CIRR_TEST}" == "1" ]]; then
#   run_eval "cirr_test"
# fi

# 2) FashionIQ categories
for cloth_type in shirt dress toptee; do
  run_eval "fashion" "${cloth_type}"
done

echo "All evaluations finished."
