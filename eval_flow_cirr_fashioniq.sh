#!/usr/bin/env bash
set -euo pipefail

# Usage example:
#   bash eval_flow_cirr_fashioniq.sh
# Optional env overrides:
#   RESUME=/path/to/checkpoint.pt GPU_ID=0 CUDA_VISIBLE_DEVICES=0,1 EVAL_CSV_PATH=./logs/my_eval.csv bash eval_flow_cirr_fashioniq.sh

GPU_ID="${GPU_ID:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
RESUME="${RESUME:-/home/sunyw/CIR/Pic2Word/logs/fm_composed_geodesic_image2/checkpoints/epoch_10.pt}"
MODEL_NAME="${MODEL_NAME:-ViT-L/14}"
LOSS_TYPE="${LOSS_TYPE:-global}"
EVAL_CIRR_TEST="${EVAL_CIRR_TEST:-0}"  # 1 => additionally run CIRR test split
EVAL_CSV_PATH="${EVAL_CSV_PATH:-./logs/eval_flow_metrics.csv}"

# Global flow config (kept aligned with existing fashioniq.sh defaults)
flow_conditioning="${FLOW_CONDITIONING:-enabled}"
flow_start_source="${FLOW_START_SOURCE:-text}"
flow_condition_source="${FLOW_CONDITION_SOURCE:-image}"
flow_compose_method="${FLOW_COMPOSE_METHOD:-pic2word}"
flow_pic2word_marker="${FLOW_PIC2WORD_MARKER:-*}"
flow_start_text_weight="${FLOW_START_TEXT_WEIGHT:-1.0}"
flow_start_image_weight="${FLOW_START_IMAGE_WEIGHT:-1.0}"
flow_condition_text_weight="${FLOW_CONDITION_TEXT_WEIGHT:-1.0}"
flow_condition_image_weight="${FLOW_CONDITION_IMAGE_WEIGHT:-1.0}"
global_start_noise_std="${GLOBAL_START_NOISE_STD:-0.0}"
flow_path_type="${FLOW_PATH_TYPE:-linear}" # linear | geodesic
flow_geodesic_eps="${FLOW_GEODESIC_EPS:-1e-4}"
flow_step_norm_mode="${FLOW_STEP_NORM_MODE:-on}" # on | off | auto
flow_step_norm_type="${FLOW_STEP_NORM_TYPE:-expmap}" # l2 | expmap
flow_hybrid_geodesic_steps="${FLOW_HYBRID_GEODESIC_STEPS:-0}"
disable_delta="${DISABLE_DELTA:-1}" # match your training script default
disable_cond_gate="${DISABLE_COND_GATE:-0}"
flow_block_type="${FLOW_BLOCK_TYPE:-film}"       # residual | film
flow_film_expansion="${FLOW_FILM_EXPANSION:-2}"      # used when FLOW_BLOCK_TYPE=film

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
  --flow-path-type "${flow_path_type}"
  --flow-geodesic-eps "${flow_geodesic_eps}"
  --flow-step-norm-mode "${flow_step_norm_mode}"
  --flow-step-norm-type "${flow_step_norm_type}"
  --flow-hybrid-geodesic-steps "${flow_hybrid_geodesic_steps}"
  --global-flow-block-type "${flow_block_type}"
  --global-flow-film-expansion "${flow_film_expansion}"
  --global-flow-pic2word-topk-text "2"
)

if [[ "${disable_delta}" == "1" ]]; then
  extra_flow_args+=(--global-flow-disable-delta)
fi

if [[ "${disable_cond_gate}" == "1" ]]; then
  extra_flow_args+=(--global-flow-disable-cond-gate)
fi

run_eval() {
  local mode="$1"
  local source="${2:-}"

  echo "=================================================="
  echo "Running eval mode: ${mode} ${source:+(source=${source})}"
  echo "Checkpoint: ${RESUME}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, GPU_ID=${GPU_ID}"
  echo "Flow cfg: conditioning=${flow_conditioning}, disable_delta=${disable_delta}, disable_cond_gate=${disable_cond_gate}, path=${flow_path_type}, block=${flow_block_type}, film_expansion=${flow_film_expansion}"
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
    --flow-num-steps 32
    "${extra_flow_args[@]}"
  )

  if [[ -n "${source}" ]]; then
    cmd+=(--source-data "${source}")
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}"
}

# 1) CIRR validation split
# run_eval "cirr"

# Optional: CIRR test split (will write jsons under res_cirr/)
# if [[ "${EVAL_CIRR_TEST}" == "1" ]]; then
#   run_eval "cirr_test"
# fi

# 2) FashionIQ categories
# for cloth_type in shirt dress toptee; do
#   run_eval "fashion" "${cloth_type}"
# done

for cloth_type in dress; do
  run_eval "fashion" "${cloth_type}"
done

echo "All evaluations finished."
