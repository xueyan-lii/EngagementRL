#!/usr/bin/env bash
# Launcher for engagement-conditioned GRPO (smoke or full).
#   ./run_smoke_engagement.sh server [config]  -> rollout/reward server (.venv-eval)
#   ./run_smoke_engagement.sh train  [config]  -> GRPO trainer (.venv-train)
# config defaults to smoke_engagement; pass full_engagement (Qwen2.5 x UserLM)
# or osim_qwen3 (Qwen3-8B x OSIM, reference-solution teacher) for a real run.
#
# Trainer GPUs are env-overridable and MUST NOT overlap the server GPUs pinned
# by gpu_ids in the config -- e.g. osim_qwen3.yaml puts the three server models
# on 0/1/2, so:  TRAIN_GPUS=3,4,5 ./run_smoke_engagement.sh train osim_qwen3
set -euo pipefail
cd /home/xli/user_modelling/PedagogicalRL
CONFIG="${2:-smoke_engagement}"

# The cached HF OAuth token is stale (401 even on public repos); every model
# and the dataset are already in the local cache.
export HF_HUB_OFFLINE=1

case "${1:-}" in
  server)
    source ../env_eval.sh
    export PYTHONUNBUFFERED=1
    exec ../.venv-eval/bin/python train_server.py --config-name "$CONFIG"
    ;;
  train)
    export PYTHONUNBUFFERED=1
    export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS:-2,3,7}
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    exec ../.venv-train/bin/accelerate launch \
      --config_file ${DEEPSPEED_CFG:-config/deepspeed/zero2_3GPU.yaml} \
      train_rl_engagement.py --config-name "$CONFIG"
    ;;
  *)
    echo "usage: $0 {server|train}" >&2
    exit 1
    ;;
esac
