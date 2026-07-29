#!/usr/bin/env bash
# Launcher for engagement-conditioned GRPO (smoke or full).
#   ./run_smoke_engagement.sh server [config]  -> rollout/reward server (.venv-eval)
#   ./run_smoke_engagement.sh train  [config]  -> GRPO trainer (.venv-train, GPUs 1,2)
# config defaults to smoke_engagement; pass full_engagement for the real run.
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
    export CUDA_VISIBLE_DEVICES=2,3,7
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    exec ../.venv-train/bin/accelerate launch \
      --config_file config/deepspeed/zero2_3GPU.yaml \
      train_rl_engagement.py --config-name "$CONFIG"
    ;;
  *)
    echo "usage: $0 {server|train}" >&2
    exit 1
    ;;
esac
