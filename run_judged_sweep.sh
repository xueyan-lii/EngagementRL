#!/usr/bin/env bash
# Baseline Judge-only sweep: dialogue-level engagement + learning for
# (teacher x student-simulator) cell. No solve-rate probe, no transfer test.
#
# Cells run SEQUENTIALLY -- each loads three models (teacher/simulator/judge)
# on GPUs 0/1/2 by default. Override GPUS to move them.
#
# Results: logs/judged/<run_name>.json (+ _dialogues.jsonl transcripts).
# Collate with:  python utils/collate_judged.py
#
#   N=200 ./run_judged_sweep.sh              # all cells
#   CELLS="qwen3-8b_osim-passive" ./run_judged_sweep.sh   # one cell
set -u
cd "$(dirname "$0")"
source ../env_eval.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1

N=${N:-200}
MAXT=${MAXT:-15}
TG=${TG:-0}; SG=${SG:-1}; JG=${JG:-2}   # teacher / simulator / judge GPUs
OUT=${OUT:-logs/judged}

# name|teacher|simulator|simulator_model|persona
CELLS_ALL=(
  "tutorrl_userlm|eth-nlped/TutorRL-7B|userlm|microsoft/UserLM-8b|"
  "tutorrl_osim-passive|eth-nlped/TutorRL-7B|osim|cmu-lti/osim-8b|prompt_templates/personas/osim_passive.txt"
  "tutorrl_osim-diseng|eth-nlped/TutorRL-7B|osim|cmu-lti/osim-8b|prompt_templates/personas/osim_disengaged.txt"
  "qwen25-7b_userlm|Qwen/Qwen2.5-7B-Instruct|userlm|microsoft/UserLM-8b|"
  "qwen25-7b_osim-passive|Qwen/Qwen2.5-7B-Instruct|osim|cmu-lti/osim-8b|prompt_templates/personas/osim_passive.txt"
  "qwen25-7b_osim-diseng|Qwen/Qwen2.5-7B-Instruct|osim|cmu-lti/osim-8b|prompt_templates/personas/osim_disengaged.txt"
  "qwen3-8b_userlm|Qwen/Qwen3-8B|userlm|microsoft/UserLM-8b|"
  "qwen3-8b_osim-passive|Qwen/Qwen3-8B|osim|cmu-lti/osim-8b|prompt_templates/personas/osim_passive.txt"
  "qwen3-8b_osim-diseng|Qwen/Qwen3-8B|osim|cmu-lti/osim-8b|prompt_templates/personas/osim_disengaged.txt"
)

# Kill only THIS script's leftover engines, by PID lineage. Never match on
# command name: vLLM's GPU processes are called VLLM::EngineCore, so a
# name-based filter matches nothing and would kill other users' jobs.
cleanup_run () {
  local root=$1
  [ -z "$root" ] && return
  local kids
  kids=$(pgrep -P "$root" 2>/dev/null; pstree -p "$root" 2>/dev/null | grep -oP '\(\K[0-9]+')
  for p in $kids $root; do kill -TERM "$p" 2>/dev/null; done
  sleep 8
}

run_cell () {
  IFS='|' read -r name teacher sim sim_model persona <<< "$1"
  if [ -f "$OUT/$name.json" ]; then echo "[skip] $name (already done)"; return; fi
  echo "=========== $name ==========="
  local extra=""
  [ -n "$persona" ] && extra="persona_path=$persona"
  ../.venv-eval/bin/python -B -u eval_judged.py --config-name judged \
    teacher_model.model_name_or_path="$teacher" \
    engagement_model.model_name_or_path="$sim_model" \
    simulator="$sim" run_name="$name" out_dir="$OUT" $extra \
    ++teacher_model.vllm.gpu_ids="[$TG]" \
    ++engagement_model.vllm.gpu_ids="[$SG]" \
    ++judge_model.vllm.gpu_ids="[$JG]" \
    ++dataset.max_val_examples=$N ++generation.max_turns=$MAXT \
    > "$OUT/$name.log" 2>&1
  local rc=$?
  cleanup_run $!
  [ $rc -eq 0 ] && echo "[ok] $name" || echo "[FAIL rc=$rc] $name -- see $OUT/$name.log"
}

mkdir -p "$OUT"
if [ -n "${CELLS:-}" ]; then
  for want in $CELLS; do
    for c in "${CELLS_ALL[@]}"; do
      [ "${c%%|*}" = "$want" ] && run_cell "$c"
    done
  done
else
  for c in "${CELLS_ALL[@]}"; do run_cell "$c"; done
fi
echo "=========== SWEEP DONE ==========="
../.venv-eval/bin/python utils/collate_judged.py --dir "$OUT" 2>/dev/null || true
