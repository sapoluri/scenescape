#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end SceneScape deployment orchestrator with checkpoint resume.
#
# Usage:
#   deploy_scenescape.sh \
#     --deploy-dir <dir> \
#     --skill-dir <scenescape-setup-path> \
#     --streams <url> [<url> ...] \
#     --camera-ids <id> [<id> ...] \
#     --scene-name <name> \
#     [--phase all|bootstrap|calibrate|scene] \
#     [--resume|--fresh]

set -euo pipefail

DEPLOY_DIR=""
SKILL_DIR=""
SCENE_NAME=""
PHASE="all"
RESUME_MODE="auto"
declare -a STREAMS=()
declare -a CAMERA_IDS=()

STATE_FILE=""
LOG_FILE=""
DETECTION_MODELS_PID=""

usage() {
  cat <<'EOF'
Usage: deploy_scenescape.sh --deploy-dir DIR --skill-dir DIR \
  --streams URL [URL ...] --camera-ids ID [ID ...] --scene-name NAME \
  [--phase all|bootstrap|calibrate|scene] [--resume|--fresh]
EOF
}

log() {
  # shellcheck disable=SC2329
  echo "$*" | tee -a "$LOG_FILE"
}

state_read() {
  python3 - "$STATE_FILE" "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
if not path.is_file():
  print("", end="")
  sys.exit(0)
data = json.loads(path.read_text(encoding="utf-8"))
value = data.get(key, "")
if isinstance(value, list):
  print(" ".join(value))
else:
  print(value)
PY
}

state_write() {
  DEPLOY_DIR="$DEPLOY_DIR" SKILL_DIR="$SKILL_DIR" SCENE_NAME="$SCENE_NAME" \
  STREAMS_JSON=$(printf '%s\n' "${STREAMS[@]}" | python3 -c 'import json,sys; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))') \
  CAMERA_IDS_JSON=$(printf '%s\n' "${CAMERA_IDS[@]}" | python3 -c 'import json,sys; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))') \
  python3 - "$STATE_FILE" "$1" "${2:-}" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
step = int(sys.argv[2])
extra_key = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else ""
extra_val = os.environ.get("STATE_EXTRA", "")

data = {}
if path.is_file():
  data = json.loads(path.read_text(encoding="utf-8"))

data.update({
  "version": 1,
  "last_completed_step": step,
  "deploy_dir": os.environ["DEPLOY_DIR"],
  "skill_dir": os.environ["SKILL_DIR"],
  "scene_name": os.environ["SCENE_NAME"],
  "streams": json.loads(os.environ["STREAMS_JSON"]),
  "camera_ids": json.loads(os.environ["CAMERA_IDS_JSON"]),
})
if extra_key:
  data[extra_key] = extra_val

path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
}

phase_start_step() {
  case "$1" in
    bootstrap) echo 6 ;;
    calibrate) echo 9 ;;
    scene) echo 12 ;;
    all) echo 6 ;;
    *) echo "Unknown phase: $1" >&2; exit 2 ;;
  esac
}

phase_end_step() {
  case "$1" in
    bootstrap) echo 8 ;;
    calibrate) echo 10 ;;
    scene) echo 13 ;;
    all) echo 13 ;;
    *) echo "Unknown phase: $1" >&2; exit 2 ;;
  esac
}

connect_rtsp_hosts_to_network() {
  local net_name
  net_name=$(docker network ls --format '{{.Name}}' | grep '_scenescape$' | head -1 || true)
  [[ -n "$net_name" ]] || return 0

  local stream host container
  for stream in "${STREAMS[@]}"; do
    host=$(python3 -c 'from urllib.parse import urlparse; import sys; print(urlparse(sys.argv[1]).hostname or "")' "$stream")
    [[ -n "$host" ]] || continue
    container=$(docker ps --format '{{.Names}}' | grep -E "${host}" | head -1 || true)
    if [[ -n "$container" ]]; then
      docker network connect "$net_name" "$container" 2>/dev/null || true
    fi
  done
}

step_bootstrap() {
  log "STEP 6: bootstrap deployment files"
  python3 "$SKILL_DIR/scripts/bootstrap_deploy.py" \
    --deploy-dir "$DEPLOY_DIR" \
    --skill-dir "$SKILL_DIR" \
    --camera-ids "${CAMERA_IDS[@]}" \
    --streams "${STREAMS[@]}" >>"$LOG_FILE" 2>&1
  state_write 6
  log "STEP 6: PASS"
}

step_warmup_rtsp() {
  log "STEP 7: parallel warmup, models, RTSP, video-analytics"
  cd "$DEPLOY_DIR"
  bash scripts/parallel_warmup.sh >>"$LOG_FILE" 2>&1

  bash scripts/download_detection_models.sh >>"$LOG_FILE" 2>&1 &
  DETECTION_MODELS_PID=$!

  connect_rtsp_hosts_to_network
  sleep 15
  bash scripts/check_video_analytics.sh "$DEPLOY_DIR" >>"$LOG_FILE" 2>&1

  if ! bash scripts/verify_rtsp.sh "$DEPLOY_DIR" "${STREAMS[@]}" >>"$LOG_FILE" 2>&1; then
  log "STEP 7: FAIL (RTSP)"
    exit 1
  fi

  if [[ -n "$DETECTION_MODELS_PID" ]] && kill -0 "$DETECTION_MODELS_PID" 2>/dev/null; then
    wait "$DETECTION_MODELS_PID" >>"$LOG_FILE" 2>&1 || true
  fi
  DETECTION_MODELS_PID=""

  docker compose restart video-analytics >>"$LOG_FILE" 2>&1
  sleep 20
  bash scripts/check_video_analytics.sh "$DEPLOY_DIR" >>"$LOG_FILE" 2>&1

  state_write 7
  log "STEP 7: PASS"
}

step_full_stack() {
  log "STEP 8: bring up full stack"
  cd "$DEPLOY_DIR"
  docker compose --profile mapping up -d >>"$LOG_FILE" 2>&1
  docker compose restart video-analytics >>"$LOG_FILE" 2>&1
  sleep 30

  for svc in broker ntpserv pgserver web scene mapping; do
    health=$(docker compose --profile mapping ps "$svc" --format '{{.Health}}' 2>/dev/null | head -1)
    status=$(docker compose --profile mapping ps "$svc" --format '{{.Status}}' 2>/dev/null | head -1)
    if [[ "$status" != *"Up"* ]]; then
      log "STEP 8: FAIL ($svc not running)"
      exit 1
    fi
    log "  $svc: ${health:-n/a} ($status)"
  done

  state_write 8
  log "STEP 8: PASS"
}

step_calibration() {
  log "STEP 9: calibration frame gate"
  local frames_dir="$DEPLOY_DIR/calibration-frames"
  cd "$DEPLOY_DIR"
  python3 scripts/capture_calibration_frames.py \
    --deploy-dir "$DEPLOY_DIR" \
    --cameras "${CAMERA_IDS[@]}" \
    --out-dir "$frames_dir" >>"$LOG_FILE" 2>&1

  STATE_EXTRA="$frames_dir" state_write 9 "frames_dir"
  log "STEP 9: PASS"
}

step_mapping_health() {
  log "STEP 10: mapping service health"
  cd "$DEPLOY_DIR"
  bash scripts/check_mapping_health.sh 30 10 >>"$LOG_FILE" 2>&1
  state_write 10
  log "STEP 10: PASS"
}

step_reconstruct() {
  log "STEP 11-12: reconstruction and scene finalization"
  local frames_dir
  frames_dir=$(state_read frames_dir)
  if [[ -z "$frames_dir" ]]; then
    frames_dir="$DEPLOY_DIR/calibration-frames"
  fi

  cd "$DEPLOY_DIR"
  scene_uid=$(
    python3 scripts/reconstruct_and_finalize.py \
      --deploy-dir "$DEPLOY_DIR" \
      --frames-dir "$frames_dir" \
      --cameras "${CAMERA_IDS[@]}" \
      --scene-name "$SCENE_NAME" 2>>"$LOG_FILE" \
      | tee -a "$LOG_FILE" \
      | awk '/^Done\. Scene UID:/ {print $NF}'
  )

  if [[ -z "$scene_uid" ]]; then
    log "STEP 12: FAIL (no scene UID)"
    exit 1
  fi

  STATE_EXTRA="$scene_uid" state_write 12 "scene_uid"
  log "STEP 12: PASS (scene_uid=$scene_uid)"
}

step_tracking() {
  log "STEP 13: verify object tracking"
  local scene_uid
  scene_uid=$(state_read scene_uid)
  if [[ -z "$scene_uid" ]]; then
    log "STEP 13: FAIL (scene_uid missing in state)"
    exit 1
  fi

  cd "$DEPLOY_DIR"
  docker compose logs scene --tail 30 2>&1 \
    | grep -E 'NEW SCENE|Subscribed to scenescape/data/camera' \
    | tail -10 >>"$LOG_FILE" || true

  bash scripts/verify_tracking.sh "$DEPLOY_DIR" "$scene_uid" 120 >>"$LOG_FILE" 2>&1
  state_write 13
  log "STEP 13: PASS"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deploy-dir) DEPLOY_DIR=$(realpath "$2"); shift 2 ;;
    --skill-dir) SKILL_DIR=$(realpath "$2"); shift 2 ;;
    --scene-name) SCENE_NAME="$2"; shift 2 ;;
    --streams)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do STREAMS+=("$1"); shift; done
      ;;
    --camera-ids)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do CAMERA_IDS+=("$1"); shift; done
      ;;
    --phase) PHASE="$2"; shift 2 ;;
    --resume) RESUME_MODE="auto"; shift ;;
    --fresh) RESUME_MODE="fresh"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$DEPLOY_DIR" && -n "$SKILL_DIR" && -n "$SCENE_NAME" ]] || { usage; exit 2; }
[[ ${#STREAMS[@]} -gt 0 && ${#CAMERA_IDS[@]} -gt 0 ]] || { echo "streams and camera-ids required" >&2; exit 2; }
[[ ${#STREAMS[@]} -eq ${#CAMERA_IDS[@]} ]] || { echo "streams and camera-ids length mismatch" >&2; exit 2; }

STATE_FILE="$DEPLOY_DIR/.deploy-state.json"
LOG_FILE="$DEPLOY_DIR/deploy.log"
mkdir -p "$DEPLOY_DIR"
touch "$LOG_FILE"

if [[ "$RESUME_MODE" == "fresh" && -f "$STATE_FILE" ]]; then
  rm -f "$STATE_FILE"
fi

LAST_STEP=0
if [[ "$RESUME_MODE" == "auto" && -f "$STATE_FILE" ]]; then
  LAST_STEP=$(state_read last_completed_step)
  LAST_STEP=${LAST_STEP:-0}
  saved_scene=$(state_read scene_name)
  if [[ -n "$saved_scene" && "$saved_scene" != "$SCENE_NAME" ]]; then
    echo "WARN: scene_name differs from checkpoint; use --fresh to restart" >&2
  fi
fi

START_STEP=$(phase_start_step "$PHASE")
END_STEP=$(phase_end_step "$PHASE")
if [[ "$RESUME_MODE" == "auto" && "$LAST_STEP" =~ ^[0-9]+$ && "$LAST_STEP" -ge "$START_STEP" ]]; then
  START_STEP=$((LAST_STEP + 1))
fi
if [[ "$PHASE" == "scene" ]]; then
  frames_dir=$(state_read frames_dir)
  if [[ -z "$frames_dir" || ! -d "$frames_dir" ]]; then
    START_STEP=9
  fi
fi

log "=== deploy_scenescape phase=$PHASE steps=${START_STEP}-${END_STEP} (checkpoint=${LAST_STEP}) ==="

run_step() {
  case "$1" in
    6) step_bootstrap ;;
    7) step_warmup_rtsp ;;
    8) step_full_stack ;;
    9) step_calibration ;;
    10) step_mapping_health ;;
    12) step_reconstruct ;;
    13) step_tracking ;;
    *) return 0 ;;
  esac
}

for step in 6 7 8 9 10 12 13; do
  if (( step < START_STEP || step > END_STEP )); then
    continue
  fi
  run_step "$step"
done

scene_uid=$(state_read scene_uid)
log "=== DEPLOY COMPLETE scene_uid=${scene_uid:-n/a} supass=$(cat "$DEPLOY_DIR/secrets/supass" 2>/dev/null || echo n/a) ==="
