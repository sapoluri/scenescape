---
name: scenescape-setup-calibrate
description: >
  SceneScape calibration phase: MQTT calibration frame gate (step 9) and mapping service
  health (step 10). Requires bootstrap complete (stack running, video-analytics healthy).
argument-hint: "<deploy_dir> with checkpoint from bootstrap or full inputs"
---

# SceneScape Setup — Calibrate (steps 9–10)

Parent skill: [scenescape-setup](../scenescape-setup/SKILL.md).

## Prerequisites

Bootstrap complete (`last_completed_step` ≥ 8) or run
[scenescape-setup-bootstrap](../scenescape-setup-bootstrap/SKILL.md) first.

## Run

```bash
SKILL_DIR=<path-to-scenescape>/.github/skills/scenescape-setup

bash "$SKILL_DIR/scripts/deploy_scenescape.sh" \
  --deploy-dir <deploy_dir> \
  --skill-dir "$SKILL_DIR" \
  --streams <rtsp_url> [...] \
  --camera-ids <id> [...] \
  --scene-name <scene_name> \
  --phase calibrate
```

## Pass

- Calibration JPEGs under `<deploy_dir>/calibration-frames/`
- `check_mapping_health.sh` exits 0
- Checkpoint `last_completed_step` ≥ 10

## On failure

[runtime-verification.md](../scenescape-setup/references/runtime-verification.md) (Step 9)

## Agent guardrails

- Use `capture_calibration_frames.py` — not manual mosquitto one-liners on the happy path.
- Resume from `.deploy-state.json` when present.
