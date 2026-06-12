---
name: scenescape-setup-scene
description: >
  SceneScape scene phase: 3D reconstruction, mesh finalization, camera registration (steps
  11–12), and regulated-topic tracking verification (step 13). Requires calibrate complete.
argument-hint: "<deploy_dir> with scene_name and calibration frames"
---

# SceneScape Setup — Scene (steps 11–13)

Parent skill: [scenescape-setup](../scenescape-setup/SKILL.md).

## Prerequisites

Calibrate complete (`last_completed_step` ≥ 10, frames in `calibration-frames/`).

## Run

```bash
SKILL_DIR=<path-to-scenescape>/.github/skills/scenescape-setup

bash "$SKILL_DIR/scripts/deploy_scenescape.sh" \
  --deploy-dir <deploy_dir> \
  --skill-dir "$SKILL_DIR" \
  --streams <rtsp_url> [...] \
  --camera-ids <id> [...] \
  --scene-name <scene_name> \
  --phase scene
```

## Pass

- `Done. Scene UID: …` in log
- `verify_tracking.sh` reports tracked objects
- Checkpoint `last_completed_step` ≥ 13

## On failure

- Reconstruction: [reconstruction.md](../scenescape-setup/references/reconstruction.md)
- Tracking: [verify-tracking.md](../scenescape-setup/references/verify-tracking.md)

## Agent guardrails

- Use `reconstruct_and_finalize.py` and `verify_tracking.sh` only.
- Do not re-read scene REST API docs unless finalize or tracking fails.
