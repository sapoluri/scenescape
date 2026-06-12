---
name: scenescape-setup-bootstrap
description: >
  SceneScape deployment bootstrap only: generate configs/secrets (step 6), verify RTSP and
  video-analytics (step 7), bring up the full Docker stack (step 8). Use before calibration
  or when infra is not yet running.
argument-hint: "<deploy_dir> with streams, camera_ids, scene_name"
---

# SceneScape Setup — Bootstrap (steps 6–8)

Parent skill: [scenescape-setup](../scenescape-setup/SKILL.md).

## Inputs

Same as parent: `deploy_dir`, `streams`, `camera_ids`, `scene_name`.

## Run

```bash
SKILL_DIR=<path-to-scenescape>/.github/skills/scenescape-setup

bash "$SKILL_DIR/scripts/deploy_scenescape.sh" \
  --deploy-dir <deploy_dir> \
  --skill-dir "$SKILL_DIR" \
  --streams <rtsp_url> [...] \
  --camera-ids <id> [...] \
  --scene-name <scene_name> \
  --phase bootstrap
```

## Pass

- Checkpoint `last_completed_step` ≥ 8 in `<deploy_dir>/.deploy-state.json`
- `docker compose --profile mapping ps` shows core services up

## On failure

[runtime-verification.md](../scenescape-setup/references/runtime-verification.md)

## Agent guardrails

- Use `check_video_analytics.sh` and `verify_rtsp.sh` — not raw log dumps.
- Do not read `docker-compose-template.md` or repo sample compose files.
