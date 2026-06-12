---
name: scenescape-setup
description: >
  Deploy a working Intel® SceneScape installation from scratch (outside the repo). Runs
  bootstrap, RTSP/pipeline validation, full stack bring-up, calibration, 3D mapping,
  scene creation, and tracking verification via scripts/deploy_scenescape.sh.
argument-hint: "<deploy_dir> — optional; gather streams, camera_ids, scene_name if not in checkpoint"
---

# SceneScape End-to-End Setup

Deploy from a clean directory. Host needs **Docker**, **docker-compose**, and **Python 3.10+**
with `requests`. Calibration helpers use `eclipse-mosquitto:2` on the deployment network.

## Agent guardrails (token efficiency)

- **Prefer the orchestrator** — run `scripts/deploy_scenescape.sh` once; read stdout and
  `<deploy_dir>/deploy.log` only on failure.
- **Do not read** `references/docker-compose-template.md`, `queuing-config.json`, or repo
  `sample_data/` unless troubleshooting a template bug.
- **Do not** dump raw `docker compose logs`; use `scripts/check_video_analytics.sh` and
  `docker compose logs <svc> --tail 30` with grep.
- **Do not modify** generated `docker-compose.yml` in the deploy dir — fix the skill template
  upstream instead.
- **Resume by default** — if `<deploy_dir>/.deploy-state.json` exists, use `--resume` (default);
  use `--fresh` only when inputs change or the user requests a clean redeploy.
- Load `references/runtime-verification.md` only when a step fails.

## Step 1 — Gather inputs

| Input | Description |
|-------|-------------|
| `deploy_dir` | Writable output directory (e.g. `~/deployment`) |
| `streams` | RTSP URL per camera, in order |
| `camera_ids` | Unique IDs (no `/`), same order as streams |
| `scene_name` | Human-readable scene name |

Validate: `len(streams) == len(camera_ids)`, ≥1 camera. The skill does **not** start MediaMTX
or `queuing-cams`; simulators must already be reachable on the SceneScape Docker network.

Superuser password: auto-generated to `<deploy_dir>/secrets/supass` during bootstrap.

## Orchestrator (steps 2–13)

From the scenescape repo (or copied skill tree), run:

```bash
SKILL_DIR=<path-to-scenescape>/.github/skills/scenescape-setup

bash "$SKILL_DIR/scripts/deploy_scenescape.sh" \
  --deploy-dir <deploy_dir> \
  --skill-dir "$SKILL_DIR" \
  --streams <rtsp_url> [<rtsp_url> ...] \
  --camera-ids <id> [<id> ...] \
  --scene-name <scene_name>
```

Options:

| Flag | Purpose |
|------|---------|
| `--phase all` | Full deploy (default) |
| `--phase bootstrap` | Steps 6–8 only |
| `--phase calibrate` | Steps 9–10 |
| `--phase scene` | Steps 11–13 |
| `--resume` | Continue from `<deploy_dir>/.deploy-state.json` (default) |
| `--fresh` | Ignore checkpoint and restart |

**Pass:** final line includes `DEPLOY COMPLETE` with `scene_uid`. **Fail:** inspect
`<deploy_dir>/deploy.log` and load the troubleshooting reference for the failed step.

### Step map

| Step | Script / action | Pass |
|------|-----------------|------|
| 6 | `bootstrap_deploy.py` | `secrets/`, `.env`, `docker-compose.yml` exist |
| 7 | `parallel_warmup.sh`, `download_detection_models.sh`, `verify_rtsp.sh`, `check_video_analytics.sh` | RTSP `PASS`, pipelines started |
| 8 | `docker compose --profile mapping up -d` | `broker`, `ntpserv`, `pgserver`, `web`, `scene`, `mapping` up |
| 9 | `capture_calibration_frames.py` | Valid JPEG per camera |
| 10 | `check_mapping_health.sh` | `model_loaded` or `status: healthy` |
| 11–12 | `reconstruct_and_finalize.py` | `Done. Scene UID: …` |
| 13 | `verify_tracking.sh` | ≥1 object on regulated topic |

Checkpoint file: `<deploy_dir>/.deploy-state.json` (`last_completed_step`, `scene_uid`,
`frames_dir`).

## Phased sub-skills

For partial runs, use the focused skills (same inputs, narrower `--phase`):

| Skill | Phase | Steps |
|-------|-------|-------|
| [scenescape-setup-bootstrap](../scenescape-setup-bootstrap/SKILL.md) | `bootstrap` | 6–8 |
| [scenescape-setup-calibrate](../scenescape-setup-calibrate/SKILL.md) | `calibrate` | 9–10 |
| [scenescape-setup-scene](../scenescape-setup-scene/SKILL.md) | `scene` | 11–13 |

## On failure

| Step | Reference |
|------|-----------|
| 7, 9 | [runtime-verification.md](./references/runtime-verification.md) |
| 11–12 | [reconstruction.md](./references/reconstruction.md) |
| 13 | [verify-tracking.md](./references/verify-tracking.md) |

Manual API details: [scene-and-cameras.md](./references/scene-and-cameras.md),
[pipeline-config.md](./references/pipeline-config.md).

## Prerequisites

- Network access to GitHub (sparse checkout of `dlstreamer-pipeline-server`)
- Corporate proxy: set `http_proxy` / `https_proxy` / `no_proxy`; RTSP Docker hostnames are
  appended via `write_deployment_env.py --append-no-proxy`
- TLS certs auto-generated in step 6
