---
name: scenescape-setup
description: >
  Use when you need to deploy a working Intel® SceneScape installation from scratch, outside the
  scenescape repo. Handles everything end-to-end: prompting for camera streams, generating
  docker-compose, DLStreamer pipeline config, tracker config, bringing up containers, verifying
  MQTT data flow, running 3D mapping, creating the scene and cameras via REST API, and confirming
  object tracking is live on the regulated topic.
argument-hint: "Optional: path to a directory where deployment files should be created (default: current directory)"
---

# SceneScape End-to-End Setup

Deploys a complete SceneScape installation from a clean directory. Only Docker and Python
required on the host — no SceneScape source checkout needed.

---

## Prerequisites

Before starting, ensure you have:

- **Docker** and **docker-compose** installed
- **Python 3.10+** with `requests` installed. Calibration-frame helper scripts use the
  `eclipse-mosquitto:2` container for MQTT and do not require `paho-mqtt` on the host.
- **Network access** to GitHub (for sparse checkout of dlstreamer-pipeline-server)
- **Proxy configuration** (if behind corporate proxy):
  - Set environment variables: `http_proxy`, `https_proxy`, `no_proxy`
  - The deployment will automatically append `.scenescape.intel.com` to `no_proxy` in container configs
- **Writable directory** for deployment files (e.g., `/opt/scenescape`, `./scenescape-deploy`)
- **TLS certificates** will be auto-generated during Step 7

---

## Procedure Overview

Execute these steps in order. Each step links to a reference file with the exact content
to generate or commands to run.

| #   | Step                                                        | Reference                                                                                        |
| --- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 1   | Gather inputs from user                                     | (below)                                                                                          |
| 2   | Create deployment directory                                 | (below)                                                                                          |
| 2a  | Download `dlstreamer-pipeline-server/` from scenescape repo | (below)                                                                                          |
| 2b  | Copy setup helper scripts into deployment directory         | (below)                                                                                          |
| 3   | Generate `docker-compose.yml`                               | [docker-compose-template.md](./references/docker-compose-template.md)                            |
| 4   | Adapt pipeline-server config from canonical template        | [pipeline-config.md](./references/pipeline-config.md)                                            |
| 5   | Generate tracker and ReID config                            | (below — two JSON blocks)                                                                        |
| 6   | Generate broker config, secrets, and `.env`                 | [generate_secrets.sh](./references/generate_secrets.sh), [openssl.cnf](./references/openssl.cnf) |
| 7   | Verify user-provided RTSP sources and pipeline integration  | [runtime-verification.md](./references/runtime-verification.md)                                  |
| 8   | Bring up containers and download AI models                  | (below)                                                                                          |
| 9   | Verify camera MQTT data flow                                | [runtime-verification.md](./references/runtime-verification.md)                                  |
| 10  | Check mapping service health                                | (below)                                                                                          |
| 11  | Capture frames, reconstruct, then align mesh/camera poses   | [reconstruction.md](./references/reconstruction.md)                                              |
| 12  | Create scene and cameras via REST API                       | [scene-and-cameras.md](./references/scene-and-cameras.md)                                        |
| 13  | Verify object tracking                                      | [verify-tracking.md](./references/verify-tracking.md)                                            |

Load a reference file only when you reach that step.

---

## Step 1 — Gather Inputs from the User

Prompt for required inputs:

- `streams`: RTSP URL per camera
- `camera_ids`: unique camera IDs in the same order as `streams`
- `scene_name`: human-readable scene name
- `deploy_dir`: output directory for generated deployment files

Validate: `len(streams) == len(camera_ids)`, IDs are unique and contain no `/`, at least 1 camera.
The skill does **not** start or manage MediaMTX, `queuing-cams`, or any other RTSP simulator. If
the user wants simulated streams, they must start those containers separately and provide RTSP URLs
that are reachable from the SceneScape Docker network.

The superuser password is **generated automatically** in Step 6 and written to
`<deploy_dir>/secrets/supass`.

---

## Step 2 — Create Deployment Directory

```bash
mkdir -p <deploy_dir>
```

All generated files go under `<deploy_dir>/`.

---

## Step 2a — Download `dlstreamer-pipeline-server/` from the SceneScape Repo

The canonical pipeline configs, `sscape_adapter.py` user scripts, and mosquitto config all live in the
SceneScape repository. Download the `dlstreamer-pipeline-server/` directory into `<deploy_dir>/` using
a sparse checkout (no full clone needed):

```bash
cd <deploy_dir>
git clone --filter=blob:none --sparse \
  https://github.com/open-edge-platform/scenescape.git _scenescape-tmp
cd _scenescape-tmp
git sparse-checkout set dlstreamer-pipeline-server
cp -r dlstreamer-pipeline-server ../
cd .. && rm -rf _scenescape-tmp
```

Normalize the broker config so MQTT port `1883` is plaintext for DLStreamer and controller MQTT
clients, while websocket port `1884` keeps TLS:

```bash
python3 <path-to-scenescape-repo>/.github/skills/scenescape-setup/scripts/normalize_broker_config.py
```

Alternatively, if `git` sparse checkout is unavailable, use `curl` to download individual files:

```bash
mkdir -p dlstreamer-pipeline-server/user_scripts/gvapython/sscape
curl -fsSL https://raw.githubusercontent.com/open-edge-platform/scenescape/main/dlstreamer-pipeline-server/user_scripts/gvapython/sscape/sscape_adapter.py \
  -o dlstreamer-pipeline-server/user_scripts/gvapython/sscape/sscape_adapter.py
curl -fsSL https://raw.githubusercontent.com/open-edge-platform/scenescape/main/dlstreamer-pipeline-server/queuing-config.json \
  -o dlstreamer-pipeline-server/queuing-config.json
```

If using the fallback download, also download
`dlstreamer-pipeline-server/mosquitto/mosquitto-secure.conf` and run the same broker config
normalization command above.

After this step, `<deploy_dir>/dlstreamer-pipeline-server/` contains:

- `queuing-config.json` — canonical pipeline config template (used in Step 4)
- `user_scripts/gvapython/sscape/sscape_adapter.py` — SceneScape MQTT bridge (mounted into video-analytics)
- `mosquitto/mosquitto-secure.conf` — broker config (mounted into broker service)
- `model-proc-files/` — model processing descriptors

---

## Step 2b — Copy Setup Helper Scripts

Copy the helper scripts from this skill into the deployment directory so later steps can run with
`python scripts/...`:

```bash
cd <deploy_dir>
mkdir -p scripts
cp <path-to-scenescape-repo>/.github/skills/scenescape-setup/scripts/*.py scripts/
```

---

## Step 3 — Generate `docker-compose.yml`

Read [docker-compose-template.md](./references/docker-compose-template.md) now.

Copy the template and generate your docker-compose file:

````bash
DEPLOY_DIR=<deploy_dir>
SKILL_DIR=<path-to-scenescape-repo>/.github/skills/scenescape-setup
awk '/^```yaml$/ {flag=1; next} /^```$/ && flag {exit} flag {print}' \
  "$SKILL_DIR/references/docker-compose-template.md" \
  | sed "s|\${SECRETSDIR}|$DEPLOY_DIR/secrets|g" \
  > "$DEPLOY_DIR/docker-compose.yml"
````

Or, if you prefer to edit manually:

1. Get the docker-compose template from [docker-compose-template.md](./references/docker-compose-template.md)
2. Replace `${SECRETSDIR}` with the **absolute path** to `<deploy_dir>/secrets` (will be created in Step 7)
3. Save as `<deploy_dir>/docker-compose.yml`

**Key points:**

- All services have `no_proxy` set to include `.scenescape.intel.com` for proper service alias resolution
- RTSP streams are user-provided inputs. The compose file does not define MediaMTX or `queuing-cams`.
- All inter-service communication uses DNS aliases within the `scenescape` Docker network
- External healthchecks use `localhost` (within containers); external API calls use service aliases

**Do not modify the docker-compose file further** — it is template-driven and includes all required services.

---

## Step 4 — Adapt Pipeline-Server Config from Canonical Template

Read [pipeline-config.md](./references/pipeline-config.md) for detailed instructions.

The canonical pipeline template is `queuing-config.json` (downloaded in Step 2a). Adapt it by:

1. Copy the template: `cp <deploy_dir>/dlstreamer-pipeline-server/queuing-config.json <deploy_dir>/pipeline-config.json`
2. Edit `pipeline-config.json`:
   - Add or remove pipeline entries to match your camera count
   - For each camera, replace placeholders:
     - `"name": "qcam1"` → `"name": "<camera_id>"`
     - `rtsp://mediaserver:8554/...` → your actual RTSP URL
     - `"cameraid": "..."` → `"cameraid": "<camera_id>"`
   - Keep `add-reference-timestamp-meta=true` on `rtspsrc` (required for NTP timestamps)
   - Keep sscape_adapter.py paths unchanged (container-internal)
   - Use the exact user-provided RTSP URL for each camera. Do not replace it with a MediaMTX or
     `queuing-cams` URL unless the user explicitly provided that simulator URL.

**Detection Model**: `person-detection-retail-0013` (FP32, person detection for retail scenarios).
This model is downloaded automatically by the `model_downloader` service.

---

## Step 5 — Tracker and ReID Config

Write `<deploy_dir>/tracker-config.json`:

```json
{
  "max_unreliable_time_s": 1.0,
  "non_measurement_time_dynamic_s": 0.8,
  "non_measurement_time_static_s": 1.6,
  "time_chunking_enabled": true,
  "time_chunking_rate_fps": 30,
  "suspended_track_timeout_secs": 60.0
}
```

Write `<deploy_dir>/reid-config.json`:

```json
{
  "similarity_metric": "COSINE",
  "stale_feature_timeout_secs": 5.0,
  "stale_feature_check_interval_secs": 1.0,
  "feature_accumulation_threshold": 12,
  "minimum_bbox_area": 5000,
  "feature_slice_size": 10,
  "similarity_threshold": 0.5
}
```

---

## Step 6 — Generate Secrets and `.env`

Objective: generate secrets and deployment environment file.

Command:

```bash
cd <deploy_dir>
bash generate_secrets.sh
python3 <path-to-scenescape-repo>/.github/skills/scenescape-setup/scripts/write_deployment_env.py \
  --deploy-dir <deploy_dir>
```

If any user-provided RTSP URL uses an internal hostname, append that hostname to `no_proxy`
in `<deploy_dir>/.env` (example: `no_proxy=${no_proxy},mediaserver`).

Pass: `<deploy_dir>/secrets/` and `<deploy_dir>/.env` exist; `.env` contains
`SECRETSDIR`, `DATABASE_PASSWORD`, and `SUPASS`.

On fail: reload [generate_secrets.sh](./references/generate_secrets.sh) and
[openssl.cnf](./references/openssl.cnf), then rerun.

---

## Step 7 — Verify User-Provided RTSP Sources and Pipeline-Server Integration

Objective: confirm RTSP reachability and initial pipeline startup before full bring-up.

Command:

Start only the SceneScape services needed for initial validation:

```bash
cd <deploy_dir>
docker compose up -d broker ntpserv init-models video-analytics
sleep 10
docker compose logs video-analytics --tail 100
```

Validate each user-provided RTSP URL first with ffmpeg from the SceneScape Docker network:

Use the RTSP gate template in [command-templates.md](./references/command-templates.md).

Pass:

- ffmpeg exits with `EXIT:0`
- `video-analytics` logs show pipelines initialized/started
- No persistent RTSP connection failures

On fail: load [runtime-verification.md](./references/runtime-verification.md) and follow
Step 7 troubleshooting.

**Do not proceed to Step 8 until Step 7 passes.**

---

## Step 8 — Bring Up Containers

Objective: bring up full stack and ensure model availability.

Command:

```bash
cd <deploy_dir>
docker compose --profile mapping up -d
```

Wait for all containers to be healthy:

```bash
docker compose --profile mapping ps
```

Pass: `broker`, `ntpserv`, `pgserver`, `web`, `scene`, and `mapping` are healthy.

### Download AI models

The `model_downloader` service (`scenescape-model-installer:latest`) exits immediately without
downloading models by itself. Run the download manually using the openvino omz_downloader:

```bash
docker run --rm --user root \
  -e http_proxy="${http_proxy}" \
  -e https_proxy="${https_proxy}" \
  -v <project_name>_vol-models:/models \
  scenescape-model-installer:latest bash -c "
pip3 install --break-system-packages openvino-dev 2>&1 | grep Successfully
/usr/local/bin/omz_downloader --name person-detection-retail-0013 -o /models/
chmod -R a+rX /models/
"
```

Replace `<project_name>` with the Docker Compose project name (default: `scenescape` from the
`name:` field in `docker-compose.yml`, so the volume is `scenescape_vol-models`).

After models download, restart `video-analytics` so it picks them up:

```bash
docker compose restart video-analytics
```

On fail: inspect `docker compose logs --tail 100 <service>` for the failing service and
resolve before continuing.

---

## Step 9 — Verify Calibration Frame Gate (Before Reconstruction)

Objective: verify calibration command/response flow before reconstruction.

Command:

After all containers are healthy, wait up to **2 minutes** for pipeline initialization, then:

- publish `getcalibrationimage` to `scenescape/cmd/camera/<camera_id>`
- receive one message on `scenescape/image/calibration/camera/<camera_id>`
- confirm payload has `image` and decodes to a valid JPEG frame

Use the MQTT publish/subscribe templates in
[command-templates.md](./references/command-templates.md), selecting plaintext or TLS mode to
match listener `1883`.

Pass:

- Response topic is `scenescape/image/calibration/camera/<camera_id>`.
- JSON payload contains keys `id`, `timestamp`, and `image`.
- Base64-decoded `image` is a valid JPEG (`FFD8FF ... FFD9`).

On fail: load [runtime-verification.md](./references/runtime-verification.md) and follow
Step 9 troubleshooting.

---

## Step 10 — Check Mapping Service Health

Objective: confirm mapping service readiness.

Command: poll `GET https://mapping.scenescape.intel.com:8444/v1/health` every 10 s for up to
3 minutes.

```bash
curl -sk https://mapping.scenescape.intel.com:8444/v1/health
```

Pass: response contains `{"status": "healthy"}` or `{"model_loaded": true}`.

**Note**: The model installer downloads `person-detection-retail-0013` on first run — allow extra
time on a fresh deployment. The service may take 2–3 minutes to load the model.

On fail:

```bash
docker compose logs mapping --tail 50
```

Look for model download errors or GPU/memory issues.
