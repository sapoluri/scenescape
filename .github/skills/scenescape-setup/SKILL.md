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
- **Python 3.7+** with modules: `paho-mqtt`, `requests`, `json`, `re`, `base64`
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
| 3   | Generate `docker-compose.yml`                               | [docker-compose-template.md](./references/docker-compose-template.md)                            |
| 4   | Adapt pipeline-server config from canonical template        | [pipeline-config.md](./references/pipeline-config.md)                                            |
| 5   | Verify user-provided RTSP sources and pipeline integration  | [runtime-verification.md](./references/runtime-verification.md)                                  |
| 6   | Generate tracker and ReID config                            | (below — two JSON blocks)                                                                        |
| 7   | Generate broker config, secrets, and bring up containers    | [generate_secrets.sh](./references/generate_secrets.sh), [openssl.cnf](./references/openssl.cnf) |
| 8   | Verify camera MQTT data flow                                | [runtime-verification.md](./references/runtime-verification.md)                                  |
| 9   | Check mapping service health                                | (below)                                                                                          |
| 10  | Capture frames, reconstruct, then align mesh/camera poses   | [reconstruction.md](./references/reconstruction.md)                                              |
| 11  | Create scene and cameras via REST API                       | [scene-and-cameras.md](./references/scene-and-cameras.md)                                        |
| 12  | Verify object tracking                                      | [verify-tracking.md](./references/verify-tracking.md)                                            |

Load a reference file only when you reach that step.

---

## Step 1 — Gather Inputs from the User

Prompt for:

| Field        | Description                       | Example                          |
| ------------ | --------------------------------- | -------------------------------- |
| `streams`    | RTSP URL per camera               | `rtsp://192.168.1.10:554/stream` |
| `camera_ids` | Unique ID per stream (same order) | `cam1`, `cam2`                   |
| `scene_name` | Human-readable scene name         | `Warehouse Floor A`              |
| `deploy_dir` | Directory for generated files     | `./scenescape-deploy`            |

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

Alternatively, if `git` sparse checkout is unavailable, use `curl` to download individual files:

```bash
mkdir -p dlstreamer-pipeline-server/user_scripts/gvapython/sscape
curl -fsSL https://raw.githubusercontent.com/open-edge-platform/scenescape/main/dlstreamer-pipeline-server/user_scripts/gvapython/sscape/sscape_adapter.py \
  -o dlstreamer-pipeline-server/user_scripts/gvapython/sscape/sscape_adapter.py
curl -fsSL https://raw.githubusercontent.com/open-edge-platform/scenescape/main/dlstreamer-pipeline-server/queuing-config.json \
  -o dlstreamer-pipeline-server/queuing-config.json
```

After this step, `<deploy_dir>/dlstreamer-pipeline-server/` contains:

- `queuing-config.json` — canonical pipeline config template (used in Step 4)
- `user_scripts/gvapython/sscape/sscape_adapter.py` — SceneScape MQTT bridge (mounted into video-analytics)
- `mosquitto/mosquitto-secure.conf` — broker config (mounted into broker service)
- `model-proc-files/` — model processing descriptors

---

## Step 3 — Generate `docker-compose.yml`

Read [docker-compose-template.md](./references/docker-compose-template.md) now.

Copy the template and generate your docker-compose file:

```bash
# Copy template to your deployment directory
cat > <deploy_dir>/docker-compose.yml << 'EOF'
[Content from docker-compose-template.md — replace ${SECRETSDIR} with the absolute path to secrets directory]
EOF
```

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

## Step 5 — Verify User-Provided RTSP Sources and Pipeline-Server Integration

This step is a gate, not full troubleshooting. Run the quick checks below and only load the
runtime reference if a gate fails.

The user must already have each RTSP stream source running. For simulated streams, this means the
user starts MediaMTX and `queuing-cams` outside this skill and gives the skill the RTSP URLs to use.

Start only the SceneScape services needed for initial validation:

```bash
cd <deploy_dir>
docker compose up -d broker ntpserv init-models video-analytics
sleep 10
docker compose logs video-analytics --tail 100
```

Validate each user-provided RTSP URL first with ffmpeg from the SceneScape Docker network:

```bash
NET_NAME=$(docker network ls --format '{{.Name}}' | grep '_scenescape$' | head -1)
docker run --rm --network "$NET_NAME" \
  linuxserver/ffmpeg:version-8.1-cli \
  -nostdin -v error -rtsp_transport tcp \
  -i '<rtsp_url>' \
  -t 5 -f null -
echo "EXIT:$?"
```

Pass criteria:

- ffmpeg exits with `EXIT:0`
- `video-analytics` logs show pipelines initialized/started
- No persistent RTSP connection failures

If any gate fails, load [runtime-verification.md](./references/runtime-verification.md)
and follow the Step 5 troubleshooting flow.

**Do not proceed to Step 6 until Step 5 passes.**

---

## Step 6 — Tracker and ReID Config

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

## Step 7 — Generate Secrets and Bring Up Containers

Read and execute [generate_secrets.sh](./references/generate_secrets.sh) using the
[openssl.cnf](./references/openssl.cnf) template. Then create `.env` and start all services:

```bash
cd <deploy_dir>
bash generate_secrets.sh

# Build .env — read DATABASE_PASSWORD from generated secrets.py
SECRETSDIR=$(pwd)/secrets
DATABASE_PASSWORD=$(python3 -c "
import re
txt = open('secrets/django/secrets.py').read()
print(re.search(r\"DATABASE_PASSWORD='([^']+)'\", txt).group(1))
")
SUPASS=$(cat secrets/supass)
# If any user-provided RTSP URL uses an internal hostname, append that hostname to no_proxy.
# Example: no_proxy="${no_proxy},mediaserver"
cat > .env <<EOF
SECRETSDIR=${SECRETSDIR}
DATABASE_PASSWORD=${DATABASE_PASSWORD}
SUPASS=${SUPASS}
http_proxy=${http_proxy}
https_proxy=${https_proxy}
no_proxy=${no_proxy}
EOF

docker compose up -d
```

Wait for all containers to be healthy:

```bash
docker compose ps
```

Expected healthy: `broker`, `ntpserv`, `pgserver`, `web`, `scene`.

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

---

## Step 8 — Verify Calibration Frame Gate (Before Reconstruction)

This is the required gate before mapping reconstruction. Do not block on object detections here.

After all containers are healthy, wait up to **2 minutes** for video-analytics to initialize pipelines.
Then verify command/response flow for calibration images:

- publish `getcalibrationimage` to `scenescape/cmd/camera/<camera_id>`
- receive one message on `scenescape/image/calibration/camera/<camera_id>`
- confirm payload has `image` and decodes to a valid JPEG frame

Use MQTT flags that match your broker listener mode:

- If listener `1883` is plaintext: no TLS flags
- If listener `1883` is TLS: use `--cafile` (and `--insecure` when connecting via `localhost`)

Pass criteria:

- Response topic is `scenescape/image/calibration/camera/<camera_id>`.
- JSON payload contains keys `id`, `timestamp`, and `image`.
- Base64-decoded `image` is a valid JPEG (`FFD8FF ... FFD9`).

If this step fails or times out, load [runtime-verification.md](./references/runtime-verification.md)
and follow the Step 8 troubleshooting flow.

---

## Step 9 — Check Mapping Service Health

Poll `GET https://mapping.scenescape.intel.com:8444/v1/health` every 10 s for up to 3 minutes.

```bash
curl -sk https://mapping.scenescape.intel.com:8444/v1/health
```

Expected response: `{"status": "healthy"}` or `{"model_loaded": true}`.

**Note**: The model installer downloads `person-detection-retail-0013` on first run — allow extra
time on a fresh deployment. The service may take 2–3 minutes to load the model.

If health check fails:

```bash
docker compose logs mapping --tail 50
```

Look for model download errors or GPU/memory issues.
