<!--
SPDX-FileCopyrightText: (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Runtime Verification and Troubleshooting

Load this file only when Step 7 or Step 9 in SKILL.md fails.

## Step 5 — Pipeline and User-Provided RTSP Verification

Shared runtime commands are in [command-templates.md](./command-templates.md).

### 1) Ensure services are running

```bash
cd <deploy_dir>
docker compose up -d broker ntpserv init-models video-analytics
docker compose ps
```

The RTSP stream source is external to this skill. If the user is simulating streams with MediaMTX
and `queuing-cams`, those containers must already be running and reachable from the SceneScape
Docker network before this verification step.

### 2) Verify RTSP first using separate ffmpeg container

Use the RTSP gate command template from [command-templates.md](./command-templates.md).

- `EXIT:0` means cross-container RTSP is working.
- Decoder warnings at stream start can be acceptable when exit code is 0.
- If RTSP fails here, DLSPS will not publish detections. Fix the user-provided RTSP source,
  Docker network reachability, or pipeline URL first.

### 3) Check pipeline startup

```bash
docker compose logs --tail 200 video-analytics
```

Verify:

- pipelines initialize and enter running state
- no persistent RTSP connection failures
- no fatal model-load errors
- MQTT connects over TLS (look for `Connected to MQTT Broker` without connection errors)

### 4) Additional checks when startup still fails

```bash
docker compose exec video-analytics ls /home/pipeline-server/models/person-detection-retail-0013/
docker compose exec video-analytics ls /run/secrets/certs/scenescape-ca.pem
docker compose exec ntpserv chronyc tracking
docker compose config | grep -A 12 "video-analytics:"
```

## Step 8 — Calibration Image Gate Before Reconstruction

### 1) Confirm broker TLS listener

```bash
sed -n '1,120p' <deploy_dir>/dlstreamer-pipeline-server/mosquitto/mosquitto-secure.conf
```

Listener `1883` must include `keyfile`, `certfile`, and `tls_version`. All MQTT clients use
TLS with the SceneScape CA (see [command-templates.md](./command-templates.md)).

### 2) Subscribe to calibration image topic, then send command

Use the MQTT subscribe/publish templates from [command-templates.md](./command-templates.md):

- subscribe topic: `scenescape/image/calibration/camera/<camera_id>`
- publish topic: `scenescape/cmd/camera/<camera_id>` with payload `getcalibrationimage`

Verify message contains:

- topic: `scenescape/image/calibration/camera/<camera_id>`
- JSON payload keys: `id`, `timestamp`, `image`
- `image` decodes from base64 to valid JPEG bytes (`FFD8FF ... FFD9`)

### 3) Correlate broker and DLSPS logs when no frame returns

```bash
docker compose logs --tail 120 broker
docker compose logs --tail 120 video-analytics
```

Look for:

- command traffic arriving at broker
- TLS handshake or certificate errors
- video-analytics publishing calibration images after `getcalibrationimage`
