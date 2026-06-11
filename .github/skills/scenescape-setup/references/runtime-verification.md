<!--
SPDX-FileCopyrightText: (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Runtime Verification and Troubleshooting

Load this file only when Step 5 or Step 8 in SKILL.md fails.

## Step 5 — Pipeline and User-Provided RTSP Verification

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

```bash
NET_NAME=$(docker network ls --format '{{.Name}}' | grep '_scenescape$' | head -1)
docker run --rm --network "$NET_NAME" \
  linuxserver/ffmpeg:version-8.1-cli \
  -nostdin -v error -rtsp_transport tcp \
  -i '<rtsp_url>' \
  -t 5 -f null -
echo "EXIT:$?"
```

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

### 4) Additional checks when startup still fails

```bash
docker compose exec video-analytics ls /home/pipeline-server/models/person-detection-retail-0013/
docker compose exec ntpserv chronyc tracking
docker compose config | grep -A 8 "video-analytics:"
```

## Step 8 — Calibration Image Gate Before Reconstruction

### 1) Determine listener mode on broker

```bash
sed -n '1,120p' <deploy_dir>/dlstreamer-pipeline-server/mosquitto/mosquitto-secure.conf
```

Use listener mode for port `1883`. The setup skill normalizes the generated broker config so
`1883` is plaintext and `1884` is TLS websockets:

- plaintext listener: no TLS flags
- TLS listener: use `--cafile` (and `--insecure` when host is `localhost`)

### 2) Subscribe to calibration image topic, then send command

```bash
# plaintext 1883 example
docker run --rm --network <project>_scenescape eclipse-mosquitto:2 \
  mosquitto_sub -h broker.scenescape.intel.com -p 1883 \
  -t 'scenescape/image/calibration/camera/<camera_id>' -C 1

docker run --rm --network <project>_scenescape eclipse-mosquitto:2 \
  mosquitto_pub -h broker.scenescape.intel.com -p 1883 \
  -t 'scenescape/cmd/camera/<camera_id>' -m 'getcalibrationimage'
```

Verify message contains:

- topic: `scenescape/image/calibration/camera/<camera_id>`
- JSON payload keys: `id`, `timestamp`, `image`
- `image` decodes from base64 to valid JPEG bytes (`FFD8FF ... FFD9`)

### 3) Correlate broker and DLSPS logs when no frame returns

```bash
docker compose logs --tail 120 broker
docker compose logs --tail 200 video-analytics
```

Look for:

- DLSPS connected/subscribed to `scenescape/cmd/camera/<camera_id>`
- command traffic arriving at broker
- publish activity on `scenescape/image/calibration/camera/<camera_id>`
