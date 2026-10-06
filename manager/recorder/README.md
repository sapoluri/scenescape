# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Scenescape Recorder (Phase 2.0)

Always-on service that subscribes to scene MQTT topics and writes
time-partitioned `.rrd` files via `rerun-sdk` (one file per scene per hour)
to a local disk volume.

## Topics

- `scenescape/regulated/scene/+` — telemetry / object tracks (from the
  dlstreamer pipeline; same topics the manager UI uses)
- `scenescape/image/camera/+` — camera frames (JPEG bytes)

## Entity mapping (Phase 2.1)

- Application ID = scene id; timeline = `log_time` (wall clock)
- `scene/<id>/objects/<track_id>` → `Points3D` (per-frame positions)
- `scene/<id>/cameras/<cam_id>/image` → `EncodedImage` (frames)

## Configuration (environment)

| Var | Default | Description |
|-----|---------|-------------|
| `RECORDER_MQTT_BROKER` | `localhost` | MQTT broker host |
| `RECORDER_MQTT_PORT` | `1883` | MQTT broker port |
| `RECORDER_MQTT_CAFILE` | _(empty)_ | PEM CA file for MQTT TLS (required on the demo broker) |
| `RECORDER_STORAGE_DIR` | `/data/recordings` | Local volume for `.rrd` files |
| `RECORDER_RETENTION_DAYS` | `1` | Delete files older than this |
| `RECORDER_SCENES` | _(empty = all)_ | Comma-separated scene IDs to record |
| `RECORDER_MANAGER_API` | `http://localhost:8000/api/v1` | Manager REST for camera→scene map |
| `RECORDER_MANAGER_TOKEN` | _(empty)_ | API token for the manager |

## Running

```bash
pip install -r requirements.txt
python recorder.py
```

Docker:

```bash
docker build -t scenescape-recorder .
docker run -d \
  -e RECORDER_MQTT_BROKER=mqtt \
  -v recordings:/data/recordings \
  scenescape-recorder
```

The manager lists recordings via `GET /api/v1/recordings/?scene=<id>`
and serves `.rrd` files for the Rerun web viewer.

## Testing

See `docs/design/phase2-testing.md` for the end-to-end testing guide
(starting the recorder, verifying `.rrd` files, using Replay mode).
