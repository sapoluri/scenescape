# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Testing Phase 2: Recording & Replay

End-to-end guide for verifying the recorder service and Replay mode.

## Prerequisites

- Manager running (Django + frontend built with `make -C manager ui-build`)
- MQTT broker reachable (same one the manager uses)
- dlstreamer pipeline publishing to `scenescape/regulated/scene/+` and
  `scenescape/image/camera/+`
- A scene with at least one camera, and live tracks flowing

## 1. Start the recorder

### Docker (recommended)

```bash
cd manager/recorder
docker build -t scenescape-recorder .
docker run -d --name recorder \
  -e RECORDER_MQTT_BROKER=<mqtt-host> \
  -e RECORDER_MQTT_PORT=1883 \
  -e RECORDER_MANAGER_API=http://<manager-host>:8000/api/v1 \
  -e RECORDER_MANAGER_TOKEN=<api-token> \
  -e RECORDER_RETENTION_DAYS=1 \
  -v recordings:/data/recordings \
  scenescape-recorder
```

### Local Python

```bash
cd manager/recorder
pip install -r requirements.txt
RECORDER_MQTT_BROKER=<mqtt-host> python recorder.py
```

### Verify it's running

```bash
docker logs recorder
# Expect: "Recorder starting: broker=... storage=/data/recordings ..."
# Expect: "Connected to MQTT ..."
# Expect: "Recording <scene-id> -> /data/recordings/<scene-id>/2026-....rrd"
#   (appears once per scene when the first track/frame arrives)
```

## 2. Verify .rrd files are written

```bash
# Inside the container or on the shared volume:
ls -lh /data/recordings/<scene-id>/
# Expect: YYYY-MM-DD-HH.rrd files, growing as tracks/frames arrive.
```

Each file is one hour of one scene. The recorder rotates to a new file
on the hour and deletes files older than `RECORDER_RETENTION_DAYS`.

## 3. Verify the recordings API

```bash
curl -H "Authorization: Token <api-token>" \
  "http://<manager-host>:8000/api/v1/recordings/?scene=<scene-uuid>"
```

Expect:

```json
{"recordings": [
  {"id": "<scene-uuid>/2026-10-06-00.rrd",
   "scene": "<scene-uuid>",
   "start": 1728172800000, "end": 1728176400000,
   "size": 123456, "provider": "rerun",
   "url": "/api/v1/recordings/<scene-uuid>/2026-10-06-00.rrd"}
]}
```

## 4. Test Replay mode in the UI

1. Open a scene (`/<scene-uuid>/`).
2. In the viewport header, click **Replay** (next to Live in the Mode section).
3. The recording picker lists the scene's `.rrd` files with time ranges and sizes.
4. Click a recording → the Rerun web viewer loads it.
5. Scrub the timeline, play/pause. Tracks appear as 3D points; camera
   frames appear as images — synchronized on `log_time`.
6. Click **Live** (or the back button) to return to the live viewport.

### What to check

- [ ] Toggle switches between Live and Replay without errors
- [ ] Picker shows recordings with correct time ranges
- [ ] Viewer loads the `.rrd` (no CORS errors — the manager serves
      `Access-Control-Allow-Origin: *` on the download endpoint)
- [ ] Timeline scrub moves tracks and frames together
- [ ] Returning to Live resumes the MQTT-fed viewport

## 5. Troubleshooting

| Symptom | Check |
|---------|-------|
| No `.rrd` files appear | MQTT broker reachable? Topics publishing? Check `docker logs recorder` for "Connected to MQTT" |
| Picker shows "No recordings yet" | API reachable? Auth token valid? `curl` the endpoint directly |
| Viewer shows blank / CORS error | Download endpoint must return `Access-Control-Allow-Origin`; check browser network tab for the `.rrd` fetch |
| Frames missing in replay | Camera→scene map: recorder logs "Camera map refreshed" on startup; verify `RECORDER_MANAGER_API` and token |
| `.rrd` won't open in viewer | Version lock: recorder writes with `rerun-sdk` 0.38.1; viewer is `@rerun-io/web-viewer-react` 0.38.1. Mismatched versions cannot read each other's files. |

## 6. Stopping

```bash
docker stop recorder
# .rrd files remain on the volume; retention cleanup runs every 60s while
# the recorder is alive.
```
