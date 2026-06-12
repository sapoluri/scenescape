# 3D Reconstruction: Capture Frames and Finalize Scene

Scripts are in [`scripts/`](../scripts/).

## 1. Capture Calibration Frames

```bash
python scripts/capture_calibration_frames.py \
    --deploy-dir ~/scenescape-deployment \
    --cameras camera1 camera2 \
    --out-dir /tmp/frames
```

Subscribes to `scenescape/image/calibration/camera/+`, triggers `getcalibrationimage`
on each camera's command topic, and writes one JPEG per camera to `--out-dir`.
Uses TLS MQTT via the `eclipse-mosquitto:2` container on the `<project>_scenescape` Docker network.
Requires `--deploy-dir` with `secrets/certs/scenescape-ca.pem`. The default project name is
`scenescape`; pass `--project` if your Compose project name differs.

## 2. Reconstruct and Finalize

```bash
python scripts/reconstruct_and_finalize.py \
    --deploy-dir ~/scenescape-deployment \
    --frames-dir /tmp/frames \
    --cameras camera1 camera2 \
    --scene-name my_scene
```

Or finalize into an existing scene:

```bash
python scripts/reconstruct_and_finalize.py \
    --deploy-dir ~/scenescape-deployment \
    --frames-dir /tmp/frames \
    --cameras camera1 camera2 \
    --scene-uid <existing-scene-uid>
```

The script:

1. POSTs frames to the mapping service (`/v1/reconstruction`) and gets a `request_id`.
2. Creates an empty scene in the manager (if `--scene-name` is used).
3. Creates placeholder cameras for each `--cameras` ID if they do not already exist in the scene.
4. Polls `GET /scene/generate-mesh-status/<uid>/?request_id=...` until `finalized=true`.

The manager handles mesh alignment (`alignMeshToXYPlane`) and camera pose sync
(`_transformCamerasWithMeshAlignment`) automatically before persisting. Cameras must exist before
finalization; the script creates placeholder cameras so manager finalization can update them by
`camera_id`.

After finalization, verify the controller loaded the scene and subscribed to the camera topic:

```bash
cd <deploy_dir>
docker compose logs scene --tail 120 | grep -E "NEW SCENE|Subscribed to scenescape/data/camera"
```

If those log lines are missing, restart the controller and wait for it to become healthy:

```bash
docker compose restart scene
docker compose ps scene
```

## Notes

- Defaults are local host ports: `--mapping-url https://localhost:8444/v1` and
  `--manager-url https://localhost`. Override them if your deployment uses different endpoints.
- The finalize step requires a Django superuser session. The script logs in via the
  `/sign_in/` form using the password from `<deploy-dir>/secrets/supass`.
- By default the script uses `--insecure`-style local TLS behavior for generated certificates.
  Pass `--verify-tls` to verify TLS with `<deploy-dir>/secrets/certs/scenescape-ca.pem`.
- Result cache on the mapping service is ephemeral — run finalization promptly after
  queuing reconstruction.
