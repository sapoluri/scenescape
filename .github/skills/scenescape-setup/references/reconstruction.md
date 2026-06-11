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
Reads broker credentials from `<deploy-dir>/secrets/browser.auth`.

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
3. Polls `POST /scene/generate-mesh-status/<uid>/?request_id=...` until `finalized=true`.

The manager handles mesh alignment (`alignMeshToXYPlane`) and camera pose sync
(`_transformCamerasWithMeshAlignment`) automatically before persisting.

## Notes

- Override `--mapping-url` and `--manager-url` if the defaults
  (`https://mapping.scenescape.intel.com:8444/v1` and `https://web.scenescape.intel.com`)
  don't match your deployment.
- The finalize step requires a Django superuser session. The script logs in via the
  `/sign-in` form using the password from `<deploy-dir>/secrets/supass`.
- Result cache on the mapping service is ephemeral — run finalization promptly after
  queuing reconstruction.
