# DLStreamer Pipeline Config

The canonical pipeline config template is `queuing-config.json` downloaded in Step 2a from
`https://github.com/open-edge-platform/scenescape/blob/main/dlstreamer-pipeline-server/queuing-config.json`.

**Do not generate this file from scratch.** Instead, adapt the downloaded file for the user's cameras.
The RTSP source for each pipeline is the URL provided by the user. The skill does not start or
manage MediaMTX, `queuing-cams`, or any other RTSP simulator.

## Procedure

1. Copy the canonical template as the starting point:

```bash
cp <deploy_dir>/dlstreamer-pipeline-server/queuing-config.json <deploy_dir>/pipeline-config.json
```

2. Edit `<deploy_dir>/pipeline-config.json` with a JSON-aware tool. For example, set the camera IDs
   and streams as JSON arrays in matching order, then run:

```bash
export DEPLOY_DIR=<deploy_dir>
export CAMERA_IDS_JSON='["camera1"]'
export STREAMS_JSON='["rtsp://mediaserver:8554/queuing-cam1"]'
python3 - <<'PY'
import copy
import json
import os
from pathlib import Path

deploy_dir = Path(os.environ["DEPLOY_DIR"])
camera_ids = json.loads(os.environ["CAMERA_IDS_JSON"])
streams = json.loads(os.environ["STREAMS_JSON"])

if len(camera_ids) != len(streams):
  raise SystemExit("CAMERA_IDS_JSON and STREAMS_JSON must have the same length")
if not camera_ids or len(set(camera_ids)) != len(camera_ids) or any("/" in camera_id for camera_id in camera_ids):
  raise SystemExit("camera IDs must be non-empty, unique, and contain no slash")

path = deploy_dir / "pipeline-config.json"
config = json.loads(path.read_text())
templates = config["config"]["pipelines"]
if not templates:
  raise SystemExit("pipeline template contains no pipelines")

pipelines = []
for index, (camera_id, stream) in enumerate(zip(camera_ids, streams)):
  template = templates[min(index, len(templates) - 1)]
  entry = copy.deepcopy(template)
  entry["name"] = camera_id
  entry["pipeline"] = entry["pipeline"].replace(
    "rtsp://mediaserver:8554/queuing-cam1", stream
  ).replace(
    "rtsp://mediaserver:8554/queuing-cam2", stream
  ).replace(
    "/home/pipeline-server/models/object_detection/person/person-detection-retail-0013.json",
    "/home/pipeline-server/model-proc-files/person-detection-retail-0013.json"
  )
  entry["payload"]["parameters"]["camera_config"]["cameraid"] = camera_id
  pipelines.append(entry)

config["config"]["pipelines"] = pipelines
path.write_text(json.dumps(config, indent=2) + "\n")
PY
```

This does the following:

- The template has two entries (`qcam1`, `qcam2`). **Add or remove entries to match the user's
  camera count.**
- For each camera entry, substitute:

  | Placeholder in template                     | Replace with                |
  | ------------------------------------------- | --------------------------- |
  | `"name": "qcam1"` / `"name": "qcam2"`       | `"name": "<camera_id>"`     |
  | `rtsp://mediaserver:8554/queuing-cam1`      | `<rtsp_url>`                |
  | `rtsp://mediaserver:8554/queuing-cam2`      | `<rtsp_url>`                |
  | `"cameraid": "atag-qcam1"` / `"atag-qcam2"` | `"cameraid": "<camera_id>"` |

- Keep `add-reference-timestamp-meta=true` on `rtspsrc` — required for NTP timestamp extraction.
- Keep all `sscape_adapter.py` module paths unchanged — they are container-internal paths.
- Rewrite the `person-detection-retail-0013.json` model-proc path to
  `/home/pipeline-server/model-proc-files/person-detection-retail-0013.json`, which matches the
  compose mount.
- If the user is simulating streams with MediaMTX/`queuing-cams`, use the RTSP URLs that the user
  provides for that simulator, and verify those URLs from the SceneScape Docker network.

## Notes

- Detection model: `person-detection-retail-0013` (FP32). Downloaded automatically by the
  `model_downloader` service into the shared `vol-models` volume.
- The `ntpServer` value `ntpserv` matches the NTP service hostname in `docker-compose.yml`.
- Each pipeline entry must have a unique `"name"` — use the camera ID.
- The `user_scripts/` directory containing `sscape_adapter.py` must be volume-mounted into the
  `video-analytics` container (see `docker-compose.yml`).
- If an RTSP URL uses a Docker hostname such as `mediaserver`, that hostname must be resolvable from
  the SceneScape Docker network. The skill will not create that service.
