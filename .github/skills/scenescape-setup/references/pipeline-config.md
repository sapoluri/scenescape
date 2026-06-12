# DLStreamer Pipeline Config

The canonical template is `queuing-config.json` (downloaded in bootstrap). **Do not generate
from scratch.**

## Procedure

Bootstrap runs this automatically via `scripts/adapt_pipeline_config.py`. To re-run manually:

```bash
python3 <skill-dir>/scripts/adapt_pipeline_config.py \
  --deploy-dir <deploy_dir> \
  --camera-ids camera1 camera2 \
  --streams rtsp://mediaserver:8554/queuing-cam1 rtsp://mediaserver:8554/queuing-cam2
```

Writes `<deploy_dir>/pipeline-config.json`. RTSP Docker hostnames are printed as
`no_proxy_hosts=…` for `write_deployment_env.py --append-no-proxy`.

## Substitutions performed

| Template placeholder | Replaced with |
| -------------------- | ------------- |
| `"name": "qcam1"` / `"qcam2"` | camera ID |
| `rtsp://mediaserver:8554/queuing-cam*` | user RTSP URL |
| `"cameraid": "atag-qcam*"` | camera ID |
| `models/object_detection/.../person-detection-retail-0013.json` | `model-proc-files/person-detection-retail-0013.json` |

Keep `add-reference-timestamp-meta=true` on `rtspsrc`. Keep `sscape_adapter.py` paths unchanged.

## Notes

- Model: `person-detection-retail-0013` (downloaded by `download_detection_models.sh`)
- `ntpServer: ntpserv` matches `docker-compose.yml`
- External RTSP simulators must be reachable from the SceneScape Docker network
