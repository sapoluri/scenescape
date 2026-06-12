# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Adapt queuing-config.json for the user's cameras and RTSP streams."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from urllib.parse import urlparse

MODEL_PROC_FROM = "/home/pipeline-server/models/object_detection/person/person-detection-retail-0013.json"
MODEL_PROC_TO = "/home/pipeline-server/model-proc-files/person-detection-retail-0013.json"
TEMPLATE_STREAMS = (
  "rtsp://mediaserver:8554/queuing-cam1",
  "rtsp://mediaserver:8554/queuing-cam2",
)


def validate_camera_ids(camera_ids: list[str]) -> None:
  if not camera_ids or len(set(camera_ids)) != len(camera_ids):
    raise ValueError("camera IDs must be non-empty and unique")
  if any("/" in camera_id for camera_id in camera_ids):
    raise ValueError("camera IDs must not contain '/'")


def rtsp_no_proxy_hosts(streams: list[str]) -> list[str]:
  hosts: list[str] = []
  seen: set[str] = set()
  for stream in streams:
    host = urlparse(stream).hostname
    if host and host not in seen:
      seen.add(host)
      hosts.append(host)
  return hosts


def adapt_pipeline_config(
  deploy_dir: Path,
  camera_ids: list[str],
  streams: list[str],
) -> list[str]:
  if len(camera_ids) != len(streams):
    raise ValueError("camera_ids and streams must have the same length")

  validate_camera_ids(camera_ids)

  template_path = deploy_dir / "dlstreamer-pipeline-server" / "queuing-config.json"
  if not template_path.is_file():
    raise FileNotFoundError(f"Missing pipeline template: {template_path}")

  config_path = deploy_dir / "pipeline-config.json"
  config = json.loads(template_path.read_text(encoding="utf-8"))
  templates = config["config"]["pipelines"]
  if not templates:
    raise ValueError("pipeline template contains no pipelines")

  pipelines = []
  for index, (camera_id, stream) in enumerate(zip(camera_ids, streams)):
    template = templates[min(index, len(templates) - 1)]
    entry = copy.deepcopy(template)
    entry["name"] = camera_id
    pipeline = entry["pipeline"].replace(MODEL_PROC_FROM, MODEL_PROC_TO)
    for template_stream in TEMPLATE_STREAMS:
      pipeline = pipeline.replace(template_stream, stream)
    entry["pipeline"] = pipeline
    entry["payload"]["parameters"]["camera_config"]["cameraid"] = camera_id
    pipelines.append(entry)

  config["config"]["pipelines"] = pipelines
  config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
  return rtsp_no_proxy_hosts(streams)


def main() -> None:
  parser = argparse.ArgumentParser(description="Adapt queuing-config.json for deployment cameras")
  parser.add_argument("--deploy-dir", required=True, type=Path)
  parser.add_argument("--camera-ids", required=True, nargs="+", metavar="CAMERA_ID")
  parser.add_argument("--streams", required=True, nargs="+", metavar="RTSP_URL")
  args = parser.parse_args()

  hosts = adapt_pipeline_config(args.deploy_dir, args.camera_ids, args.streams)
  if hosts:
    print("no_proxy_hosts=" + ",".join(hosts))


if __name__ == "__main__":
  main()
