# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Collect calibration frames from one or more SceneScape cameras via MQTT.

Usage:
    python capture_calibration_frames.py \
        --deploy-dir ~/scenescape-deployment \
        --cameras camera1 camera2 \
        --out-dir /tmp/frames

Outputs one file per camera: <out_dir>/<camera_id>.jpg
"""

import argparse
import base64
import json
import ssl
import sys
import threading
from pathlib import Path


def collect_frames(deploy_dir: Path, camera_ids: list[str], out_dir: Path, timeout_per_camera: int = 30) -> None:
    import paho.mqtt.client as mqtt

    ca_cert = deploy_dir / "secrets" / "certs" / "scenescape-ca.pem"
    auth_file = deploy_dir / "secrets" / "browser.auth"

    with open(auth_file) as f:
        auth = json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    images: dict[str, bytes] = {}
    expected = set(camera_ids)
    done = threading.Event()

    def on_message(client, userdata, msg):
        data = json.loads(msg.payload)
        camera_id = msg.topic.split("/")[-1]
        if "image" in data and camera_id in expected:
            images[camera_id] = base64.b64decode(data["image"])
            if expected <= set(images):
                done.set()

    client = mqtt.Client()
    client.tls_set(ca_certs=str(ca_cert))
    client.username_pw_set(auth["user"], auth["password"])
    client.on_message = on_message
    client.connect("broker.scenescape.intel.com", 1883, 60)
    client.subscribe("scenescape/image/calibration/camera/+", qos=2)
    client.loop_start()

    for camera_id in camera_ids:
        client.publish(f"scenescape/cmd/camera/{camera_id}", "getcalibrationimage", qos=2)

    timeout = timeout_per_camera * len(camera_ids)
    if not done.wait(timeout=timeout):
        missing = expected - set(images)
        client.loop_stop()
        client.disconnect()
        raise TimeoutError(f"No calibration image received from: {missing}")

    client.loop_stop()
    client.disconnect()

    for camera_id, img_bytes in images.items():
        out_path = out_dir / f"{camera_id}.jpg"
        out_path.write_bytes(img_bytes)
        print(f"Saved {out_path} ({len(img_bytes)} bytes)")

    print(f"Collected {len(images)} frame(s): {sorted(images)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect calibration frames from SceneScape cameras via MQTT")
    parser.add_argument("--deploy-dir", required=True, type=Path, help="Path to scenescape-deployment directory")
    parser.add_argument("--cameras", required=True, nargs="+", metavar="CAMERA_ID", help="One or more camera IDs")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory to write captured JPEG files")
    parser.add_argument("--timeout-per-camera", type=int, default=30, metavar="SECONDS", help="Seconds to wait per camera (default: 30)")
    args = parser.parse_args()

    collect_frames(
        deploy_dir=args.deploy_dir,
        camera_ids=args.cameras,
        out_dir=args.out_dir,
        timeout_per_camera=args.timeout_per_camera,
    )


if __name__ == "__main__":
    main()
