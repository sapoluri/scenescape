# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Submit a reconstruction job to the SceneScape mapping service, then poll the manager's
generate-mesh-status endpoint to finalize.  The manager automatically applies
alignMeshToXYPlane() and _transformCamerasWithMeshAlignment() before persisting.

Usage:
    python reconstruct_and_finalize.py \
        --deploy-dir ~/scenescape-deployment \
        --frames-dir /tmp/frames \
        --cameras camera1 camera2 \
        --scene-uid <existing-scene-uid>

    # Or let the script create a new scene:
    python reconstruct_and_finalize.py \
        --deploy-dir ~/scenescape-deployment \
        --frames-dir /tmp/frames \
        --cameras camera1 camera2 \
        --scene-name my_scene

The script exits 0 on success and prints the scene UID.
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

POLL_INTERVAL_S = 5
POLL_TIMEOUT_S = 900  # 15 minutes
IDENTITY_TRANSFORM = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def manager_session(manager_url: str, ca_cert: Path, username: str, password: str) -> tuple[requests.Session, str]:
    """Create a requests Session and return (session, token) authenticated to the manager."""
    session = requests.Session()
    session.verify = str(ca_cert)

    resp = session.post(
        f"{manager_url}/api/v1/auth",
        json={"username": username, "password": password},
        timeout=10,
    )
    resp.raise_for_status()
    token = resp.json()["token"]
    session.headers.update({"Authorization": f"Token {token}"})
    return session, token


def submit_reconstruction(mapping_url: str, ca_cert: Path, frames_dir: Path, camera_ids: list[str]) -> str:
    """POST images to the mapping service and return request_id."""
    files = []
    for camera_id in camera_ids:
        frame_path = frames_dir / f"{camera_id}.jpg"
        if not frame_path.exists():
            raise FileNotFoundError(f"Missing frame: {frame_path}")
        files.append(("images", (f"{camera_id}.jpg", io.BytesIO(frame_path.read_bytes()), "image/jpeg")))
        files.append(("camera_ids", (None, camera_id)))

    resp = requests.post(
        f"{mapping_url}/reconstruction",
        data={"output_format": "glb", "mesh_type": "mesh"},
        files=files,
        verify=str(ca_cert),
        timeout=90,
    )
    resp.raise_for_status()
    request_id = resp.json()["request_id"]
    print(f"Reconstruction queued: {request_id}")
    return request_id


def create_scene(session: requests.Session, manager_url: str, scene_name: str) -> str:
    """Create an empty scene and return its UID."""
    resp = session.post(
        f"{manager_url}/api/v1/scene",
        json={"name": scene_name, "transform": IDENTITY_TRANSFORM},
        timeout=10,
    )
    resp.raise_for_status()
    scene_uid = resp.json()["uid"]
    print(f"Scene created: {scene_uid}")
    return scene_uid


def finalize_mesh(manager_url: str, ca_cert: Path, username: str, password: str, scene_uid: str, request_id: str) -> None:
    """
    Poll the manager's generate-mesh-status endpoint until finalized.

    The manager's generate-mesh-status view requires a Django superuser session
    (not just a token), so we log in with a cookie-based session here.
    """
    import re

    session = requests.Session()
    session.verify = str(ca_cert)

    # Obtain CSRF token from login page
    login_page = session.get(f"{manager_url}/sign-in")
    match = re.search(r'<input[^>]*name="csrfmiddlewaretoken"[^>]*value="([^"]*)"', login_page.text)
    csrf_token = match.group(1) if match else session.cookies.get("csrftoken", "")

    login_resp = session.post(
        f"{manager_url}/sign-in",
        data={"username": username, "password": password, "csrfmiddlewaretoken": csrf_token},
        allow_redirects=True,
    )
    login_resp.raise_for_status()

    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        resp = session.post(
            f"{manager_url}/scene/generate-mesh-status/{scene_uid}/",
            params={"request_id": request_id},
            timeout=30,
        )
        resp.raise_for_status()
        status = resp.json()
        state = status.get("state", "")

        if status.get("finalized"):
            print(f"Mesh finalized. (state={state})")
            return

        if state == "failed":
            raise RuntimeError(f"Mesh finalization failed: {status.get('error')}")

        print(f"  state={state}")
        time.sleep(POLL_INTERVAL_S)

    raise TimeoutError("Mesh finalization did not complete within 15 minutes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct mesh and finalize scene via SceneScape manager")
    parser.add_argument("--deploy-dir", required=True, type=Path)
    parser.add_argument("--frames-dir", required=True, type=Path, help="Directory containing per-camera JPEG frames")
    parser.add_argument("--cameras", required=True, nargs="+", metavar="CAMERA_ID")
    parser.add_argument("--mapping-url", default="https://mapping.scenescape.intel.com:8444/v1")
    parser.add_argument("--manager-url", default="https://web.scenescape.intel.com")

    scene_group = parser.add_mutually_exclusive_group(required=True)
    scene_group.add_argument("--scene-uid", help="UID of an existing scene to finalize into")
    scene_group.add_argument("--scene-name", help="Name for a new scene to create")
    args = parser.parse_args()

    deploy_dir: Path = args.deploy_dir
    ca_cert = deploy_dir / "secrets" / "certs" / "scenescape-ca.pem"
    supass = (deploy_dir / "secrets" / "supass").read_text().strip()

    # Authenticate
    session, _token = manager_session(args.manager_url, ca_cert, "admin", supass)

    # Get or create scene
    scene_uid = args.scene_uid
    if scene_uid is None:
        scene_uid = create_scene(session, args.manager_url, args.scene_name)

    # Submit reconstruction
    request_id = submit_reconstruction(args.mapping_url, ca_cert, args.frames_dir, args.cameras)

    # Finalize via manager (applies alignment automatically)
    finalize_mesh(args.manager_url, ca_cert, "admin", supass, scene_uid, request_id)

    print(f"Done. Scene UID: {scene_uid}")


if __name__ == "__main__":
    main()
