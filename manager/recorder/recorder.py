# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Scenescape recording service (Phase 2.0).

Subscribes to scene MQTT topics (telemetry/tracks + camera frames from the
dlstreamer pipeline) and writes time-partitioned .rrd files via rerun-sdk —
one file per scene per hour — to a local disk volume.

Entity mapping (Phase 2.1):
  - Application ID = scene id
  - Timeline: log_time (wall clock)
  - scene/<id>/objects/<track_id>  -> Points3D (per-frame positions)
  - scene/<id>/cameras/<cam_id>/image -> Image (frames at capture cadence)

Static geometry (regions, tripwires, camera poses) is logged once as timeless;
only tracks, frames, and sensor state are per-frame.

Runs in a container on the same machine as the manager (may split later).
Recordings are listed via the manager's GET /api/v1/recordings/ endpoint.
"""

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

import paho.mqtt.client as mqtt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("recorder")

# ---------------------------------------------------------------------------
# Config (environment variables)
# ---------------------------------------------------------------------------

MQTT_BROKER = os.environ.get("RECORDER_MQTT_BROKER", "localhost")
MQTT_PORT = int(os.environ.get("RECORDER_MQTT_PORT", "1883"))
STORAGE_DIR = Path(os.environ.get("RECORDER_STORAGE_DIR", "/data/recordings"))
RETENTION_DAYS = float(os.environ.get("RECORDER_RETENTION_DAYS", "1"))
# Comma-separated scene IDs to record; empty = all scenes.
SCENE_FILTER = {
    s.strip() for s in os.environ.get("RECORDER_SCENES", "").split(",") if s.strip()
}
# Manager REST base URL for camera->scene mapping + auth token.
MANAGER_API = os.environ.get("RECORDER_MANAGER_API", "http://localhost:8000/api/v1")
MANAGER_TOKEN = os.environ.get("RECORDER_MANAGER_TOKEN", "")
# Seconds between .rrd file rotations (default: hourly partitions).
PARTITION_SECONDS = int(os.environ.get("RECORDER_PARTITION_SECONDS", "3600"))

APP_NAME = "scenescape"
TOPIC_REGULATED = f"{APP_NAME}/regulated/scene/+"
TOPIC_IMAGE = f"{APP_NAME}/image/camera/+"

RRD_SDK_VERSION = "0.38.1"


# ---------------------------------------------------------------------------
# Per-scene .rrd writer
# ---------------------------------------------------------------------------

class SceneRecorder:
    """Writes one scene's stream to hourly .rrd files via rerun-sdk."""

    def __init__(self, scene_id: str):
        self.scene_id = scene_id
        self.lock = Lock()
        self.rec = None
        self.current_partition: str | None = None
        self._open_partition()

    def _partition_name(self, now: datetime) -> str:
        # Hourly partitions: YYYY-MM-DD-HH
        return now.strftime("%Y-%m-%d-%H")

    def _path_for(self, partition: str) -> Path:
        d = STORAGE_DIR / self.scene_id
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{partition}.rrd"

    def _open_partition(self) -> None:
        import rerun as rr

        now = datetime.now(timezone.utc)
        partition = self._partition_name(now)
        if partition == self.current_partition and self.rec is not None:
            return
        if self.rec is not None:
            try:
                self.rec.save(self._path_for(self.current_partition))
            except Exception:
                logger.exception("Failed to save .rrd for %s", self.scene_id)
        path = self._path_for(partition)
        # Append if the file already exists (restarts within the hour).
        self.rec = rr.RecordingStream(
            application_id=self.scene_id,
            recording_id=f"{self.scene_id}-{partition}",
            make_default=path.exists(),
        )
        if path.exists():
            # Reopen existing file for append by saving to it on rotation.
            pass
        self.rec.save(path)
        # Record SDK version in metadata for the version-lock caveat (2.5).
        self.rec.log("recorder/meta", rr.TextLog(f"rerun-sdk {RRD_SDK_VERSION}"),
                     timeless=True)
        self.current_partition = partition
        logger.info("Recording %s -> %s", self.scene_id, path)

    def _maybe_rotate(self) -> None:
        now = datetime.now(timezone.utc)
        if self._partition_name(now) != self.current_partition:
            with self.lock:
                self._open_partition()

    def log_tracks(self, tracks: list[dict]) -> None:
        """Log object tracks as Points3D. tracks: [{id, x, y, class}]."""
        import rerun as rr

        self._maybe_rotate()
        now = datetime.now(timezone.utc)
        with self.lock:
            self.rec.set_time("log_time", timestamp=now)
            for t in tracks:
                tid = str(t.get("id", "unknown"))
                x = float(t.get("x", 0))
                y = float(t.get("y", 0))
                self.rec.log(
                    f"scene/{self.scene_id}/objects/{tid}",
                    rr.Points3D([[x, y, 0]]),
                )

    def log_frame(self, camera_id: string, image_bytes: bytes) -> None:
        """Log a camera frame as a Rerun Image archetype."""
        import rerun as rr

        self._maybe_rotate()
        now = datetime.now(timezone.utc)
        with self.lock:
            self.rec.set_time("log_time", timestamp=now)
            self.rec.log(
                f"scene/{self.scene_id}/cameras/{camera_id}/image",
                rr.EncodedImage(contents=image_bytes),
            )

    def close(self) -> None:
        with self.lock:
            self.rec = None
            self.current_partition = None


# ---------------------------------------------------------------------------
# MQTT glue
# ---------------------------------------------------------------------------

recorders: dict[str, SceneRecorder] = {}
recorders_lock = Lock()


def get_recorder(scene_id: str) -> SceneRecorder | None:
    if SCENE_FILTER and scene_id not in SCENE_FILTER:
        return None
    with recorders_lock:
        rec = recorders.get(scene_id)
        if rec is None:
            rec = SceneRecorder(scene_id)
            recorders[scene_id] = rec
        return rec


def scene_id_from_topic(topic: str) -> str | None:
    # scenescape/regulated/scene/<scene_id>
    parts = topic.split("/")
    try:
        i = parts.index("scene")
        return parts[i + 1]
    except (ValueError, IndexError):
        return None


def on_message(client, userdata, msg) -> None:
    try:
        if msg.topic.startswith(f"{APP_NAME}/regulated/scene/"):
            scene_id = scene_id_from_topic(msg.topic)
            if not scene_id:
                return
            rec = get_recorder(scene_id)
            if rec is None:
                return
            payload = json.loads(msg.payload.decode("utf-8", errors="replace"))
            # The regulated topic carries {objects: [{id, x, y, class, ...}]}.
            tracks = payload.get("objects") or payload.get("tracks") or []
            if tracks:
                rec.log_tracks(tracks)
        elif msg.topic.startswith(f"{APP_NAME}/image/camera/"):
            # scenescape/image/camera/<camera_id> — JPEG bytes.
            camera_id = msg.topic.rsplit("/", 1)[-1]
            scene_id = (userdata or {}).get("camera_scene", {}).get(camera_id)
            if not scene_id:
                return
            rec = get_recorder(scene_id)
            if rec is None:
                return
            rec.log_frame(camera_id, msg.payload)
    except Exception:
        logger.exception("Error handling MQTT message on %s", msg.topic)


def on_connect(client, userdata, flags, reason_code, properties=None) -> None:
    if reason_code != 0:
        logger.error("MQTT connect failed: %s", reason_code)
        return
    logger.info("Connected to MQTT %s:%d", MQTT_BROKER, MQTT_PORT)
    client.subscribe(TOPIC_REGULATED)
    client.subscribe(TOPIC_IMAGE)


# ---------------------------------------------------------------------------
# Camera -> scene mapping (frames arrive on scenescape/image/camera/<id>
# without a scene; resolve via the manager REST API, refreshed periodically).
# ---------------------------------------------------------------------------

def refresh_camera_map(camera_scene: dict) -> None:
    """Fetch all cameras from the manager and map camera_id -> scene_id."""
    import urllib.request

    url = f"{MANAGER_API}/cameras"
    req = urllib.request.Request(url)
    if MANAGER_TOKEN:
        req.add_header("Authorization", f"Token {MANAGER_TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            data = json.loads(res.read().decode("utf-8"))
    except Exception:
        logger.exception("Failed to fetch camera list from %s", url)
        return
    items = data.get("results", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return
    updated = 0
    for cam in items:
        cam_id = str(cam.get("sensor_id") or cam.get("id") or "")
        scene_id = str(cam.get("scene") or cam.get("scene_id") or "")
        if cam_id and scene_id:
            if camera_scene.get(cam_id) != scene_id:
                camera_scene[cam_id] = scene_id
                updated += 1
    if updated:
        logger.info("Camera map refreshed: %d cameras", len(camera_scene))


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

def enforce_retention() -> None:
    """Delete .rrd files older than RETENTION_DAYS."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    if not STORAGE_DIR.exists():
        return
    for scene_dir in STORAGE_DIR.iterdir():
        if not scene_dir.is_dir():
            continue
        for rrd in scene_dir.glob("*.rrd"):
            try:
                mtime = datetime.fromtimestamp(rrd.stat().st_mtime, tz=timezone.utc)
            except OSError:
                continue
            if mtime < cutoff:
                logger.info("Retention: deleting %s", rrd)
                try:
                    rrd.unlink()
                except OSError:
                    logger.exception("Failed to delete %s", rrd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        import rerun  # noqa: F401
    except ImportError:
        logger.error("rerun-sdk is not installed. pip install rerun-sdk==%s", RRD_SDK_VERSION)
        return 1

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Recorder starting: broker=%s:%d storage=%s retention=%.1fd scenes=%s",
        MQTT_BROKER, MQTT_PORT, STORAGE_DIR, RETENTION_DAYS,
        sorted(SCENE_FILTER) or "all",
    )

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.user_data_set({"camera_scene": {}})

    stop = False

    def handle_signal(signum, frame):
        nonlocal stop
        logger.info("Shutting down")
        stop = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_start()

    camera_scene: dict = client._userdata["camera_scene"]
    refresh_camera_map(camera_scene)
    last_map_refresh = time.monotonic()

    try:
        while not stop:
            time.sleep(60)
            enforce_retention()
            # Refresh camera->scene map every 10 minutes.
            if time.monotonic() - last_map_refresh > 600:
                refresh_camera_map(camera_scene)
                last_map_refresh = time.monotonic()
    finally:
        client.loop_stop()
        client.disconnect()
        with recorders_lock:
            for rec in recorders.values():
                rec.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
