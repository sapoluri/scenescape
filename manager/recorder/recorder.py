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
import ssl
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
MQTT_CAFILE = os.environ.get("RECORDER_MQTT_CAFILE", "")
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
# Recorder sinks (Phase 2.6 backend seam)
#
# The recorder writes through a pluggable sink: RrdSink (rerun-sdk) today;
# JsonlSink / McapSink later. Sink choice is deployment config via
# RECORDER_SINK (default: rrd).
# ---------------------------------------------------------------------------

class Sink:
    """Abstract recording sink."""

    def open_partition(self, scene_id: str, partition: str, path: Path) -> None:
        raise NotImplementedError

    def log_tracks(self, scene_id: str, tracks: list[dict]) -> None:
        raise NotImplementedError

    def log_frame(self, scene_id: str, camera_id: str, image_bytes: bytes) -> None:
        raise NotImplementedError

    def close_partition(self, scene_id: str) -> None:
        raise NotImplementedError


class RrdSink(Sink):
    """Writes .rrd files via rerun-sdk (reference backend)."""

    def __init__(self):
        self.streams: dict[str, object] = {}

    def open_partition(self, scene_id: str, partition: str, path: Path) -> None:
        import rerun as rr

        rec = rr.RecordingStream(
            application_id=scene_id,
            recording_id=f"{scene_id}-{partition}",
        )
        rec.save(path)
        rec.log("recorder/meta", rr.TextLog(f"rerun-sdk {RRD_SDK_VERSION}"),
                static=True)
        self.streams[scene_id] = rec

    def log_tracks(self, scene_id: str, tracks: list[dict]) -> None:
        import rerun as rr
        from datetime import datetime, timezone

        rec = self.streams.get(scene_id)
        if rec is None:
            return
        rec.set_time("log_time", timestamp=datetime.now(timezone.utc))
        for t in tracks:
            tid = str(t.get("id", "unknown"))
            x, y = _track_xy(t)
            rec.log(f"scene/{scene_id}/objects/{tid}", rr.Points3D([[x, y, 0]]))

    def log_frame(self, scene_id: str, camera_id: str, image_bytes: bytes) -> None:
        import rerun as rr
        from datetime import datetime, timezone

        rec = self.streams.get(scene_id)
        if rec is None:
            return
        rec.set_time("log_time", timestamp=datetime.now(timezone.utc))
        rec.log(f"scene/{scene_id}/cameras/{camera_id}/image",
                rr.EncodedImage(contents=image_bytes, media_type="image/jpeg"))

    def close_partition(self, scene_id: str) -> None:
        self.streams.pop(scene_id, None)


def create_sink() -> Sink:
    name = os.environ.get("RECORDER_SINK", "rrd").lower()
    if name == "rrd":
        return RrdSink()
    raise ValueError(f"Unknown recorder sink: {name}")


# ---------------------------------------------------------------------------
# Per-scene .rrd writer
# ---------------------------------------------------------------------------

class SceneRecorder:
    """Writes one scene's stream via the configured sink."""

    def __init__(self, scene_id: str, sink: Sink):
        self.scene_id = scene_id
        self.sink = sink
        self.lock = Lock()
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
        now = datetime.now(timezone.utc)
        partition = self._partition_name(now)
        if partition == self.current_partition:
            return
        if self.current_partition is not None:
            self.sink.close_partition(self.scene_id)
        path = self._path_for(partition)
        self.sink.open_partition(self.scene_id, partition, path)
        self.current_partition = partition
        logger.info("Recording %s -> %s", self.scene_id, path)

    def _maybe_rotate(self) -> None:
        now = datetime.now(timezone.utc)
        if self._partition_name(now) != self.current_partition:
            with self.lock:
                self._open_partition()

    def log_tracks(self, tracks: list[dict]) -> None:
        """Log object tracks. tracks: [{id, x, y, class}]."""
        self._maybe_rotate()
        with self.lock:
            self.sink.log_tracks(self.scene_id, tracks)

    def log_frame(self, camera_id: str, image_bytes: bytes) -> None:
        """Log a camera frame."""
        self._maybe_rotate()
        with self.lock:
            self.sink.log_frame(self.scene_id, camera_id, image_bytes)

    def close(self) -> None:
        with self.lock:
            self.sink.close_partition(self.scene_id)
            self.current_partition = None


# ---------------------------------------------------------------------------
# MQTT glue
# ---------------------------------------------------------------------------

recorders: dict[str, SceneRecorder] = {}
recorders_lock = Lock()
_sink: Sink | None = None


def get_recorder(scene_id: str) -> SceneRecorder | None:
    if SCENE_FILTER and scene_id not in SCENE_FILTER:
        return None
    with recorders_lock:
        rec = recorders.get(scene_id)
        if rec is None:
            rec = SceneRecorder(scene_id, _sink)
            recorders[scene_id] = rec
        return rec


def _track_xy(track: dict) -> tuple[float, float]:
    """Scene objects use translation[x,y,z]; some payloads also send x/y."""
    trans = track.get("translation")
    if isinstance(trans, (list, tuple)) and len(trans) >= 2:
        return float(trans[0]), float(trans[1])
    return float(track.get("x", 0) or 0), float(track.get("y", 0) or 0)


def _flatten_tracks(payload: dict) -> list[dict]:
    """Regulated scene payloads may use objects[] or objects{category: []}."""
    raw = payload.get("objects") or payload.get("tracks") or []
    if isinstance(raw, list):
        return [t for t in raw if isinstance(t, dict)]
    if isinstance(raw, dict):
        out: list[dict] = []
        for items in raw.values():
            if isinstance(items, list):
                out.extend(t for t in items if isinstance(t, dict))
        return out
    return []


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
            tracks = _flatten_tracks(payload)
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

    url = f"{MANAGER_API.rstrip('/')}/cameras"
    req = urllib.request.Request(url)
    if MANAGER_TOKEN:
        req.add_header("Authorization", f"Token {MANAGER_TOKEN}")
    ctx = None
    if url.startswith("https://"):
        ctx = ssl.create_default_context()
        if MQTT_CAFILE:
            ctx.load_verify_locations(MQTT_CAFILE)
        # Demo web cert is issued for web.scenescape.intel.com; skip hostname
        # check when talking to the in-cluster alias over HTTPS.
        ctx.check_hostname = False
    try:
        with urllib.request.urlopen(req, timeout=10, context=ctx) as res:
            data = json.loads(res.read().decode("utf-8"))
    except Exception:
        logger.exception("Failed to fetch camera list from %s", url)
        return
    items = data.get("results", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return
    updated = 0
    for cam in items:
        scene_id = str(cam.get("scene") or cam.get("scene_id") or "")
        # List API exposes camera MQTT id as uid (sensor_id is write-only).
        cam_ids = {
            str(v) for v in (
                cam.get("uid"),
                cam.get("sensor_id"),
                cam.get("name"),
            ) if v not in (None, "")
        }
        if not scene_id or not cam_ids:
            continue
        for cam_id in cam_ids:
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
    global _sink
    try:
        import rerun  # noqa: F401
    except ImportError:
        logger.error("rerun-sdk is not installed. pip install rerun-sdk==%s", RRD_SDK_VERSION)
        return 1

    _sink = create_sink()
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Recorder starting: broker=%s:%d storage=%s retention=%.1fd scenes=%s",
        MQTT_BROKER, MQTT_PORT, STORAGE_DIR, RETENTION_DAYS,
        sorted(SCENE_FILTER) or "all",
    )

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    if MQTT_CAFILE:
        client.tls_set(
            ca_certs=MQTT_CAFILE,
            tls_version=ssl.PROTOCOL_TLS_CLIENT,
        )
        logger.info("MQTT TLS enabled with CA %s", MQTT_CAFILE)
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
