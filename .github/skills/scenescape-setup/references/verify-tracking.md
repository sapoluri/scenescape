# Verify End-to-End Object Tracking

Subscribe to `scenescape/regulated/scene/<scene_uid>` and confirm that tracked objects
appear within 2 minutes of containers being live.

## MQTT Verification

Use MQTT flags that match the broker listener mode. For the default generated broker config,
listener `1883` is plaintext.

Use the MQTT subscribe template from [command-templates.md](./command-templates.md) with topic:
`scenescape/regulated/scene/<scene_uid>`.

Pass criteria: the message contains an `objects` array with at least one tracked object.

## Troubleshooting Checklist

If no objects appear:

### 1. Check the Scene Controller logs

```bash
cd <deploy_dir>
docker compose logs scene --tail=50
```

Look for errors about: MQTT connection failures, schema validation errors, missing cameras.

### 2. Verify cameras are registered in the scene

```python
import requests
resp = requests.get(
    f"https://web.scenescape.intel.com/api/v1/cameras?scene={scene_uid}",
    headers=HEADERS, verify=CA_CERT,
)
cameras = resp.json()
print(f"{len(cameras)} camera(s) registered for scene {scene_uid}")
for c in cameras:
    print(f"  {c['name']}  sensor_id={c['sensor_id']}  uid={c['uid']}")
```

### 3. Check for non-zero camera pose

A camera with all-zero translation will be ignored by the controller.
Inspect `translation` in each registered camera's response JSON.

### 4. Check scene scale is non-zero

```python
resp = requests.get(
    f"https://web.scenescape.intel.com/api/v1/scene/{scene_uid}",
    headers=HEADERS, verify=CA_CERT,
)
print("scale =", resp.json().get("scale"))
```

A zero scale prevents regulated topic output. Fix: set a real-world scale via the UI
(Scene → Edit → Scale) or PATCH the field directly.

### 5. Verify raw detections are arriving from video-analytics

Use the MQTT subscribe template from [command-templates.md](./command-templates.md) with topic
`scenescape/data/camera/+`.

If no messages appear, video-analytics pipelines are not detecting anything. Check:

- RTSP streams are accessible from inside the container
- The model files exist in the `vol-models` volume
- `docker compose logs video-analytics --tail=30`

### 6. Confirm controller sees the scene

```bash
docker compose logs scene | grep -i "scene\|camera\|calibrat"
```

The controller should log something like `"Loading scene <scene_uid>"` and
`"Camera <camera_id> calibrated"` after cameras are registered.
