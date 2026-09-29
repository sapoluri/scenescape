# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""UI island bootstrap payloads — shared by Django templates and /api/v1/ui-bootstrap."""

from __future__ import annotations

import json

from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import get_object_or_404
from django.urls import reverse

from manager.models import Asset3D, Scene


def build_chrome_bootstrap(request) -> dict:
  from manager.context_processors import chrome_bootstrap

  return chrome_bootstrap(request)


def user_auth_token(user) -> str:
  if user is None or not getattr(user, "is_authenticated", False):
    return ""
  try:
    token = user.auth_token
    return str(token) if token else ""
  except ObjectDoesNotExist:
    return ""


def build_scenes_home_bootstrap(request) -> dict:
  scenes = Scene.objects.order_by("name")
  scenes_payload = []
  for scene in scenes:
    scenes_payload.append({
      "id": str(scene.id),
      "name": scene.name,
      "georeferenced": bool(scene.output_lla and scene.map_corners_lla),
      "thumbnailUrl": scene.thumbnail.url if scene.thumbnail else None,
      "mapUrl": scene.map.url if scene.map else None,
      "detailUrl": reverse("sceneDetail", args=[scene.id]),
      "detail3dUrl": reverse("scene_detail", args=[scene.id]),
      "manageUrl": f"{reverse('index')}?ss=scene-manage&id={scene.id}",
      "deleteUrl": (
        reverse("scene_delete", args=[scene.id])
        if request.user.is_superuser else None
      ),
      "counts": {
        "sensors": scene.sensor_set.count(),
        "regions": scene.regions.count(),
        "tripwires": scene.tripwires.count(),
      },
    })
  return {
    "authToken": user_auth_token(request.user),
    "isSuperuser": request.user.is_superuser,
    "scenes": scenes_payload,
  }


def build_scene_detail_bootstrap(request, scene_id) -> dict:
  from manager.views import getAllChildrenMetaData

  scene = get_object_or_404(Scene, pk=scene_id)
  child_rois, child_trips, child_sensors = getAllChildrenMetaData(scene_id)

  cameras = []
  sensors = []
  for sensor in scene.sensor_set.all().order_by("name"):
    if sensor.type == "camera":
      cameras.append({
        "id": str(sensor.id),
        "sensorId": sensor.sensor_id,
        "name": sensor.name,
        "calibrateHref": f"?ss=calibrate-cam&id={sensor.id}",
        "cmdTopic": f"scenescape/cmd/camera/{sensor.sensor_id}",
        "deleteUrl": (
          reverse("cam_delete", args=[sensor.id])
          if request.user.is_superuser else None
        ),
      })
    elif sensor.type == "generic":
      sensors.append({
        "id": str(sensor.id),
        "sensorId": sensor.sensor_id,
        "name": sensor.name,
        "iconUrl": sensor.icon.url if sensor.icon else None,
        "areaJson": sensor.areaJSON(),
        "calibrateHref": f"?ss=calibrate-sensor&id={sensor.id}",
        "editHref": f"?ss=sensor-edit&id={sensor.sensor_id}",
        "deleteUrl": (
          reverse("singleton_sensor_delete", args=[sensor.id])
          if request.user.is_superuser else None
        ),
      })

  children = []
  for link in scene.children.all():
    child = link.child
    child_name = child.name if child else (link.child_name or "Child")
    if child is not None:
      rest_uid = str(child.id)
    elif link.remote_child_id:
      rest_uid = str(link.remote_child_id)
    else:
      rest_uid = str(link.id)
    children.append({
      "id": str(link.id),
      "name": child_name,
      "childType": link.child_type,
      "remoteChildId": (
        str(link.remote_child_id) if link.remote_child_id else None
      ),
      "detailUrl": reverse("sceneDetail", args=[child.id]) if child else None,
      "thumbnailUrl": (
        child.thumbnail.url if child and child.thumbnail else None
      ),
      "mapUrl": child.map.url if child and child.map else None,
      "restUid": rest_uid,
      "editHref": f"?ss=child-edit&id={rest_uid}",
      "deleteUrl": (
        reverse("child_delete", args=[link.id])
        if request.user.is_superuser else None
      ),
    })

  try:
    regions = json.loads(scene.roiJSON() or "[]")
  except (TypeError, json.JSONDecodeError):
    regions = []
  try:
    tripwires = json.loads(scene.tripwireJSON() or "[]")
  except (TypeError, json.JSONDecodeError):
    tripwires = []

  return {
    "scene": {
      "id": str(scene.id),
      "name": scene.name,
      "scale": scene.scale,
      "mapUrl": scene.map.url if scene.map else None,
      "thumbnailUrl": scene.thumbnail.url if scene.thumbnail else None,
      "wssConnection": scene.wssConnection(),
      "outputLla": bool(scene.output_lla),
      "georeferenced": bool(scene.output_lla and scene.map_corners_lla),
    },
    "cameras": cameras,
    "sensors": sensors,
    "children": children,
    "regions": regions if isinstance(regions, list) else [],
    "tripwires": tripwires if isinstance(tripwires, list) else [],
    "assetMarkColors": {
      asset.name: asset.mark_color
      for asset in Asset3D.objects.all()
    },
    "counts": {
      "sensors": len(sensors),
      "regions": len(regions) if isinstance(regions, list) else 0,
      "tripwires": len(tripwires) if isinstance(tripwires, list) else 0,
      "children": len(children),
    },
    "urls": {
      "scenesHome": reverse("index"),
      "camList": reverse("cam_list"),
      "sensorList": reverse("singleton_sensor_list"),
      "scene3d": reverse("scene_detail", args=[scene.id]),
      "sceneEdit": (
        reverse("scene_update", args=[scene.id])
        if request.user.is_superuser else None
      ),
      "sceneDelete": (
        reverse("scene_delete", args=[scene.id])
        if request.user.is_superuser else None
      ),
      "camCreate": (
        f"{reverse('cam_create')}?scene={scene.id}"
        if request.user.is_superuser else None
      ),
    },
    "authToken": user_auth_token(request.user),
    "isSuperuser": request.user.is_superuser,
    "isKubernetes": bool(settings.KUBERNETES_SERVICE_HOST),
    "appVersion": getattr(settings, "APP_VERSION_NUMBER", None),
    "appGitCommit": getattr(settings, "APP_GIT_COMMIT", None),
    "googleMapsApiKey": getattr(settings, "GOOGLE_MAPS_API_KEY", "") or "",
    "mapboxApiKey": getattr(settings, "MAPBOX_API_KEY", "") or "",
    "deleteImpact": {
      "sensors": scene.sensor_set.count(),
      "regions": scene.regions.count(),
      "tripwires": scene.tripwires.count(),
    },
    "childRoiJson": child_rois,
    "childTripwireJson": child_trips,
    "childSensorJson": child_sensors,
    "scenes": [
      {
        "id": str(s.id),
        "name": s.name,
        "georeferenced": bool(s.output_lla and s.map_corners_lla),
        "mapUrl": s.map.url if s.map else None,
      }
      for s in Scene.objects.order_by("name")
    ],
  }


def resolve_ui_bootstrap(request, page: str, entity_id: str | None = None) -> dict:
  """Return bootstrap dict for page name. Raises ValueError for bad page/id."""
  page = (page or "").strip().lower()
  if page == "chrome":
    return build_chrome_bootstrap(request)
  if page in ("scenes", "scenes-home", "home"):
    return build_scenes_home_bootstrap(request)
  if page in ("scene", "scene-detail"):
    if not entity_id:
      raise ValueError("scene bootstrap requires id")
    return build_scene_detail_bootstrap(request, entity_id)
  raise ValueError(f"unknown page: {page}")
