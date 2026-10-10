// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from "react";
import {
  Mesh,
  MeshBasicMaterial,
  Plane,
  Raycaster,
  SphereGeometry,
  Vector2,
  Vector3,
} from "three";
import { parseCameraOptics } from "../../lib/cameraOptics";
import { api } from "../../lib/rest";
import { restPoseToRigEuler } from "../entities";
import { getViewportState } from "../store";
import type { ViewportWorld } from "../world";
import { getCalibratePairs, useCalibrateStore } from "./calibrateStore";
import { calibrationPointColorHex } from "./colors";
import { solveCameraPose } from "./poseSolve";

const FLOOR = new Plane(new Vector3(0, 0, 1), 0);
/** World-space marker radius (legacy `SPHERE_RADIUS` ≈ 0.05). */
const MARKER_RADIUS = 0.05;
/** Screen-space pick radius for hit / drag / remove (px). */
const MARKER_HIT_PX = 18;

/**
 * Scene point under the cursor on the z=0 floor plane.
 * Do NOT raycast the full map GLB here — large meshes freeze the main
 * thread on every click. Calibration correspondences are floor points.
 */
function pickScenePoint(
  world: ViewportWorld,
  clientX: number,
  clientY: number,
): [number, number, number] | null {
  const rect = world.host.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    return null;
  }
  const ndc = new Vector2(
    ((clientX - rect.left) / rect.width) * 2 - 1,
    -(((clientY - rect.top) / rect.height) * 2 - 1),
  );
  const raycaster = new Raycaster();
  raycaster.setFromCamera(ndc, world.getActiveCamera());
  const hit = new Vector3();
  if (raycaster.ray.intersectPlane(FLOOR, hit)) {
    return [hit.x, hit.y, 0];
  }
  return null;
}

/**
 * Nearest calibration marker in screen space. Mesh raycasts miss small
 * spheres once the view is zoomed out or other entities occlude them.
 */
function nearestMapIndex(
  world: ViewportWorld,
  clientX: number,
  clientY: number,
  mapPoints: [number, number, number][],
): number {
  const rect = world.host.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    return -1;
  }
  const cam = world.getActiveCamera();
  const tmp = new Vector3();
  let best = -1;
  let bestD2 = MARKER_HIT_PX * MARKER_HIT_PX;
  mapPoints.forEach((map, i) => {
    tmp.set(map[0], map[1], map[2] + MARKER_RADIUS);
    tmp.project(cam);
    if (tmp.z < -1 || tmp.z > 1) {
      return;
    }
    const sx = (tmp.x * 0.5 + 0.5) * rect.width + rect.left;
    const sy = (-tmp.y * 0.5 + 0.5) * rect.height + rect.top;
    const dx = sx - clientX;
    const dy = sy - clientY;
    const d2 = dx * dx + dy * dy;
    if (d2 <= bestD2) {
      bestD2 = d2;
      best = i;
    }
  });
  return best;
}

/**
 * Loads optics, captures prior frustum pose, floor-picks scene points,
 * draws pair markers, and live-solves the camera frustum.
 *
 * Map interactions match the legacy 3D calibrator: double-click to add,
 * drag to move, right-click to remove. Orbit stays available on empty
 * left-drag.
 */
export function useCalibrateSession(
  world: ViewportWorld | null,
  authToken: string,
): void {
  const active = useCalibrateStore((s) => s.active);
  const cameraId = useCalibrateStore((s) => s.cameraId);
  const sensorId = useCalibrateStore((s) => s.sensorId);
  const cameraName = useCalibrateStore((s) => s.cameraName);
  const mapPoints = useCalibrateStore((s) => s.mapPoints);
  const camPoints = useCalibrateStore((s) => s.camPoints);
  const optics = useCalibrateStore((s) => s.optics);

  // Load optics + snapshot prior pose when entering / switching camera.
  // Do not preload OpenCV.js here — parsing/WASM init freezes the UI.
  useEffect(() => {
    if (!active || !cameraId || !sensorId || !authToken) {
      return;
    }
    let cancelled = false;
    const entity = getViewportState().entities[cameraId];
    if (entity?.type === "camera") {
      useCalibrateStore.getState().setPriorPose({
        position: [...entity.position] as [number, number, number],
        rotation: [...entity.rotation] as [number, number, number],
        fov: entity.fov,
      });
      getViewportState().select(cameraId);
    }
    (async () => {
      try {
        const row = (await api.getCamera(authToken, sensorId)) ?? {};
        if (cancelled) {
          return;
        }
        useCalibrateStore
          .getState()
          .setOptics(parseCameraOptics(row, cameraName));
      } catch {
        if (!cancelled) {
          useCalibrateStore
            .getState()
            .setOptics(parseCameraOptics({}, cameraName));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [active, cameraId, sensorId, authToken, cameraName]);

  // Double-click add · drag move · right-click remove on the 3D map.
  useEffect(() => {
    if (!active || !world) {
      return;
    }
    const host = world.host;
    const orbit = world.orbit;
    let dragIndex: number | null = null;
    // Avoid removing twice when both pointerdown(button=2) and contextmenu fire.
    let rightClickHandled = false;

    const tryRemoveAt = (clientX: number, clientY: number): boolean => {
      const idx = nearestMapIndex(
        world,
        clientX,
        clientY,
        useCalibrateStore.getState().mapPoints,
      );
      if (idx < 0) {
        return false;
      }
      useCalibrateStore.getState().removeMapPointAt(idx);
      return true;
    };

    const onPointerDown = (ev: PointerEvent) => {
      if (ev.button === 2) {
        if (tryRemoveAt(ev.clientX, ev.clientY)) {
          rightClickHandled = true;
          ev.preventDefault();
          ev.stopPropagation();
        }
        return;
      }
      if (ev.button !== 0) {
        return;
      }
      const idx = nearestMapIndex(
        world,
        ev.clientX,
        ev.clientY,
        useCalibrateStore.getState().mapPoints,
      );
      if (idx < 0) {
        return;
      }
      dragIndex = idx;
      orbit.enabled = false;
      ev.preventDefault();
      ev.stopPropagation();
    };

    const onPointerMove = (ev: PointerEvent) => {
      if (dragIndex === null) {
        return;
      }
      const map = pickScenePoint(world, ev.clientX, ev.clientY);
      if (!map) {
        return;
      }
      useCalibrateStore.getState().moveMapPoint(dragIndex, map);
      ev.preventDefault();
      ev.stopPropagation();
    };

    const endDrag = () => {
      if (dragIndex === null) {
        return;
      }
      dragIndex = null;
      orbit.enabled = true;
    };

    const onDblClick = (ev: MouseEvent) => {
      ev.preventDefault();
      ev.stopPropagation();
      const map = pickScenePoint(world, ev.clientX, ev.clientY);
      if (!map) {
        return;
      }
      useCalibrateStore.getState().addMapPoint(map);
    };

    const onContextMenu = (ev: MouseEvent) => {
      ev.preventDefault();
      ev.stopPropagation();
      if (rightClickHandled) {
        rightClickHandled = false;
        return;
      }
      tryRemoveAt(ev.clientX, ev.clientY);
    };

    host.addEventListener("pointerdown", onPointerDown, true);
    host.addEventListener("pointermove", onPointerMove, true);
    host.addEventListener("pointerup", endDrag, true);
    host.addEventListener("pointercancel", endDrag, true);
    host.addEventListener("dblclick", onDblClick, true);
    host.addEventListener("contextmenu", onContextMenu, true);
    return () => {
      orbit.enabled = true;
      host.removeEventListener("pointerdown", onPointerDown, true);
      host.removeEventListener("pointermove", onPointerMove, true);
      host.removeEventListener("pointerup", endDrag, true);
      host.removeEventListener("pointercancel", endDrag, true);
      host.removeEventListener("dblclick", onDblClick, true);
      host.removeEventListener("contextmenu", onContextMenu, true);
    };
  }, [active, world]);

  // Floor markers for map-side points (colors match camera feed by index).
  useEffect(() => {
    if (!world) {
      return;
    }
    const group = world.entityRoot;
    const markers: Mesh[] = [];
    const mats: MeshBasicMaterial[] = [];
    const geo = new SphereGeometry(MARKER_RADIUS, 12, 12);
    if (active) {
      mapPoints.forEach((p, i) => {
        const mat = new MeshBasicMaterial({
          color: calibrationPointColorHex(i),
          depthTest: true,
          depthWrite: true,
        });
        mats.push(mat);
        const m = new Mesh(geo, mat);
        m.position.set(p[0], p[1], p[2] + MARKER_RADIUS);
        m.name = `ss-cal-marker-${i}`;
        m.userData.ssCalMarker = true;
        m.frustumCulled = false;
        group.add(m);
        markers.push(m);
      });
    }
    return () => {
      markers.forEach((m) => {
        group.remove(m);
      });
      geo.dispose();
      mats.forEach((mat) => mat.dispose());
    };
  }, [world, active, mapPoints]);

  const feedImageUrl = useCalibrateStore((s) => s.feedImageUrl);
  const previewPose = useCalibrateStore((s) => s.previewPose);

  // Live solve when both sides have the same count (≥4).
  const equalCount =
    camPoints.length === mapPoints.length ? camPoints.length : 0;
  const pairsKey =
    equalCount >= 4
      ? camPoints
          .map(
            (c, i) =>
              `${c[0]},${c[1]}:${mapPoints[i][0]},${mapPoints[i][1]},${mapPoints[i][2]}`,
          )
          .join("|")
      : "";

  useEffect(() => {
    if (!active || !cameraId || !optics || equalCount < 4) {
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const pairs = getCalibratePairs();
        if (pairs.length < 4) {
          return;
        }
        const pose = await solveCameraPose(pairs, optics);
        if (cancelled || !pose) {
          return;
        }
        useCalibrateStore.getState().setPreviewPose(pose);
        getViewportState().updateEntity(cameraId, {
          position: pose.position,
          rotation: restPoseToRigEuler(pose.rotation),
          restRotation: pose.rotation,
          posed: true,
          fov: pose.fov,
        });
      } catch (err) {
        console.warn("Calibrate pose solve failed", err);
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- pairsKey stands in for points
  }, [active, cameraId, optics, pairsKey, equalCount]);

  // Apply solved pose to the projector once (reproject camera matrices).
  useEffect(() => {
    if (!active || !world?.cameraProjection || !previewPose) {
      return;
    }
    world.cameraProjection.updatePose({
      fov: previewPose.fov,
      position: previewPose.position,
      rotationDeg: restPoseToRigEuler(previewPose.rotation),
    });
    world.cameraProjection.setOpacity(getViewportState().projectOpacity);
    world.cameraProjection.setVisible(true);
  }, [active, world, previewPose]);

  // Refresh projection JPEG without reprojecting on every MQTT frame.
  // Throttle so decode/upload cannot freeze the UI after the 4th point.
  useEffect(() => {
    if (!active || !world?.cameraProjection || !feedImageUrl || !previewPose) {
      return;
    }
    let cancelled = false;
    const handle = world.cameraProjection;
    const timer = window.setTimeout(() => {
      if (cancelled) {
        return;
      }
      handle
        .updateFrame(feedImageUrl, {
          fov: previewPose.fov,
          position: previewPose.position,
          rotationDeg: restPoseToRigEuler(previewPose.rotation),
          reproject: false,
        })
        .then(() => {
          if (!cancelled) {
            handle.setOpacity(getViewportState().projectOpacity);
            handle.setVisible(true);
          }
        })
        .catch((err) => console.warn("Map projection update failed", err));
    }, 250);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [active, world, feedImageUrl, previewPose]);

  useEffect(() => {
    if (!active) {
      useCalibrateStore.getState().setPreviewPose(null);
      // Leave the overlay up when Live "Project frame" is still on.
      if (!getViewportState().projectCameraId) {
        world?.cameraProjection?.setVisible(false);
      }
    }
  }, [active, world]);
}
