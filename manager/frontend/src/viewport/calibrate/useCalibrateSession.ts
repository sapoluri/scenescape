// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from "react";
import {
  Mesh,
  MeshBasicMaterial,
  MOUSE,
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
import { useCalibrateStore } from "./calibrateStore";
import { calibrationPointColorHex } from "./colors";
import { solveCameraPose } from "./poseSolve";

const FLOOR = new Plane(new Vector3(0, 0, 1), 0);
/** World-space marker radius (legacy `SPHERE_RADIUS` ≈ 0.05). */
const MARKER_RADIUS = 0.05;
/** Screen-space pick radius for right-click remove (px). */
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

type MarkerHit = { kind: "pair"; index: number } | { kind: "draft" };

/**
 * Nearest calibration marker in screen space. Mesh raycasts miss small
 * spheres once the view is zoomed out or other entities occlude them.
 */
function nearestMarkerHit(
  world: ViewportWorld,
  clientX: number,
  clientY: number,
  pairs: { map: [number, number, number] }[],
  draftMap: [number, number, number] | null,
): MarkerHit | null {
  const rect = world.host.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    return null;
  }
  const cam = world.getActiveCamera();
  const tmp = new Vector3();
  let bestHit: MarkerHit | null = null;
  let bestD2 = MARKER_HIT_PX * MARKER_HIT_PX;
  const consider = (
    map: [number, number, number],
    hit: MarkerHit,
  ) => {
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
      bestHit = hit;
    }
  };
  pairs.forEach((p, i) => consider(p.map, { kind: "pair", index: i }));
  if (draftMap) {
    consider(draftMap, { kind: "draft" });
  }
  return bestHit;
}

function removeMarkerHit(hit: MarkerHit): void {
  const s = useCalibrateStore.getState();
  if (hit.kind === "draft") {
    s.clearDraft();
  } else {
    s.removePairAt(hit.index);
  }
}

/**
 * Loads optics, captures prior frustum pose, floor-picks scene points,
 * draws pair markers, and live-solves the camera frustum.
 */
export function useCalibrateSession(
  world: ViewportWorld | null,
  authToken: string,
): void {
  const active = useCalibrateStore((s) => s.active);
  const cameraId = useCalibrateStore((s) => s.cameraId);
  const sensorId = useCalibrateStore((s) => s.sensorId);
  const cameraName = useCalibrateStore((s) => s.cameraName);
  const pairs = useCalibrateStore((s) => s.pairs);
  const draftMap = useCalibrateStore((s) => s.draftMap);
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

  // Map left-click → start/complete pair; right-click → remove nearest marker.
  useEffect(() => {
    if (!active || !world) {
      return;
    }
    const host = world.host;
    const orbit = world.orbit;
    const prevLeft = orbit.mouseButtons.LEFT;
    orbit.mouseButtons.LEFT = null;
    // Avoid removing twice when both pointerdown(button=2) and contextmenu fire.
    let rightClickHandled = false;

    const tryRemoveAt = (clientX: number, clientY: number): boolean => {
      const s = useCalibrateStore.getState();
      const hit = nearestMarkerHit(
        world,
        clientX,
        clientY,
        s.pairs,
        s.draftMap,
      );
      if (!hit) {
        return false;
      }
      removeMarkerHit(hit);
      return true;
    };

    const onPointerDown = (ev: PointerEvent) => {
      if (ev.button === 2) {
        // Prefer pointerdown for remove — mesh raycasts get unreliable once
        // several markers / entities crowd the view.
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
      ev.preventDefault();
      ev.stopPropagation();
      const map = pickScenePoint(world, ev.clientX, ev.clientY);
      if (!map) {
        return;
      }
      useCalibrateStore.getState().pickMap(map);
    };

    const onContextMenu = (ev: MouseEvent) => {
      // Always suppress the browser menu while calibrating.
      ev.preventDefault();
      ev.stopPropagation();
      if (rightClickHandled) {
        rightClickHandled = false;
        return;
      }
      tryRemoveAt(ev.clientX, ev.clientY);
    };

    host.addEventListener("pointerdown", onPointerDown, true);
    host.addEventListener("contextmenu", onContextMenu, true);
    return () => {
      orbit.mouseButtons.LEFT = prevLeft ?? MOUSE.ROTATE;
      host.removeEventListener("pointerdown", onPointerDown, true);
      host.removeEventListener("contextmenu", onContextMenu, true);
    };
  }, [active, world]);

  // Floor markers: completed pairs + in-progress draft map point.
  // Colors match the camera-feed markers by pair index (legacy palette).
  useEffect(() => {
    if (!world) {
      return;
    }
    const group = world.entityRoot;
    const markers: Mesh[] = [];
    const mats: MeshBasicMaterial[] = [];
    const geo = new SphereGeometry(MARKER_RADIUS, 12, 12);
    if (active) {
      pairs.forEach((p, i) => {
        const mat = new MeshBasicMaterial({
          color: calibrationPointColorHex(i),
          depthTest: true,
          depthWrite: true,
        });
        mats.push(mat);
        const m = new Mesh(geo, mat);
        m.position.set(p.map[0], p.map[1], p.map[2] + MARKER_RADIUS);
        m.name = `ss-cal-marker-${i}`;
        m.userData.ssCalMarker = true;
        // Avoid frustum culling dropping tiny markers at glancing angles.
        m.frustumCulled = false;
        group.add(m);
        markers.push(m);
      });
      if (draftMap) {
        const mat = new MeshBasicMaterial({
          color: calibrationPointColorHex(pairs.length),
        });
        mats.push(mat);
        const m = new Mesh(geo, mat);
        m.position.set(draftMap[0], draftMap[1], draftMap[2] + MARKER_RADIUS);
        m.name = "ss-cal-marker-draft";
        m.userData.ssCalMarker = true;
        m.frustumCulled = false;
        group.add(m);
        markers.push(m);
      }
    }
    return () => {
      markers.forEach((m) => {
        group.remove(m);
      });
      geo.dispose();
      mats.forEach((mat) => mat.dispose());
    };
  }, [world, active, pairs, draftMap]);

  const feedImageUrl = useCalibrateStore((s) => s.feedImageUrl);
  const previewPose = useCalibrateStore((s) => s.previewPose);

  // Live solve → update frustum + remember REST pose for map projection.
  // Fingerprint pairs so we don't re-solve on unrelated store churn.
  const pairsKey = pairs
    .map(
      (p) =>
        `${p.cam[0]},${p.cam[1]}:${p.map[0]},${p.map[1]},${p.map[2]}`,
    )
    .join("|");

  useEffect(() => {
    if (!active || !cameraId || !optics || pairs.length < 4) {
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const pose = await solveCameraPose(
          useCalibrateStore.getState().pairs,
          optics,
        );
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
    // eslint-disable-next-line react-hooks/exhaustive-deps -- pairsKey stands in for pairs
  }, [active, cameraId, optics, pairsKey, pairs.length]);

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
