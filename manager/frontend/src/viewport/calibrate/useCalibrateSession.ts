// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef } from "react";
import {
  BufferGeometry,
  Points,
  PointsMaterial,
  Plane,
  Raycaster,
  Vector2,
  Vector3,
} from "three";
import { parseCameraOptics } from "../../lib/cameraOptics";
import { api } from "../../lib/rest";
import { restPoseToRigEuler } from "../entities";
import {
  effectiveProjectOpacity,
  getViewportState,
  useViewportStore,
} from "../store";
import type { ViewportWorld } from "../world";
import { getCalibratePairs, useCalibrateStore } from "./calibrateStore";
import { calibrationPointColor, calibrationPointColorHex, CALIBRATION_POINT_RADIUS_PX, CALIBRATION_POINT_HIT_PX } from "./colors";
import { preloadOpenCv, solveCameraPose } from "./poseSolve";
import { fovFromOptics, refineIntrinsics } from "./refineIntrinsics";
import { transformsToPairs } from "./types";

const FLOOR = new Plane(new Vector3(0, 0, 1), 0);
/** Screen-space pick radius for hit / drag / remove (px). */
const MARKER_HIT_PX = CALIBRATION_POINT_HIT_PX;

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
    tmp.set(map[0], map[1], map[2]);
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
  /** Bumped on each solve-effect schedule; in-flight work checks this. */
  const solveEpochRef = useRef(0);

  // Load optics + snapshot prior pose when entering / switching camera.
  // Defer coverage seeding and OpenCV so the calibrate chrome can paint first.
  useEffect(() => {
    if (!active || !cameraId || !sensorId || !authToken) {
      return;
    }
    let cancelled = false;
    let seedRaf = 0;
    const entity = getViewportState().entities[cameraId];
    if (entity?.type === "camera") {
      const prior = {
        position: [...entity.position] as [number, number, number],
        rotation: [...entity.rotation] as [number, number, number],
        fov: entity.fov,
      };
      useCalibrateStore.getState().setPriorPose(prior);
      getViewportState().select(cameraId);
      // Seed coverage after two frames so enter does not hitch on project().
      if (entity.posed) {
        const restRot = entity.restRotation
          ? ([...entity.restRotation] as [number, number, number])
          : restPoseToRigEuler(entity.rotation);
        seedRaf = window.requestAnimationFrame(() => {
          seedRaf = window.requestAnimationFrame(() => {
            if (cancelled) {
              return;
            }
            useCalibrateStore.getState().setPreviewPose({
              position: prior.position,
              rotation: restRot,
              fov: prior.fov,
            });
          });
        });
      }
    }
    (async () => {
      try {
        const row = (await api.getCamera(authToken, sensorId)) ?? {};
        if (cancelled) {
          return;
        }
        const store = useCalibrateStore.getState();
        store.setOptics(parseCameraOptics(row, cameraName));
        // Hydrate saved correspondences (legacy addInitialCalibrationPoints).
        const stored = transformsToPairs(
          row.transforms,
          typeof row.transform_type === "string" ? row.transform_type : null,
        );
        if (
          stored &&
          store.active &&
          store.sensorId === sensorId &&
          !store.dirty &&
          store.camPoints.length === 0 &&
          store.mapPoints.length === 0
        ) {
          store.replacePoints(stored.camPoints, stored.mapPoints, false);
        }
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
      window.cancelAnimationFrame(seedRaf);
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
    let pointerDown = false;
    const priorCursor = host.style.cursor;
    host.style.cursor = "default";
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
      pointerDown = true;
      host.style.cursor = "grabbing";
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
        const overPoint = nearestMapIndex(
          world, ev.clientX, ev.clientY, useCalibrateStore.getState().mapPoints,
        ) >= 0;
        host.style.cursor = pointerDown ? "grabbing" : overPoint ? "crosshair" : "default";
        return;
      }
      const map = pickScenePoint(world, ev.clientX, ev.clientY);
      if (!map) {
        return;
      }
      // Move the mesh immediately; store sync only patches the same slot.
      const mesh = world.entityRoot.getObjectByName(
        `ss-cal-marker-${dragIndex}`,
      );
      if (mesh) {
        mesh.position.set(map[0], map[1], map[2]);
      }
      useCalibrateStore.getState().moveMapPoint(dragIndex, map);
      ev.preventDefault();
      ev.stopPropagation();
    };

    const endDrag = (ev: PointerEvent) => {
      pointerDown = false;
      host.style.cursor = nearestMapIndex(
        world, ev.clientX, ev.clientY, useCalibrateStore.getState().mapPoints,
      ) >= 0 ? "crosshair" : "default";
      dragIndex = null;
      orbit.enabled = true;
    };
    const onPointerLeave = () => {
      if (!pointerDown) {
        host.style.cursor = "default";
      }
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
    host.addEventListener("pointerleave", onPointerLeave, true);
    window.addEventListener("pointerup", endDrag);
    host.addEventListener("dblclick", onDblClick, true);
    host.addEventListener("contextmenu", onContextMenu, true);
    return () => {
      orbit.enabled = true;
      host.style.cursor = priorCursor;
      host.removeEventListener("pointerdown", onPointerDown, true);
      host.removeEventListener("pointermove", onPointerMove, true);
      host.removeEventListener("pointerup", endDrag, true);
      host.removeEventListener("pointercancel", endDrag, true);
      host.removeEventListener("pointerleave", onPointerLeave, true);
      window.removeEventListener("pointerup", endDrag);
      host.removeEventListener("dblclick", onDblClick, true);
      host.removeEventListener("contextmenu", onContextMenu, true);
    };
  }, [active, world]);

  // Floor markers + numbered labels — stable pool; only add/remove the delta.
  useEffect(() => {
    if (!world) {
      return;
    }
    const group = world.entityRoot;
    const markers: Points[] = [];
    const labels: HTMLDivElement[] = [];
    const mats: PointsMaterial[] = [];
    const geo = new BufferGeometry().setFromPoints([new Vector3()]);
    const proj = new Vector3();

    const discardTail = (from: number) => {
      for (let i = markers.length - 1; i >= from; i--) {
        group.remove(markers[i]);
        labels[i].remove();
        mats[i].dispose();
        markers.pop();
        labels.pop();
        mats.pop();
      }
    };

    const ensureCount = (n: number) => {
      while (markers.length < n) {
        const i = markers.length;
        const mat = new PointsMaterial({
          color: calibrationPointColorHex(i),
          size: CALIBRATION_POINT_RADIUS_PX * 2,
          sizeAttenuation: false,
          transparent: true,
          depthTest: false,
          depthWrite: false,
        });
        mat.onBeforeCompile = (shader) => {
          shader.fragmentShader = shader.fragmentShader.replace(
            "#include <clipping_planes_fragment>",
            "#include <clipping_planes_fragment>\nif (distance(gl_PointCoord, vec2(0.5)) > 0.5) discard;",
          );
        };
        const m = new Points(geo, mat);
        m.renderOrder = 1000;
        m.name = `ss-cal-marker-${i}`;
        m.userData.ssCalMarker = true;
        m.frustumCulled = false;
        group.add(m);
        markers.push(m);
        mats.push(mat);

        const el = document.createElement("div");
        el.className = "ss-vlabel ss-vlabel--cal";
        el.textContent = String(i + 1);
        el.style.borderColor = calibrationPointColor(i);
        world.labelLayer.appendChild(el);
        labels.push(el);
      }
      if (markers.length > n) {
        discardTail(n);
      }
    };

    const syncMarkers = (
      points: [number, number, number][],
      isActive: boolean,
    ) => {
      if (!isActive) {
        ensureCount(0);
        return;
      }
      ensureCount(points.length);
      for (let i = 0; i < points.length; i++) {
        const p = points[i];
        markers[i].position.set(p[0], p[1], p[2]);
        mats[i].color.set(calibrationPointColorHex(i));
        const el = labels[i];
        const text = String(i + 1);
        if (el.textContent !== text) {
          el.textContent = text;
        }
        el.style.borderColor = calibrationPointColor(i);
      }
    };

    const unsub = useCalibrateStore.subscribe((s, prev) => {
      if (s.mapPoints === prev.mapPoints && s.active === prev.active) {
        return;
      }
      syncMarkers(s.mapPoints, s.active);
    });
    {
      const s = useCalibrateStore.getState();
      syncMarkers(s.mapPoints, s.active);
    }

    const updateLabels = () => {
      if (markers.length === 0) {
        return;
      }
      const cam = world.getActiveCamera();
      const w = world.host.clientWidth;
      const h = world.host.clientHeight;
      for (let i = 0; i < markers.length; i++) {
        const el = labels[i];
        proj.copy(markers[i].position);
        proj.project(cam);
        if (proj.z > 1 || proj.z < -1) {
          el.style.display = "none";
          continue;
        }
        el.style.display = "";
        el.style.transform = `translate(-50%, -100%) translate(${(
          (proj.x * 0.5 + 0.5) * w
        )}px, ${(-proj.y * 0.5 + 0.5) * h - CALIBRATION_POINT_RADIUS_PX - 4}px)`;
      }
    };
    world.onFrame.add(updateLabels);

    return () => {
      unsub();
      world.onFrame.delete(updateLabels);
      discardTail(0);
      geo.dispose();
    };
  }, [world]);

  const feedImageUrl = useCalibrateStore((s) => s.feedImageUrl);
  const previewPose = useCalibrateStore((s) => s.previewPose);
  const opticsReady = useCalibrateStore((s) =>
    s.optics != null && s.optics.fx > 0 && s.optics.fy > 0,
  );

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

  // Warm the OpenCV worker as soon as calibrate opens (worker-only; no
  // main-thread load). importScripts is slow but must not freeze the UI.
  useEffect(() => {
    if (!active) {
      return;
    }
    preloadOpenCv();
  }, [active]);

  // Debounce solve in a Worker. At ≥6 pairs, refine fx/fy via REST first
  // (longer debounce to avoid refine storms while dragging).
  // Use an epoch so in-flight solves are not dropped by effect cleanup
  // (the old `cancelled` flag discarded valid results after every re-render).
  useEffect(() => {
    if (!active || !cameraId || !authToken || equalCount < 4 || !opticsReady) {
      // Invalidate any in-flight solve from a prior schedule.
      solveEpochRef.current += 1;
      useCalibrateStore.getState().setUpdatingPose(false);
      return;
    }
    const epoch = ++solveEpochRef.current;
    const scheduledKey = pairsKey;
    const debounceMs = equalCount >= 6 ? 300 : 180;
    useCalibrateStore.getState().setUpdatingPose(true);
    const timer = window.setTimeout(() => {
      void (async () => {
        try {
          if (epoch !== solveEpochRef.current) {
            return;
          }
          const pairs = getCalibratePairs();
          let working = useCalibrateStore.getState().optics;
          if (
            pairs.length < 4 ||
            !working ||
            !(working.fx > 0) ||
            !(working.fy > 0)
          ) {
            return;
          }
          if (pairs.length >= 6) {
            try {
              const refined = await refineIntrinsics(
                authToken,
                pairs,
                working,
              );
              if (epoch !== solveEpochRef.current) {
                return;
              }
              if (refined) {
                working = refined;
                useCalibrateStore.getState().setOptics(refined);
              }
            } catch (err) {
              console.warn("Calibrate intrinsics refine failed", err);
            }
          }
          if (epoch !== solveEpochRef.current) {
            return;
          }
          const pose = await solveCameraPose(pairs, working);
          // Apply when this epoch is still latest AND points still match
          // what we solved — do not use effect-cleanup cancelled flags.
          if (epoch !== solveEpochRef.current) {
            return;
          }
          if (!pose) {
            console.warn("Calibrate pose solve produced no result");
            return;
          }
          const liveKey =
            pairs.length >= 4
              ? pairs
                  .map(
                    (p) =>
                      `${p.cam[0]},${p.cam[1]}:${p.map[0]},${p.map[1]},${p.map[2]}`,
                  )
                  .join("|")
              : "";
          // Prefer live pairs; fall back to scheduledKey if store zip differs
          // only by float formatting noise (should be identical).
          if (liveKey && liveKey !== scheduledKey) {
            return;
          }
          const fov = fovFromOptics(working);
          const withFov = { ...pose, fov };
          useCalibrateStore.getState().setPreviewPose(withFov);
          getViewportState().updateEntity(cameraId, {
            position: withFov.position,
            rotation: restPoseToRigEuler(withFov.rotation),
            restRotation: withFov.rotation,
            posed: true,
            fov,
          });
          getViewportState().setProjectCamera(cameraId);
        } catch (err) {
          console.warn("Calibrate pose solve failed", err);
        } finally {
          // Always clear when this epoch is still current. A hanging worker
          // solve is bounded by solveCameraPose's timeout so we still get here.
          if (epoch === solveEpochRef.current) {
            useCalibrateStore.getState().setUpdatingPose(false);
          }
        }
      })();
    }, debounceMs);
    return () => {
      // Clearing the timer prevents a stale debounce from starting; in-flight
      // work is gated by epoch === solveEpochRef.current.
      window.clearTimeout(timer);
      // If debounce never started, drop the Updating… flag unless a newer
      // schedule has already claimed the epoch and set it again.
      if (epoch === solveEpochRef.current) {
        useCalibrateStore.getState().setUpdatingPose(false);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- pairsKey stands in for points
  }, [active, cameraId, authToken, pairsKey, equalCount, opticsReady]);

  // Pose on rAF; JPEG debounced — avoid stacking project() + decode on enter.
  useEffect(() => {
    if (!active || !world?.projectionPool || !previewPose || !cameraId) {
      return;
    }
    let cancelled = false;
    const handle = world.projectionPool.acquire(cameraId);
    world.cameraProjection = handle;
    const pose = {
      fov: previewPose.fov,
      position: previewPose.position,
      rotationDeg: restPoseToRigEuler(previewPose.rotation),
    };
    const raf = window.requestAnimationFrame(() => {
      if (cancelled) {
        return;
      }
      handle.updatePose(pose);
      handle.setOpacity(effectiveProjectOpacity(getViewportState(), cameraId));
      handle.setVisible(true);
    });
    let feedTimer: number | undefined;
    if (feedImageUrl) {
      feedTimer = window.setTimeout(() => {
        if (cancelled) {
          return;
        }
        handle
          .updateFrame(feedImageUrl, { ...pose, reproject: false })
          .then(() => {
            if (!cancelled) {
              handle.setOpacity(
                effectiveProjectOpacity(getViewportState(), cameraId),
              );
              handle.setVisible(true);
            }
          })
          .catch((err) => console.warn("Map projection update failed", err));
      }, 200);
    }
    return () => {
      cancelled = true;
      window.cancelAnimationFrame(raf);
      if (feedTimer != null) {
        window.clearTimeout(feedTimer);
      }
    };
  }, [active, world, previewPose, feedImageUrl, cameraId]);

  // Keep opacity slider live while calibrate owns the projector.
  const projectOpacity = useViewportStore((s) =>
    cameraId ? effectiveProjectOpacity(s, cameraId) : 0.8,
  );
  useEffect(() => {
    if (!active || !world?.projectionPool || !cameraId) {
      return;
    }
    const handle = world.projectionPool.get(cameraId) ?? world.cameraProjection;
    handle?.setOpacity(projectOpacity);
  }, [active, world, projectOpacity, cameraId]);

  useEffect(() => {
    if (!active) {
      useCalibrateStore.getState().setPreviewPose(null);
      // Leave coverage up when Project frame is still enabled for this camera.
      const stillOn =
        cameraId != null &&
        getViewportState().projectCameraIds.includes(cameraId);
      if (!stillOn && cameraId) {
        world?.projectionPool?.get(cameraId)?.setVisible(false);
      }
      if (world) {
        world.cameraProjection =
          stillOn && cameraId
            ? (world.projectionPool?.get(cameraId) ?? null)
            : null;
      }
    }
  }, [active, world, cameraId]);
}
