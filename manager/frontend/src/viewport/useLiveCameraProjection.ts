// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useRef } from "react";
import { refreshCameraStrip } from "../mqtt";
import type { SceneCameraBootstrap } from "../scene/types";
import { useCalibrateStore } from "./calibrate/calibrateStore";
import { getViewportState, useViewportStore } from "./store";
import type { CameraEntity } from "./types";
import type { ViewportWorld } from "./world";

function frameSrcForSensor(sensorId: string): string | null {
  const imgs = document.querySelectorAll<HTMLImageElement>(
    `img[data-ss-card-sensor="${CSS.escape(sensorId)}"]`,
  );
  for (const img of imgs) {
    const src = img.currentSrc || img.getAttribute("src") || "";
    if (!src || src.includes("offline.png")) {
      continue;
    }
    // Accept a decoded frame even if still classed display-none (first paint).
    if (img.naturalWidth > 0 || src.startsWith("data:image")) {
      return src;
    }
  }
  return null;
}

type LiveCam = {
  id: string;
  sensorId: string;
  entity: CameraEntity;
  poseKey: string;
};

/**
 * Live map coverage when posed cameras have "Project frame" / "project all
 * frames" enabled (legacy scenecamera.js projectFrame). Calibrate owns the
 * projector for its camera while active; this hook drives every other id.
 */
export function useLiveCameraProjection(
  world: ViewportWorld | null,
  cameras: SceneCameraBootstrap[],
): void {
  const projectCameraIds = useViewportStore((s) => s.projectCameraIds);
  const projectOpacity = useViewportStore((s) => s.projectOpacity);
  const calibrating = useCalibrateStore((s) => s.active);
  const calibrateCameraId = useCalibrateStore((s) => s.cameraId);
  const entities = useViewportStore((s) => s.entities);

  const liveCams = useMemo((): LiveCam[] => {
    const out: LiveCam[] = [];
    for (const id of projectCameraIds) {
      // Calibrate session drives this camera's projector directly.
      if (calibrating && id === calibrateCameraId) {
        continue;
      }
      const e = entities[id];
      if (!e || e.type !== "camera" || !e.posed) {
        continue;
      }
      const boot = cameras.find((c) => c.id === id);
      if (!boot) {
        continue;
      }
      out.push({
        id,
        sensorId: boot.sensorId,
        entity: e,
        poseKey: `${e.position.join(",")}|${e.rotation.join(",")}|${e.fov}`,
      });
    }
    return out;
  }, [
    projectCameraIds,
    entities,
    cameras,
    calibrating,
    calibrateCameraId,
  ]);

  const liveKey = liveCams.map((c) => `${c.id}:${c.poseKey}`).join(";");
  const waitToken = useRef(0);

  // Drop unposed cameras from the project set (not while seed is loading).
  useEffect(() => {
    if (calibrating) {
      return;
    }
    for (const id of projectCameraIds) {
      const e = entities[id];
      if (e?.type === "camera" && !e.posed) {
        getViewportState().setProjectCameraEnabled(id, false);
      }
    }
  }, [projectCameraIds, entities, calibrating]);

  // Keep pool handles matched to the live project set; poll strip JPEGs.
  useEffect(() => {
    if (!world) {
      return;
    }
    const pool = world.projectionPool;
    if (!pool) {
      return;
    }

    if (liveCams.length === 0) {
      if (!calibrating && projectCameraIds.length === 0) {
        pool.sync([]);
        world.cameraProjection = null;
      } else if (!calibrating) {
        // Project ids exist but none are live-driven (all unposed / missing).
        pool.sync(projectCameraIds);
      }
      return;
    }

    let cancelled = false;
    let poll = 0;
    let observer: MutationObserver | null = null;
    const myWait = ++waitToken.current;
    const lastSrc = new Map<string, string>();
    const lastPose = new Map<string, string>();

    const apply = (cam: LiveCam, src: string) => {
      if (cancelled || !src) {
        return;
      }
      const handle = pool.acquire(cam.id);
      const poseChanged = lastPose.get(cam.id) !== cam.poseKey;
      if (!poseChanged && lastSrc.get(cam.id) === src) {
        return;
      }
      const firstFrame = !lastSrc.has(cam.id);
      lastSrc.set(cam.id, src);
      lastPose.set(cam.id, cam.poseKey);
      handle
        .updateFrame(src, {
          fov: cam.entity.fov,
          position: cam.entity.position,
          rotationDeg: cam.entity.rotation,
          reproject: poseChanged || firstFrame,
        })
        .then(() => {
          if (!cancelled && world.projectionPool) {
            handle.setOpacity(getViewportState().projectOpacity);
            handle.setVisible(true);
          }
        })
        .catch((err) => console.warn("Live map projection failed", err));
    };

    const tick = () => {
      if (cancelled) {
        return;
      }
      for (const cam of liveCams) {
        const src = frameSrcForSensor(cam.sensorId);
        if (src) {
          apply(cam, src);
        }
      }
    };

    const arm = () => {
      if (cancelled || myWait !== waitToken.current) {
        return;
      }
      if (!world.projectionPool) {
        window.setTimeout(arm, 200);
        return;
      }
      const keep = new Set(projectCameraIds);
      if (calibrating && calibrateCameraId) {
        keep.add(calibrateCameraId);
      }
      pool.sync(keep);
      pool.setOpacityAll(getViewportState().projectOpacity);
      refreshCameraStrip();
      tick();

      const root =
        document.querySelector(".ss-camera-panel") ?? document.body;
      observer = new MutationObserver(tick);
      observer.observe(root, {
        subtree: true,
        attributes: true,
        attributeFilter: ["src", "class"],
      });
      poll = window.setInterval(tick, 400);
    };

    arm();

    return () => {
      cancelled = true;
      observer?.disconnect();
      if (poll) {
        window.clearInterval(poll);
      }
    };
    // liveKey stands in for liveCams pose/membership; projectCameraIds for sync.
  }, [
    world,
    liveKey,
    liveCams,
    projectCameraIds,
    calibrating,
    calibrateCameraId,
  ]);

  // Opacity-only updates without reloading textures.
  useEffect(() => {
    if (!world?.projectionPool) {
      return;
    }
    world.projectionPool.setOpacityAll(projectOpacity);
    if (world.cameraProjection) {
      world.cameraProjection.setOpacity(projectOpacity);
    }
  }, [world, projectOpacity]);
}
