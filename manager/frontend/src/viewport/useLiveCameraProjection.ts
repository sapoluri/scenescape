// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef } from "react";
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

/**
 * Live map coverage when a posed camera has "Project frame" enabled
 * (legacy scenecamera.js projectFrame). Calibrate mode owns the projector
 * while active; this hook resumes when calibrate exits.
 */
export function useLiveCameraProjection(
  world: ViewportWorld | null,
  cameras: SceneCameraBootstrap[],
): void {
  const projectCameraId = useViewportStore((s) => s.projectCameraId);
  const projectOpacity = useViewportStore((s) => s.projectOpacity);
  const calibrating = useCalibrateStore((s) => s.active);
  const camEntity = useViewportStore((s) => {
    const id = s.projectCameraId;
    if (!id) {
      return null;
    }
    const e = s.entities[id];
    return e?.type === "camera" ? (e as CameraEntity) : null;
  });
  const poseKey = camEntity?.posed
    ? `${camEntity.position.join(",")}|${camEntity.rotation.join(",")}|${camEntity.fov}`
    : "";
  const waitToken = useRef(0);

  // Drop projection only when the camera is known and unposed (not while seed
  // is still loading — that would make Project appear to do nothing).
  // Calibrate owns projection for its camera even before the first solve.
  useEffect(() => {
    if (!projectCameraId || !camEntity || calibrating) {
      return;
    }
    if (!camEntity.posed) {
      getViewportState().setProjectCamera(null);
    }
  }, [projectCameraId, camEntity, calibrating]);

  // Wait until the map projector exists (async map load), then keep applying
  // the latest strip JPEG for the selected posed camera.
  useEffect(() => {
    if (calibrating || !world || !projectCameraId || !camEntity?.posed) {
      if (!calibrating && !projectCameraId) {
        world?.cameraProjection?.setVisible(false);
      }
      return;
    }

    const boot = cameras.find((c) => c.id === projectCameraId);
    if (!boot) {
      return;
    }

    let cancelled = false;
    let lastSrc = "";
    let poll = 0;
    let observer: MutationObserver | null = null;
    const myWait = ++waitToken.current;

    const apply = (src: string) => {
      const handle = world.cameraProjection;
      if (cancelled || !handle || !src || src === lastSrc) {
        return;
      }
      lastSrc = src;
      handle
        .updateFrame(src, {
          fov: camEntity.fov,
          position: camEntity.position,
          // Rig euler matches the frustum PerspectiveCamera (REST × Rx(π)).
          rotationDeg: camEntity.rotation,
        })
        .then(() => {
          if (!cancelled && world.cameraProjection) {
            world.cameraProjection.setOpacity(
              getViewportState().projectOpacity,
            );
            world.cameraProjection.setVisible(true);
          }
        })
        .catch((err) => console.warn("Live map projection failed", err));
    };

    const tick = () => {
      if (cancelled) {
        return;
      }
      const src = frameSrcForSensor(boot.sensorId);
      if (src) {
        apply(src);
      }
    };

    const arm = () => {
      if (cancelled || myWait !== waitToken.current) {
        return;
      }
      if (!world.cameraProjection) {
        // Map / projector still loading — retry briefly.
        window.setTimeout(arm, 200);
        return;
      }
      world.cameraProjection.setOpacity(getViewportState().projectOpacity);
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
  }, [
    world,
    cameras,
    calibrating,
    projectCameraId,
    camEntity?.posed,
    poseKey,
  ]);

  // Opacity-only updates without reloading the texture.
  // Calibrate session also listens; either path is fine while active.
  useEffect(() => {
    if (!world?.cameraProjection || !projectCameraId) {
      return;
    }
    world.cameraProjection.setOpacity(projectOpacity);
  }, [world, projectCameraId, projectOpacity]);
}
