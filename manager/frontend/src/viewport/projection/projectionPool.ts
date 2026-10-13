// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { BufferGeometry, Object3D } from "three";
import {
  createCameraProjection,
  extractMapProjectionGeometry,
  type CameraProjectionHandle,
} from "./projectOntoMap";

/**
 * One ProjectedMaterial mesh per camera so several live frames can cover the
 * map at once (legacy "project all frames").
 */
export type ProjectionPool = {
  acquire: (cameraId: string) => CameraProjectionHandle;
  release: (cameraId: string) => void;
  /** Drop handles whose ids are not in `keep`. */
  sync: (keep: Iterable<string>) => void;
  get: (cameraId: string) => CameraProjectionHandle | undefined;
  setOpacityAll: (opacity: number) => void;
  dispose: () => void;
};

export function createProjectionPool(
  mapRoot: Object3D,
  parent: Object3D,
): ProjectionPool {
  const template: BufferGeometry = extractMapProjectionGeometry(mapRoot);
  const handles = new Map<string, CameraProjectionHandle>();
  let nextSlot = 0;

  const acquire = (cameraId: string): CameraProjectionHandle => {
    const existing = handles.get(cameraId);
    if (existing) {
      return existing;
    }
    // Slight per-camera lift so overlapping projectors do not z-fight.
    const zLift = 0.002 + (nextSlot % 24) * 0.0004;
    nextSlot += 1;
    const handle = createCameraProjection(mapRoot, parent, {
      geometry: template,
      zLift,
    });
    handles.set(cameraId, handle);
    return handle;
  };

  const release = (cameraId: string) => {
    const handle = handles.get(cameraId);
    if (!handle) {
      return;
    }
    handle.dispose();
    handles.delete(cameraId);
  };

  return {
    acquire,
    release,
    sync: (keep) => {
      const want = new Set(keep);
      for (const id of [...handles.keys()]) {
        if (!want.has(id)) {
          release(id);
        }
      }
    },
    get: (cameraId) => handles.get(cameraId),
    setOpacityAll: (opacity) => {
      for (const handle of handles.values()) {
        handle.setOpacity(opacity);
      }
    },
    dispose: () => {
      for (const id of [...handles.keys()]) {
        release(id);
      }
      template.dispose();
    },
  };
}
