// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { ViewPreset } from "./types";

/**
 * Blender-style navigation for the 3D viewport (Z-up world):
 * - Left-drag: orbit, Middle-drag: zoom (dolly), Right-drag: pan
 * - Wheel: zoom, Shift+wheel: pan
 * - Numpad 1/3/7: front/side/top, 5: persp/ortho toggle
 */
export function applyBlenderBindings(controls: OrbitControls): void {
  controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: THREE.MOUSE.DOLLY,
    RIGHT: THREE.MOUSE.PAN,
  };
  controls.touches = {
    ONE: THREE.TOUCH.ROTATE,
    TWO: THREE.TOUCH.DOLLY_PAN,
  };
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  // Keep the camera above the floor plane.
  controls.maxPolarAngle = Math.PI * 0.495;
}

/**
 * Position a Z-up camera for a view preset. `distance` scales with the scene
 * extent; `target` is the orbit target in [x, y, z] meters.
 */
export function applyViewPreset(
  camera: THREE.PerspectiveCamera | THREE.OrthographicCamera,
  controls: OrbitControls,
  preset: ViewPreset,
  target: THREE.Vector3,
  distance: number,
): void {
  const d = distance;
  switch (preset) {
    case "top":
      // Looking straight down -Z: up must not be parallel to the view dir.
      camera.up.set(0, 1, 0);
      camera.position.set(target.x, target.y, target.z + d);
      break;
    case "front":
      camera.up.set(0, 0, 1);
      camera.position.set(target.x, target.y - d, target.z);
      break;
    case "side":
      camera.up.set(0, 0, 1);
      camera.position.set(target.x + d, target.y, target.z);
      break;
    case "persp":
    default:
      camera.up.set(0, 0, 1);
      camera.position.set(
        target.x + d * 0.75,
        target.y - d * 0.85,
        target.z + d * 0.65,
      );
      break;
  }
  controls.target.copy(target);
  camera.lookAt(target);
  controls.update();
}

/** Restore the default Z-up camera orientation after the top preset. */
export function resetCameraUp(camera: THREE.Camera): void {
  camera.up.set(0, 0, 1);
}
