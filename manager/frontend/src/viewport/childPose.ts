// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Parse ChildScene REST / form transform payloads into a Z-up TRS pose for
 * dashed footprint meshes in the parent scene.
 */

export type ChildPose = {
  position: [number, number, number];
  rotation: [number, number, number];
  scale: [number, number, number];
};

/** Default map footprint in meters when child map extents are unknown. */
export const CHILD_FOOTPRINT_SIZE: [number, number, number] = [10, 10, 0.15];

/** Design token for linked child footprints in the 3D viewport. */
export const CHILD_FOOTPRINT_COLOR = "#ffd60a";

function finiteNumber(v: unknown, fallback: number): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function triple(
  arr: unknown,
  fallback: [number, number, number],
): [number, number, number] {
  if (Array.isArray(arr) && arr.length >= 3) {
    return [
      finiteNumber(arr[0], fallback[0]),
      finiteNumber(arr[1], fallback[1]),
      finiteNumber(arr[2], fallback[2]),
    ];
  }
  return fallback;
}

/**
 * Accept either nested `transform: { translation, rotation, scale }` or the
 * flat transform1..9 euler layout used by ChildSheet save payloads.
 */
export function parseChildPose(row: Record<string, unknown>): ChildPose {
  const nested =
    row.transform && typeof row.transform === "object"
      ? (row.transform as Record<string, unknown>)
      : null;
  if (nested && (nested.translation != null || nested.rotation != null)) {
    return {
      position: triple(nested.translation, [0, 0, 0]),
      rotation: triple(nested.rotation, [0, 0, 0]),
      scale: triple(nested.scale, [1, 1, 1]),
    };
  }
  return {
    position: [
      finiteNumber(row.transform1, 0),
      finiteNumber(row.transform2, 0),
      finiteNumber(row.transform3, 0),
    ],
    rotation: [
      finiteNumber(row.transform4, 0),
      finiteNumber(row.transform5, 0),
      finiteNumber(row.transform6, 0),
    ],
    scale: [
      finiteNumber(row.transform7, 1),
      finiteNumber(row.transform8, 1),
      finiteNumber(row.transform9, 1),
    ],
  };
}

/**
 * Apply child pose TRS to a three.js Object3D-like target (for unit tests and
 * builders). Degrees → radians for rotation.
 */
export function applyChildPoseToObject(
  object: {
    position: { set: (x: number, y: number, z: number) => void };
    rotation: { set: (x: number, y: number, z: number) => void };
    scale: { set: (x: number, y: number, z: number) => void };
  },
  pose: ChildPose,
): void {
  const d2r = Math.PI / 180;
  object.position.set(pose.position[0], pose.position[1], pose.position[2]);
  object.rotation.set(
    pose.rotation[0] * d2r,
    pose.rotation[1] * d2r,
    pose.rotation[2] * d2r,
  );
  object.scale.set(pose.scale[0], pose.scale[1], pose.scale[2]);
}
