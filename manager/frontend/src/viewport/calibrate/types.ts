// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/** One 2D (camera image px) ↔ 3D (scene meters, Z-up) correspondence. */
export type CalibratePair = {
  cam: [number, number];
  map: [number, number, number];
};

export type CalibrateStep = "select" | "pick" | "save";

export type CalibratePending = "cam" | "scene" | null;

export type CameraOptics = {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  k1: number;
  k2: number;
  p1: number;
  p2: number;
  k3: number;
  width: number;
  height: number;
  name: string;
};

export const TRANSFORM_TYPE_POINT =
  "3d-2d point correspondence" as const;

export function pairsToTransforms(pairs: CalibratePair[]): number[] {
  const cam = pairs.flatMap((p) => [p.cam[0], p.cam[1]]);
  const map = pairs.flatMap((p) => [p.map[0], p.map[1], p.map[2]]);
  return [...cam, ...map];
}
