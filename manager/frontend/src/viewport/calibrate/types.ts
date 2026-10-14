// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/** One 2D (camera image px) ↔ 3D (scene meters, Z-up) correspondence. */
export type CalibratePair = {
  cam: [number, number];
  map: [number, number, number];
};

export type CalibrateStep = "select" | "pick" | "save";

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

/** Zip equal-index points into pairs (legacy cam/map lists by index). */
export function zipCalibratePairs(
  camPoints: [number, number][],
  mapPoints: [number, number, number][],
): CalibratePair[] {
  const n = Math.min(camPoints.length, mapPoints.length);
  const out: CalibratePair[] = [];
  for (let i = 0; i < n; i++) {
    out.push({ cam: camPoints[i], map: mapPoints[i] });
  }
  return out;
}

export function pairsToTransforms(pairs: CalibratePair[]): number[] {
  const cam = pairs.flatMap((p) => [p.cam[0], p.cam[1]]);
  const map = pairs.flatMap((p) => [p.map[0], p.map[1], p.map[2]]);
  return [...cam, ...map];
}

/**
 * Parse stored camera `transforms` back into cam/map point lists.
 * Mirrors legacy `addInitialCalibrationPoints` in cameracalibrate.js:
 * length % 5 === 0 → 2D cam + 3D map; length % 2 === 0 → 2D cam + 2D map (z=0).
 */
export function transformsToPairs(
  transforms: unknown,
  transformType?: string | null,
): {
  camPoints: [number, number][];
  mapPoints: [number, number, number][];
} | null {
  const kind = String(transformType ?? "").trim();
  if (kind && kind !== TRANSFORM_TYPE_POINT) {
    return null;
  }
  let raw: unknown[] = [];
  if (Array.isArray(transforms)) {
    raw = transforms;
  } else if (typeof transforms === "string" && transforms.trim()) {
    try {
      const parsed = JSON.parse(transforms) as unknown;
      raw = Array.isArray(parsed) ? parsed : transforms.split(",");
    } catch {
      raw = transforms.split(",");
    }
  } else {
    return null;
  }
  const values = raw.map((v) => Number(v)).filter((v) => Number.isFinite(v));
  if (values.length === 0) {
    return null;
  }
  const camPoints: [number, number][] = [];
  const mapPoints: [number, number, number][] = [];
  if (values.length % 5 === 0) {
    const split = (values.length / 5) * 2;
    for (let i = 0; i < split; i += 2) {
      camPoints.push([values[i], values[i + 1]]);
    }
    for (let i = split; i < values.length; i += 3) {
      mapPoints.push([values[i], values[i + 1], values[i + 2]]);
    }
  } else if (values.length % 2 === 0) {
    const split = values.length / 2;
    for (let i = 0; i < split; i += 2) {
      camPoints.push([values[i], values[i + 1]]);
    }
    for (let i = split; i < values.length; i += 2) {
      mapPoints.push([values[i], values[i + 1], 0]);
    }
  } else {
    return null;
  }
  if (camPoints.length === 0 && mapPoints.length === 0) {
    return null;
  }
  return { camPoints, mapPoints };
}
