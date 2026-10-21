// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { CameraOptics } from "../viewport/calibrate/types";

function num(v: unknown, d: number): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
}

/** Parse REST camera row into editable optics (intrinsics + distortion). */
export function parseCameraOptics(
  row: Record<string, unknown>,
  fallbackName = "",
): CameraOptics {
  const intr = (row.intrinsics as Record<string, unknown>) || {};
  const dist = (row.distortion as Record<string, unknown>) || {};
  const res = (row.resolution as Record<string, unknown>) || {};
  const width = num(res.width, 1280);
  const height = num(res.height, 720);
  return {
    name: String(row.name || fallbackName || ""),
    fx: num(intr.fx, width),
    fy: num(intr.fy, width),
    cx: num(intr.cx, width / 2),
    cy: num(intr.cy, height / 2),
    k1: num(dist.k1, 0),
    k2: num(dist.k2, 0),
    p1: num(dist.p1, 0),
    p2: num(dist.p2, 0),
    k3: num(dist.k3, 0),
    width,
    height,
  };
}

export function opticsPersistKey(o: CameraOptics): string {
  return JSON.stringify({
    fx: o.fx,
    fy: o.fy,
    cx: o.cx,
    cy: o.cy,
    k1: o.k1,
    k2: o.k2,
    p1: o.p1,
    p2: o.p2,
    k3: o.k3,
    width: o.width,
    height: o.height,
  });
}

/** Vertical FOV (degrees) from fy and image height. */
export function fovFromFy(fy: number, height: number): number {
  if (!(fy > 0) || !(height > 0)) {
    return 60;
  }
  return (2 * Math.atan(height / (2 * fy)) * 180) / Math.PI;
}

/** fy from vertical FOV (degrees) and image height. */
export function fyFromFov(fovDeg: number, height: number): number {
  if (!(height > 0) || !(fovDeg > 0) || fovDeg >= 179) {
    return height;
  }
  return height / (2 * Math.tan((fovDeg * Math.PI) / 360));
}

export function diagonalFovFromOptics(
  optics: Pick<CameraOptics, "cx" | "cy" | "fx" | "fy">,
): number {
  return 2 * Math.atan(Math.hypot(optics.cx / optics.fx, optics.cy / optics.fy)) * 180 / Math.PI;
}

export function focalFromDiagonalFov(angle: number, cx: number, cy: number): number {
  return Math.hypot(cx, cy) / Math.tan(angle * Math.PI / 360);
}
