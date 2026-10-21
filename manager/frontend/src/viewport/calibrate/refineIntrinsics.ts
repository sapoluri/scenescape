// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { focalFromDiagonalFov, fovFromFy } from "../../lib/cameraOptics";
import { api } from "../../lib/rest";
import type { CalibratePair, CalibrationLocks, CameraOptics } from "./types";

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

export function calibrationOpticsForFrame(
  optics: CameraOptics,
  width: number,
  height: number,
  diagonalFov?: number | null,
): CameraOptics {
  const focal = diagonalFov == null ? optics.fy : focalFromDiagonalFov(diagonalFov, width / 2, height / 2);
  return {
    ...optics,
    fx: focal,
    fy: focal,
    width,
    height,
    cx: width / 2,
    cy: height / 2,
  };
}

/**
 * Refine fx/fy via Manager `/api/v1/calculateintrinsics` when ≥6 point
 * pairs are available (legacy cameracalibrate.js behaviour).
 */
export async function refineIntrinsics(
  authToken: string,
  pairs: CalibratePair[],
  optics: CameraOptics,
  locks: CalibrationLocks = { focal: true, k1: true },
): Promise<CameraOptics | null> {
  const estimateFocal = !locks.focal && pairs.length >= 6;
  const estimateK1 = !locks.k1 && pairs.length >= 8;
  if ((!estimateFocal && !estimateK1) || !authToken) {
    return null;
  }
  const resp = await api.calculateIntrinsics(authToken, {
    camPoints: pairs.map((p) => [p.cam[0], p.cam[1]]),
    mapPoints: pairs.map((p) => [p.map[0], p.map[1], p.map[2]]),
    // false = unlocked (solve). API treats missing keys as fixed.
    fixIntrinsics: {
      fx: !estimateFocal,
      fy: !estimateFocal,
      cx: true,
      cy: true,
      k1: !estimateK1,
      k2: true,
      k3: true,
      p1: true,
      p2: true,
    },
    intrinsics: [
      [optics.fx, 0, optics.cx],
      [0, optics.fy, optics.cy],
      [0, 0, 1],
    ],
    distortion: [optics.k1, optics.k2, optics.p1, optics.p2, optics.k3],
    imageSize: [optics.width, optics.height],
  });
  const mtx = resp?.mtx;
  if (!Array.isArray(mtx) || mtx.length < 2) {
    return null;
  }
  const focal = estimateFocal ? Number(mtx[1]?.[1]) : optics.fy;
  const distortion = Array.isArray(resp.dist) ? resp.dist.flat() : [];
  const k1 = estimateK1 ? Number(distortion[0]) : optics.k1;
  if (!(Number.isFinite(focal) && focal > 0) || !Number.isFinite(k1)) {
    return null;
  }
  return {
    ...optics,
    fx: estimateFocal ? round2(focal) : optics.fx,
    fy: estimateFocal ? round2(focal) : optics.fy,
    k1,
  };
}

export function fovFromOptics(optics: CameraOptics): number {
  return fovFromFy(optics.fy, 2 * optics.cy);
}
