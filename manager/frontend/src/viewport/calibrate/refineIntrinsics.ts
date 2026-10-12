// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fovFromFy } from "../../lib/cameraOptics";
import { api } from "../../lib/rest";
import type { CalibratePair, CameraOptics } from "./types";

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

/**
 * Refine fx/fy via Manager `/api/v1/calculateintrinsics` when ≥6 point
 * pairs are available (legacy cameracalibrate.js behaviour).
 */
export async function refineIntrinsics(
  authToken: string,
  pairs: CalibratePair[],
  optics: CameraOptics,
): Promise<CameraOptics | null> {
  if (pairs.length < 6 || !authToken) {
    return null;
  }
  const resp = await api.calculateIntrinsics(authToken, {
    camPoints: pairs.map((p) => [p.cam[0], p.cam[1]]),
    mapPoints: pairs.map((p) => [p.map[0], p.map[1], p.map[2]]),
    // false = unlocked (solve). API treats missing keys as fixed.
    fixIntrinsics: {
      fx: false,
      fy: false,
      cx: true,
      cy: true,
      k1: true,
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
  const fx = Number(mtx[0]?.[0]);
  const fy = Number(mtx[1]?.[1]);
  if (![fx, fy].every((v) => Number.isFinite(v) && v > 0)) {
    return null;
  }
  return {
    ...optics,
    fx: round2(fx),
    fy: round2(fy),
  };
}

export function fovFromOptics(optics: CameraOptics): number {
  return round2(
    Math.min(120, Math.max(10, fovFromFy(optics.fy, optics.height))),
  );
}
