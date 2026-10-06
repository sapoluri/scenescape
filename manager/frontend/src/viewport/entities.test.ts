// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Euler, Object3D, Quaternion, Vector3 } from "three";
import { describe, expect, it } from "vitest";
import { lookAtOriginEuler, restPoseToRigEuler } from "./entities";

const D2R = Math.PI / 180;

function quatFromDeg([x, y, z]: [number, number, number]): Quaternion {
  return new Quaternion().setFromEuler(new Euler(x * D2R, y * D2R, z * D2R, "XYZ"));
}

/** Replicates the legacy scenecamera.js orientation exactly:
 *  rotation.copy(restEuler); rotateY(PI); rotateZ(PI); */
function legacyCameraQuat(restDeg: [number, number, number]): Quaternion {
  const o = new Object3D();
  o.rotation.set(restDeg[0] * D2R, restDeg[1] * D2R, restDeg[2] * D2R, "XYZ");
  o.rotateY(Math.PI);
  o.rotateZ(Math.PI);
  return o.quaternion.clone();
}

describe("restPoseToRigEuler", () => {
  const cases: [number, number, number][] = [
    [0, 0, 0],
    [30, 0, 0],
    [0, -15, 0],
    [0, 0, 90],
    [12.5, -33.7, 178.2],
    [-90, 45, 10],
  ];
  it.each(cases)(
    "matches the legacy rotateY(pi)+rotateZ(pi) orientation for rest %j",
    (rx, ry, rz) => {
      const rest: [number, number, number] = [rx, ry, rz];
      const mine = quatFromDeg(restPoseToRigEuler(rest));
      const legacy = legacyCameraQuat(rest);
      // Quaternions q and -q encode the same rotation.
      const dot = Math.abs(mine.dot(legacy));
      expect(dot).toBeGreaterThan(1 - 1e-9);
    },
  );

  it("keeps the optical axis on +Z at REST identity (scenescape convention)", () => {
    const q = quatFromDeg(restPoseToRigEuler([0, 0, 0]));
    const lensDir = new Vector3(0, 0, -1).applyQuaternion(q);
    // REST identity = optical axis +Z world; the rig flip preserves it so the
    // CameraHelper frustum matches the legacy 3D view exactly.
    expect(lensDir.z).toBeCloseTo(1, 9);
    expect(lensDir.x).toBeCloseTo(0, 9);
    expect(lensDir.y).toBeCloseTo(0, 9);
  });
});

describe("lookAtOriginEuler", () => {
  it("aims the -Z lens axis from the position toward the origin", () => {
    const positions: [number, number, number][] = [
      [15, 0, 3],
      [0, -12, 5],
      [10, 10, 8],
    ];
    for (const p of positions) {
      const q = quatFromDeg(lookAtOriginEuler(p));
      const lensDir = new Vector3(0, 0, -1).applyQuaternion(q);
      const toOrigin = new Vector3(-p[0], -p[1], -p[2]).normalize();
      // Mostly horizontal aim with a downward component (z + 2.5 offset).
      expect(lensDir.dot(toOrigin)).toBeGreaterThan(0.9);
      expect(lensDir.z).toBeLessThan(0);
    }
  });
});
