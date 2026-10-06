// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Euler, Object3D, Quaternion, Vector3 } from "three";
import { describe, expect, it } from "vitest";
import { buildCamera, lookAtOriginEuler, restPoseToRigEuler } from "./entities";
import type { CameraEntity } from "./types";

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

describe("lookAtOriginEuler", () => {  it("aims the -Z lens axis from the position toward the origin", () => {
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

describe("buildCamera frustum alignment", () => {
  function makeCamEntity(): CameraEntity {
    return {
      id: "cam1",
      type: "camera",
      name: "Test cam",
      visible: true,
      position: [10, 5, 3],
      target: [0, 0, 0],
      rotation: [15, -20, 45],
      fov: 60,
      color: "#0a84ff",
    };
  }

  it("renders the CameraHelper frustum at the camera world transform (no double transform)", () => {
    const nodes = buildCamera(makeCamEntity());
    try {
      nodes.group.updateMatrixWorld(true);
      const helper = nodes.group.getObjectByProperty("type", "CameraHelper");
      const persp = nodes.group.getObjectByProperty("type", "PerspectiveCamera");
      expect(helper).toBeTruthy();
      expect(persp).toBeTruthy();
      // CameraHelper bakes the camera's matrixWorld into its own matrix; its
      // world matrix must equal the camera's — not the rig transform squared.
      const h = helper!.matrixWorld.elements;
      const p = persp!.matrixWorld.elements;
      for (let i = 0; i < 16; i++) {
        expect(h[i]).toBeCloseTo(p[i], 9);
      }
    } finally {
      nodes.dispose();
    }
  });

  it("aims the helper frustum along the rig -Z (lens) direction", () => {
    const nodes = buildCamera(makeCamEntity());
    try {
      nodes.group.updateMatrixWorld(true);
      const persp = nodes.group.getObjectByProperty("type", "PerspectiveCamera")!;
      // A point down the camera's local -Z must land along the rig's -Z axis.
      const rig = nodes.gizmoTarget;
      const rigDir = new Vector3(0, 0, -1).applyQuaternion(rig.quaternion);
      const camPoint = new Vector3(0, 0, -5).applyMatrix4(persp.matrixWorld);
      const camPos = new Vector3().setFromMatrixPosition(persp.matrixWorld);
      const frustumDir = camPoint.sub(camPos).normalize();
      expect(frustumDir.dot(rigDir)).toBeGreaterThan(1 - 1e-9);
    } finally {
      nodes.dispose();
    }
  });
});
