// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
  BoxGeometry,
  ExtrudeGeometry,
  FrontSide,
  ConeGeometry,
  Euler,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Object3D,
  PerspectiveCamera,
  Quaternion,
  Vector3,
} from "three";
import { describe, expect, it } from "vitest";
import { CHILD_FOOTPRINT_COLOR, CHILD_FOOTPRINT_SIZE } from "./childPose";
import {
  buildCamera,
  buildChild,
  buildRegion,
  buildTripwire,
  childViewportId,
  entityLabelAnchor,
  lookAtOriginEuler,
  restPoseToRigEuler,
  TRIPWIRE_CURTAIN_HEIGHT,
  tripwirePlusNormal,
} from "./entities";
import type {
  CameraEntity,
  ChildSceneEntity,
  RegionEntity,
  TripwireEntity,
} from "./types";

const D2R = Math.PI / 180;

function quatFromDeg([x, y, z]: [number, number, number]): Quaternion {
  return new Quaternion().setFromEuler(
    new Euler(x * D2R, y * D2R, z * D2R, "XYZ"),
  );
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
      restRotation: [15, -20, 45],
      posed: true,
      fov: 60,
      color: "#0a84ff",
    };
  }

  it("renders the CameraHelper frustum at the camera world transform (no double transform)", () => {
    const nodes = buildCamera(makeCamEntity());
    try {
      nodes.group.updateMatrixWorld(true);
      const helper = nodes.group.getObjectByProperty("type", "CameraHelper");
      const persp = nodes.group.getObjectByProperty(
        "type",
        "PerspectiveCamera",
      );
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
      const persp = nodes.group.getObjectByProperty(
        "type",
        "PerspectiveCamera",
      )!;
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

  it("draws a unit frustum and a half-size camera body", () => {
    const nodes = buildCamera(makeCamEntity());
    try {
      nodes.group.updateMatrixWorld(true);
      const persp = nodes.group.getObjectByProperty(
        "type",
        "PerspectiveCamera",
      ) as PerspectiveCamera;
      // Legacy CustomCameraHelper: visual near 0.2 m, far 1 m.
      expect(persp.near).toBeCloseTo(0.2);
      expect(persp.far).toBeCloseTo(1);
      const asset = nodes.gizmoTarget.getObjectByName("camera-asset");
      expect(asset).toBeTruthy();
      expect(asset!.scale.x).toBeCloseTo(0.5);
      expect(asset!.scale.y).toBeCloseTo(0.5);
      expect(asset!.scale.z).toBeCloseTo(0.5);
      expect(
        asset!.getObjectByProperty("type", "CameraHelper"),
      ).toBeUndefined();
    } finally {
      nodes.dispose();
    }
  });
});

describe("childViewportId", () => {
  it("uses the child scene uuid so a camera pk cannot replace it", () => {
    const cameraId = "1";
    const id = childViewportId({
      id: cameraId,
      restUid: "17efd356-8858-4bf7-80fa-5bc58b2175f8",
    });
    expect(id).toBe("17efd356-8858-4bf7-80fa-5bc58b2175f8");
    expect(id).not.toBe(cameraId);
  });

  it("falls back to the link id when no uuid is assigned", () => {
    expect(childViewportId({ id: "9", restUid: "  " })).toBe("9");
    expect(childViewportId({ id: "9", restUid: null })).toBe("9");
  });
});

describe("buildChild footprint", () => {
  function makeChild(): ChildSceneEntity {
    return {
      id: "child1",
      type: "child",
      name: "Annex",
      visible: true,
      position: [2, 3, 0.5],
      rotation: [0, 0, 45],
      scale: [1, 1, 1],
      size: CHILD_FOOTPRINT_SIZE,
      color: CHILD_FOOTPRINT_COLOR,
      childType: "local",
      restUid: "scene-uid",
      remoteChildId: null,
      detailUrl: "/scene-uid/",
      editHref: "?ss=child-edit&id=scene-uid",
      transformSource: "visual",
      deleteUrl: "/child/delete/1/",
    };
  }

  it("applies pose translation and entityId for picking", () => {
    const nodes = buildChild(makeChild());
    try {
      expect(nodes.group.userData.entityId).toBe("child1");
      expect(nodes.group.position.x).toBeCloseTo(2, 9);
      expect(nodes.group.position.y).toBeCloseTo(3, 9);
      expect(nodes.group.position.z).toBeCloseTo(0.5, 9);
      expect(nodes.group.rotation.z).toBeCloseTo(Math.PI / 4, 9);
    } finally {
      nodes.dispose();
    }
  });
});

describe("tripwirePlusNormal", () => {
  it("is the right-hand normal of the directed segment", () => {
    const horizontal = tripwirePlusNormal([0, 0], [10, 0]);
    expect(horizontal).not.toBeNull();
    expect(horizontal![0]).toBeCloseTo(0, 6);
    expect(horizontal![1]).toBeCloseTo(-1, 6);
    const vertical = tripwirePlusNormal([5, 0], [5, 10]);
    expect(vertical![0]).toBeCloseTo(1, 6);
    expect(vertical![1]).toBeCloseTo(0, 6);
  });

  it("returns null for a zero-length segment", () => {
    expect(tripwirePlusNormal([1, 2], [1, 2])).toBeNull();
  });
});

describe("buildRegion", () => {
  it("renders ROI geometry at its configured height independent of volumetric", () => {
    const region: RegionEntity = {
      id: "roi-height",
      type: "region",
      name: "Height check",
      visible: true,
      points: [
        [0, 0],
        [4, 0],
        [4, 3],
      ],
      min: [0, 0],
      max: [4, 3],
      height: 2.75,
      color: "#30d158",
      volumetric: false,
      bufferSize: 0,
    };
    const nodes = buildRegion(region);
    try {
      const fill = nodes.gizmoTarget.children.find(
        (node): node is Mesh =>
          node instanceof Mesh && node.geometry instanceof ExtrudeGeometry,
      );
      expect(fill).toBeDefined();
      fill!.geometry.computeBoundingBox();
      expect(fill!.geometry.boundingBox!.max.z).toBeCloseTo(region.height);
      const material = fill!.material as MeshStandardMaterial;
      expect(material.side).toBe(FrontSide);
      expect(material.depthWrite).toBe(true);
    } finally {
      nodes.dispose();
    }
  });
});

describe("entityLabelAnchor", () => {
  it("anchors labels to the top center of world-space bounds", () => {
    const root = new Object3D();
    const geometry = new BoxGeometry(2, 4, 6);
    const material = new MeshBasicMaterial();
    const mesh = new Mesh(geometry, material);
    mesh.position.set(4, -2, 3);
    root.add(mesh);
    const proxyGeometry = new BoxGeometry(10, 10, 20);
    const proxyMaterial = new MeshBasicMaterial({ colorWrite: false });
    const proxy = new Mesh(proxyGeometry, proxyMaterial);
    proxy.position.z = 10;
    root.add(proxy);
    try {
      expect(entityLabelAnchor(root, new Vector3()).toArray()).toEqual([
        4, -2, 6,
      ]);
    } finally {
      geometry.dispose();
      material.dispose();
      proxyGeometry.dispose();
      proxyMaterial.dispose();
    }
  });
});

describe("buildTripwire light curtain", () => {
  function wire(points: [number, number][]): TripwireEntity {
    return {
      id: "tw",
      type: "tripwire",
      name: "Checkout",
      visible: true,
      points,
      a: points[0],
      b: points[points.length - 1],
      color: "#ff9f0a",
    };
  }

  function plusArrows(root: Object3D): Object3D[] {
    const found: Object3D[] = [];
    root.traverse((o) => {
      if (o.name === "tripwire-plus") {
        found.push(o);
      }
    });
    return found;
  }

  it("raises a curtain and aims one +1 arrow to the right of the wire", () => {
    const nodes = buildTripwire(
      wire([
        [0, 0],
        [10, 0],
      ]),
    );
    try {
      const curtain = nodes.group.getObjectByName("tripwire-curtain") as Mesh;
      expect(curtain).toBeTruthy();
      expect(nodes.group.getObjectByName("tripwire-beams")).toBeTruthy();
      curtain.geometry.computeBoundingBox();
      expect(curtain.geometry.boundingBox!.max.z).toBeCloseTo(
        TRIPWIRE_CURTAIN_HEIGHT,
        5,
      );
      const arrows = plusArrows(nodes.group);
      expect(arrows).toHaveLength(1);
      const axis = new Vector3(0, 1, 0).applyQuaternion(arrows[0].quaternion);
      expect(axis.x).toBeCloseTo(0, 5);
      expect(axis.y).toBeCloseTo(-1, 5);
      expect(axis.z).toBeCloseTo(0, 5);
      expect(arrows[0].position.y).toBeLessThan(0);
    } finally {
      nodes.dispose();
    }
  });

  it("aims the +1 arrow to +X on an upward wire", () => {
    const nodes = buildTripwire(
      wire([
        [5, 0],
        [5, 10],
      ]),
    );
    try {
      const arrow = nodes.group.getObjectByName("tripwire-plus");
      expect(arrow).toBeTruthy();
      const axis = new Vector3(0, 1, 0).applyQuaternion(arrow!.quaternion);
      expect(axis.x).toBeCloseTo(1, 5);
      expect(axis.y).toBeCloseTo(0, 5);
      expect(arrow!.position.x).toBeGreaterThan(0);
    } finally {
      nodes.dispose();
    }
  });

  it("uses the tripwire color and exposes +1 as an entity label", () => {
    const nodes = buildTripwire(
      wire([
        [0, 0],
        [10, 0],
      ]),
    );
    try {
      const arrow = nodes.group.getObjectByName("tripwire-plus") as Mesh;
      expect((arrow.material as MeshBasicMaterial).color.getHex()).toBe(
        0xff9f0a,
      );
      expect(nodes.extraLabels).toHaveLength(1);
      expect(nodes.extraLabels?.[0]).toMatchObject({
        key: "plus",
        text: "+1",
        color: "#ff9f0a",
      });
      expect((arrow.geometry as ConeGeometry).parameters.radius).toBeCloseTo(
        0.054,
      );
      expect((arrow.geometry as ConeGeometry).parameters.height).toBeCloseTo(
        0.24,
      );
      expect(nodes.extraLabels?.[0].position.x).toBeCloseTo(5);
      expect(nodes.extraLabels?.[0].position.y).toBeCloseTo(-0.4);
      expect(nodes.extraLabels?.[0].position.z).toBeCloseTo(1.1);
    } finally {
      nodes.dispose();
    }
  });
});
