// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from "react";
import {
  Box3,
  BoxGeometry,
  BufferAttribute,
  BufferGeometry,
  CameraHelper,
  CircleGeometry,
  Color,
  ConeGeometry,
  CylinderGeometry,
  DoubleSide,
  EdgesGeometry,
  Euler,
  ExtrudeGeometry,
  Group,
  LineBasicMaterial,
  LineLoop,
  LineSegments,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Object3D,
  PerspectiveCamera,
  Quaternion,
  Shape,
  ShapeGeometry,
  SphereGeometry,
  Vector2,
  Vector3,
} from "three";
import { api } from "../lib/rest";
import { loadSceneObject } from "../placement/loadSceneObject";
import { applyScenePoseToObject } from "../placement/poseThree";
import { sceneGeometryFromRest } from "../placement/sceneGeometry";
import { subscribeGeometry } from "../scene/map/geometryModel";
import type {
  SceneCameraBootstrap,
  SceneChildBootstrap,
  SceneSensorBootstrap,
} from "../scene/types";
import {
  applyChildPoseToObject,
  CHILD_FOOTPRINT_COLOR,
  CHILD_FOOTPRINT_SIZE,
  parseChildPose,
} from "./childPose";
import { bboxOf, centroidOf, seedEntitiesFromGeometry } from "./geometrySync";
import { getPersistState } from "./persistStore";
import { getViewportState, useViewportStore } from "./store";
import { useCalibrateStore } from "./calibrate/calibrateStore";
import type {
  CameraEntity,
  ChildSceneEntity,
  MapEntity,
  RegionEntity,
  SensorEntity,
  TripwireEntity,
  ViewportEntity,
} from "./types";
import type { ViewportWorld } from "./world";

/**
 * Phase 1.2 entity builders + reconciler.
 *
 * Z-up ports of the mockup's builders. The camera frustum follows the
 * legacy 3D view's tested recipe (scenescape3d.js / thing/scenecamera.js):
 * a PerspectiveCamera rig with q = qEuler(XYZ) * Rx(π), drawn with
 * CameraHelper — the camera looks down its local -Z.
 */

export interface EntityNodes {
  group: Group;
  labelRoot?: Object3D;
  ready?: Promise<void>;
  /** Object the TransformControls gizmo attaches to. */
  gizmoTarget: Object3D;
  /** World-space anchor for the name label. */
  labelPos: Vector3;
  extraLabels?: {
    key: string;
    text: string;
    position: Vector3;
    color?: string;
  }[];
  setSelected: (on: boolean) => void;
  setFrustumVisible?: (visible: boolean) => void;
  /**
   * Apply pose/optics from the store without disposing meshes. Used when
   * `poseVersion` bumps (calibrate preview, gizmo drag, properties).
   */
  applyPose?: (e: ViewportEntity) => void;
  dispose: () => void;
}

const d2r = (d: number) => (d * Math.PI) / 180;
const r2d = (r: number) => (r * 180) / Math.PI;

function disposeObject(root: Object3D): void {
  root.traverse((o) => {
    const mesh = o as Mesh;
    const geo = (mesh as { geometry?: { dispose?: () => void } }).geometry;
    geo?.dispose?.();
    const mat = mesh.material as
      { dispose?: () => void } | { dispose?: () => void }[] | undefined;
    const disposeMat = (m: {
      dispose?: () => void;
      map?: { dispose?: () => void };
    }) => {
      m.map?.dispose?.();
      m.dispose?.();
    };
    if (Array.isArray(mat)) {
      mat.forEach((m) => disposeMat(m));
    } else if (mat) {
      disposeMat(mat);
    }
  });
}

/** Invisible-but-raycastable selection proxy material (fresh per proxy). */
function proxyMaterial(): MeshBasicMaterial {
  return new MeshBasicMaterial({
    colorWrite: false,
    depthWrite: false,
    depthTest: false,
  });
}

export function entityLabelAnchor(
  root: Object3D,
  fallback: Vector3,
): Vector3 {
  root.updateWorldMatrix(true, true);
  const bounds = new Box3();
  root.traverse((object) => {
    if (!object.visible) {
      return;
    }
    const renderable = object as Mesh;
    if (!renderable.geometry) {
      return;
    }
    const materials = Array.isArray(renderable.material)
      ? renderable.material
      : [renderable.material];
    if (
      materials.length > 0 &&
      materials.every((material) => material.colorWrite === false)
    ) {
      return;
    }
    bounds.union(new Box3().setFromObject(object));
  });
  if (bounds.isEmpty()) {
    return fallback.clone();
  }
  return new Vector3(
    (bounds.min.x + bounds.max.x) / 2,
    (bounds.min.y + bounds.max.y) / 2,
    bounds.max.z,
  );
}

/* ------------------------------------------------------------------ */
/* Regions                                                             */
/** Fallback slab height for malformed ROI height data. */
const DEFAULT_REGION_HEIGHT = 0.15;

export function buildRegion(e: RegionEntity): EntityNodes {
  const group = new Group();
  const baseColor = new Color(e.color);
  const [cx, cy] = centroidOf(e.points);
  const pivot = new Group();
  pivot.position.set(cx, cy, 0);

  const shape = new Shape(
    e.points.map(([x, y]) => new Vector2(x - cx, y - cy)),
  );
  const solidHeight =
    Number.isFinite(e.height) && e.height > 0
      ? e.height
      : DEFAULT_REGION_HEIGHT;
  const idleOpacity = 0.22;
  const selectedOpacity = 0.42;
  // Thin extrusion (not ShapeGeometry) so flat ROIs keep side faces from oblique views.
  const geo = new ExtrudeGeometry(shape, {
    depth: solidHeight,
    bevelEnabled: false,
  });
  const fillMat = new MeshStandardMaterial({
    color: baseColor.clone(),
    emissive: baseColor.clone(),
    emissiveIntensity: 0.05,
    transparent: true,
    opacity: idleOpacity,
    roughness: 0.85,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
  });
  const fill = new Mesh(geo, fillMat);
  fill.renderOrder = 1;
  const edgeMat = new LineBasicMaterial({
    color: baseColor.clone(),
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
  });
  const edges = new LineSegments(new EdgesGeometry(geo), edgeMat);
  edges.renderOrder = 2;
  const { min, max } = bboxOf(e.points);
  const proxyH = Math.max(solidHeight, 0.5);
  const proxy = new Mesh(
    new BoxGeometry(
      Math.max(max[0] - min[0], 0.5),
      Math.max(max[1] - min[1], 0.5),
      proxyH,
    ),
    proxyMaterial(),
  );
  proxy.position.set(
    (min[0] + max[0]) / 2 - cx,
    (min[1] + max[1]) / 2 - cy,
    proxyH / 2,
  );
  pivot.add(fill, edges, proxy);
  group.add(pivot);
  group.userData.entityId = e.id;

  return {
    group,
    gizmoTarget: pivot,
    labelPos: new Vector3(cx, cy, solidHeight + 0.6),
    setSelected: (on: boolean) => {
      edgeMat.color.set(on ? "#ffffff" : e.color);
      fillMat.opacity = on ? selectedOpacity : idleOpacity;
      fillMat.emissiveIntensity = on
        ? 0.12
        : 0.05;
    },
    dispose: () => disposeObject(group),
  };
}

/* ------------------------------------------------------------------ */
/* Tripwires — light curtain                                           */
/* ------------------------------------------------------------------ */

/** Vertical sheet height in meters. */
export const TRIPWIRE_CURTAIN_HEIGHT = 2;

const TRIPWIRE_CURTAIN_FLOOR = 0.02;
const TRIPWIRE_PLUS_OFFSET = 0.28;
const TRIPWIRE_PLUS_ARROW_SCALE = 0.6;
const TRIPWIRE_CURTAIN_OPACITY = 0.26;
const TRIPWIRE_CURTAIN_OPACITY_SELECTED = 0.42;

/**
 * Unit normal of the +1 crossing side.
 *
 * Scene Controller counts a crossing +1 when the object's current point lies
 * to the right of the directed segment (see Tripwire.lineCrosses). Walking
 * from the first point toward the next, +1 is to the right: (dy, -dx) / len.
 */
export function tripwirePlusNormal(
  a: [number, number],
  b: [number, number],
): [number, number] | null {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy);
  if (len < 1e-6) {
    return null;
  }
  return [dy / len, -dx / len];
}

interface CurtainMaterials {
  curtain: MeshBasicMaterial;
  beams: LineBasicMaterial;
  posts: MeshBasicMaterial;
  arrow: MeshBasicMaterial;
  plusLabelPosition?: Vector3;
}

/** Point and +1 normal at half the polyline length. Skips zero-length spans. */
function curtainMidpoint(points: [number, number][]): {
  point: [number, number];
  normal: [number, number];
} | null {
  const segs: {
    a: [number, number];
    b: [number, number];
    len: number;
    normal: [number, number];
  }[] = [];
  let total = 0;
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i];
    const b = points[i + 1];
    const normal = tripwirePlusNormal(a, b);
    if (!normal) {
      continue;
    }
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    segs.push({ a, b, len, normal });
    total += len;
  }
  if (total < 1e-6 || segs.length === 0) {
    return null;
  }
  let remain = total / 2;
  for (const seg of segs) {
    if (remain <= seg.len + 1e-9) {
      const t = seg.len > 1e-9 ? remain / seg.len : 0;
      return {
        point: [
          seg.a[0] + (seg.b[0] - seg.a[0]) * t,
          seg.a[1] + (seg.b[1] - seg.a[1]) * t,
        ],
        normal: seg.normal,
      };
    }
    remain -= seg.len;
  }
  const last = segs[segs.length - 1];
  return { point: [...last.b], normal: last.normal };
}

/**
 * Vertical light curtain for a floor polyline, in the parent's local XY.
 * One small arrow sits on the +1 side at the path midpoint.
 */
export function addTripwireCurtain(
  parent: Object3D,
  points: [number, number][],
  color: string | number,
  options?: { showLabel?: boolean },
): CurtainMaterials {
  const base = new Color(color);
  const curtainMat = new MeshBasicMaterial({
    color: base.clone(),
    transparent: true,
    opacity: TRIPWIRE_CURTAIN_OPACITY,
    depthWrite: false,
    side: DoubleSide,
  });
  const beamMat = new LineBasicMaterial({ color: base.clone() });
  const postMat = new MeshBasicMaterial({ color: base.clone() });
  const arrowMat = new MeshBasicMaterial({ color: base.clone() });
  const z0 = TRIPWIRE_CURTAIN_FLOOR;
  const z1 = TRIPWIRE_CURTAIN_HEIGHT;

  const positions: number[] = [];
  const indices: number[] = [];
  const beams: number[] = [];
  const pushBeam = (x: number, y: number) => {
    beams.push(x, y, z0, x, y, z1);
  };
  let vertex = 0;
  let beamed = false;
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i];
    const b = points[i + 1];
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const len = Math.hypot(dx, dy);
    if (len < 1e-6) {
      continue;
    }
    positions.push(
      a[0],
      a[1],
      z0,
      b[0],
      b[1],
      z0,
      b[0],
      b[1],
      z1,
      a[0],
      a[1],
      z1,
    );
    indices.push(
      vertex,
      vertex + 1,
      vertex + 2,
      vertex,
      vertex + 2,
      vertex + 3,
    );
    vertex += 4;
    if (!beamed) {
      pushBeam(a[0], a[1]);
      beamed = true;
    }
    beams.push(a[0], a[1], z0, b[0], b[1], z0, a[0], a[1], z1, b[0], b[1], z1);
    pushBeam(b[0], b[1]);
  }

  if (positions.length > 0) {
    const geo = new BufferGeometry();
    geo.setAttribute(
      "position",
      new BufferAttribute(new Float32Array(positions), 3),
    );
    geo.setIndex(indices);
    const sheet = new Mesh(geo, curtainMat);
    sheet.name = "tripwire-curtain";
    sheet.renderOrder = 2;
    parent.add(sheet);
  }
  if (beams.length > 0) {
    const geo = new BufferGeometry();
    geo.setAttribute(
      "position",
      new BufferAttribute(new Float32Array(beams), 3),
    );
    const lines = new LineSegments(geo, beamMat);
    lines.name = "tripwire-beams";
    lines.renderOrder = 3;
    parent.add(lines);
  }

  const postGeo = new CylinderGeometry(0.04, 0.04, z1 - z0, 8);
  postGeo.rotateX(Math.PI / 2);
  const seen = new Set<string>();
  for (const [x, y] of points) {
    const key = `${x.toFixed(4)},${y.toFixed(4)}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    const post = new Mesh(postGeo, postMat);
    post.position.set(x, y, (z0 + z1) / 2);
    parent.add(post);
  }

  const mid = curtainMidpoint(points);
  let plusLabelPosition: Vector3 | undefined;
  if (mid) {
    const [nx, ny] = mid.normal;
    const cone = new Mesh(
      new ConeGeometry(
        0.09 * TRIPWIRE_PLUS_ARROW_SCALE,
        0.4 * TRIPWIRE_PLUS_ARROW_SCALE,
        10,
      ),
      arrowMat,
    );
    cone.name = "tripwire-plus";
    cone.quaternion.setFromUnitVectors(
      new Vector3(0, 1, 0),
      new Vector3(nx, ny, 0),
    );
    cone.position.set(
      mid.point[0] + nx * TRIPWIRE_PLUS_OFFSET,
      mid.point[1] + ny * TRIPWIRE_PLUS_OFFSET,
      z1 * 0.55,
    );
    cone.renderOrder = 4;
    parent.add(cone);

    if (options?.showLabel !== false) {
      const labelOffset =
        TRIPWIRE_PLUS_OFFSET + (0.4 * TRIPWIRE_PLUS_ARROW_SCALE) / 2;
      plusLabelPosition = new Vector3(
        mid.point[0] + nx * labelOffset,
        mid.point[1] + ny * labelOffset,
        z1 * 0.55,
      );
    }
  }

  return {
    curtain: curtainMat,
    beams: beamMat,
    posts: postMat,
    arrow: arrowMat,
    plusLabelPosition,
  };
}

export function buildTripwire(e: TripwireEntity): EntityNodes {
  const group = new Group();
  const [cx, cy] = centroidOf(e.points);
  const pivot = new Group();
  pivot.position.set(cx, cy, 0);
  const local = e.points.map(([x, y]) => [x - cx, y - cy] as [number, number]);
  const mats = addTripwireCurtain(pivot, local, e.color);

  const { min, max } = bboxOf(e.points);
  const proxy = new Mesh(
    new BoxGeometry(
      Math.max(max[0] - min[0], 0.5),
      Math.max(max[1] - min[1], 0.5),
      TRIPWIRE_CURTAIN_HEIGHT,
    ),
    proxyMaterial(),
  );
  proxy.position.set(
    (min[0] + max[0]) / 2 - cx,
    (min[1] + max[1]) / 2 - cy,
    TRIPWIRE_CURTAIN_HEIGHT / 2,
  );
  pivot.add(proxy);
  group.add(pivot);
  group.userData.entityId = e.id;
  const extraLabels = mats.plusLabelPosition
    ? [
        {
          key: "plus",
          text: "+1",
          position: mats.plusLabelPosition.clone().add(pivot.position),
          color: e.color,
        },
      ]
    : undefined;

  return {
    group,
    gizmoTarget: pivot,
    labelPos: new Vector3(cx, cy, TRIPWIRE_CURTAIN_HEIGHT + 0.35),
    extraLabels,
    setSelected: (on: boolean) => {
      mats.curtain.opacity = on
        ? TRIPWIRE_CURTAIN_OPACITY_SELECTED
        : TRIPWIRE_CURTAIN_OPACITY;
      const c = on ? "#ffffff" : e.color;
      mats.beams.color.set(c);
      mats.posts.color.set(c);
      mats.arrow.color.set(c);
    },
    dispose: () => disposeObject(group),
  };
}

/* ------------------------------------------------------------------ */
/* Cameras (legacy-tested frustum recipe)                              */
/* ------------------------------------------------------------------ */

/**
 * Convert REST calibration (translation + XYZ-degree euler) to the camera
 * rig's world euler. Mirrors thing/scenecamera.js: q = qEuler * Rx(π).
 */
export function restPoseToRigEuler(
  rotationDeg: [number, number, number],
): [number, number, number] {
  const q = new Quaternion().setFromEuler(
    new Euler(
      d2r(rotationDeg[0]),
      d2r(rotationDeg[1]),
      d2r(rotationDeg[2]),
      "XYZ",
    ),
  );
  q.multiply(new Quaternion().setFromAxisAngle(new Vector3(1, 0, 0), Math.PI));
  const e = new Euler().setFromQuaternion(q, "XYZ");
  return [r2d(e.x), r2d(e.y), r2d(e.z)];
}

/** Default rig euler that points the camera (-Z) at the scene origin. */
export function lookAtOriginEuler(
  position: [number, number, number],
): [number, number, number] {
  const o = new Object3D();
  o.position.set(position[0], position[1], position[2]);
  o.up.set(0, 0, 1);
  // +Z aims away from the origin so -Z (the lens axis) aims at it.
  o.lookAt(position[0] * 2, position[1] * 2, position[2] + 2.5);
  const e = new Euler().setFromQuaternion(o.quaternion, "XYZ");
  return [r2d(e.x), r2d(e.y), r2d(e.z)];
}

export function buildCamera(e: CameraEntity): EntityNodes {
  const group = new Group();
  const rig = new Group();
  rig.position.set(e.position[0], e.position[1], e.position[2]);
  rig.quaternion.setFromEuler(
    new Euler(
      d2r(e.rotation[0]),
      d2r(e.rotation[1]),
      d2r(e.rotation[2]),
      "XYZ",
    ),
  );

  // Frustum rig: a real PerspectiveCamera (never rendered) + CameraHelper,
  // exactly like the legacy 3D view. The lens looks down local -Z.
  // NOTE: CameraHelper replaces its own matrix with the camera's matrixWorld,
  // so it must live under the identity outer group — parenting it under the
  // transformed rig would apply the rig transform twice and the frustum
  // would not align with the camera body.
  // Visual size matches legacy CustomCameraHelper (near 0.2, far 1). This
  // camera is never rendered; the drawn frustum is one meter along -Z.
  const persp = new PerspectiveCamera(e.fov, 4 / 3, 0.2, 1);
  const helper = new CameraHelper(persp);
  rig.add(persp);

  const bodyMat = new MeshStandardMaterial({
    color: 0x2b2d36,
    roughness: 0.6,
    emissive: 0x0a84ff,
    emissiveIntensity: 0,
  });
  const body = new Mesh(new BoxGeometry(0.45, 0.45, 0.7), bodyMat);
  const lensGeo = new CylinderGeometry(0.12, 0.16, 0.3, 12);
  lensGeo.rotateX(Math.PI / 2); // axis Y -> Z
  const lens = new Mesh(
    lensGeo,
    new MeshStandardMaterial({ color: 0x0a84ff, roughness: 0.3 }),
  );
  lens.position.z = -0.45; // toward the -Z view direction
  // Body, lens, and pick proxy at half the original mesh size. The helper
  // stays on the unscaled rig so the unit frustum is not shrunk with them.
  const asset = new Group();
  asset.name = "camera-asset";
  asset.scale.setScalar(0.5);
  const proxy = new Mesh(new BoxGeometry(1.6, 1.6, 1.6), proxyMaterial());
  asset.add(body, lens, proxy);
  rig.add(asset);
  group.add(rig);
  // Added after the rig: CameraHelper reads the camera's matrixWorld during
  // the scene-graph update, so it must be traversed after the rig subtree
  // to see a fresh transform (no one-frame lag).
  group.add(helper);
  group.userData.entityId = e.id;

  const labelPos = new Vector3(
    e.position[0],
    e.position[1],
    e.position[2] + 0.7,
  );

  return {
    group,
    gizmoTarget: rig,
    labelPos,
    setSelected: (on: boolean) => {
      bodyMat.emissiveIntensity = on ? 0.9 : 0;
    },
    setFrustumVisible: (visible: boolean) => {
      helper.visible = visible;
    },
    applyPose: (ent) => {
      if (ent.type !== "camera") {
        return;
      }
      rig.position.set(ent.position[0], ent.position[1], ent.position[2]);
      rig.quaternion.setFromEuler(
        new Euler(
          d2r(ent.rotation[0]),
          d2r(ent.rotation[1]),
          d2r(ent.rotation[2]),
          "XYZ",
        ),
      );
      if (persp.fov !== ent.fov) {
        persp.fov = ent.fov;
        persp.updateProjectionMatrix();
        helper.update();
      }
      labelPos.set(ent.position[0], ent.position[1], ent.position[2] + 0.7);
    },
    dispose: () => {
      helper.dispose();
      disposeObject(group);
    },
  };
}

/* ------------------------------------------------------------------ */
/* Sensors                                                             */
/* ------------------------------------------------------------------ */

export function buildSensor(e: SensorEntity): EntityNodes {
  const group = new Group();
  const marker = new Group();
  marker.name = "sensor-marker";
  const base = new Color(e.color);
  const [x, y, z] = e.position;
  const polyPoints = e.areaMode === "poly" ? e.points : [];
  const showPoly = polyPoints.length >= 3;
  const radius = Math.max(e.radius, 0.5);
  const headMat = new MeshStandardMaterial({
    color: base.clone(),
    emissive: base.clone(),
    emissiveIntensity: 0.6,
    roughness: 0.4,
  });
  const head = new Mesh(new SphereGeometry(0.11, 16, 12), headMat);
  head.name = "sensor-sphere";
  head.position.z = 0.11;
  const proxy = new Mesh(new SphereGeometry(0.16, 12, 8), proxyMaterial());
  proxy.position.z = 0.11;
  marker.add(head, proxy);
  marker.position.set(x, y, z);

  const coverage = new Group();
  coverage.name = "sensor-coverage";
  if (showPoly) {
    const [cx, cy] = centroidOf(polyPoints);
    const shape = new Shape(
      polyPoints.map(([px, py]) => new Vector2(px - cx, py - cy)),
    );
    const geo = new ShapeGeometry(shape);
    const fill = new Mesh(
      geo,
      new MeshBasicMaterial({
        color: base.clone(),
        transparent: true,
        opacity: 0.14,
        side: DoubleSide,
        depthWrite: false,
      }),
    );
    fill.position.z = 0.05;
    const edges = new LineSegments(
      new EdgesGeometry(geo),
      new LineBasicMaterial({ color: base.clone() }),
    );
    edges.position.z = 0.06;
    coverage.position.set(cx, cy, 0);
    coverage.add(fill, edges);
  } else if (e.areaMode === "circle") {
    // RingGeometry/CircleGeometry already lie in the XY plane = Z-up floor.
    const ring = new LineLoop(
      new BufferGeometry().setFromPoints(Array.from({ length: 64 }, (_, index) => {
        const angle = index * Math.PI * 2 / 64;
        return new Vector3(Math.cos(angle) * radius, Math.sin(angle) * radius, 0);
      })),
      new LineBasicMaterial({
        color: base.clone(),
        depthWrite: false,
      }),
    );
    ring.name = "sensor-coverage-outline";
    ring.renderOrder = 2;
    ring.position.z = 0.06;
    const disc = new Mesh(
      new CircleGeometry(radius, 48),
      new MeshBasicMaterial({
        color: base.clone(),
        transparent: true,
        opacity: 0.06,
        side: DoubleSide,
        depthWrite: false,
      }),
    );
    disc.position.z = 0.05;
    coverage.position.set(x, y, 0);
    coverage.add(ring, disc);
  }

  group.add(marker, coverage);
  group.userData.entityId = e.id;

  const labelPos = new Vector3(x, y, z + 0.45);

  return {
    group,
    labelRoot: marker,
    gizmoTarget: marker,
    labelPos,
    setSelected: (on: boolean) => {
      headMat.emissiveIntensity = on ? 1.6 : 0.6;
    },
    applyPose: (ent) => {
      if (ent.type !== "sensor") {
        return;
      }
      marker.position.set(ent.position[0], ent.position[1], ent.position[2]);
      if (ent.areaMode === "circle") {
        coverage.position.set(ent.position[0], ent.position[1], 0);
      }
      labelPos.set(ent.position[0], ent.position[1], ent.position[2] + 0.45);
    },
    dispose: () => disposeObject(group),
  };
}

/* ------------------------------------------------------------------ */
/* Dispatcher + seeding                                              */
/* ------------------------------------------------------------------ */

/** Linked scene map beneath its child-to-parent pose. */
export function buildChild(e: ChildSceneEntity): EntityNodes {
  const group = new Group();
  let disposed = false;

  applyChildPoseToObject(group, {
    position: e.position,
    rotation: e.rotation,
    scale: e.scale,
  });
  group.userData.entityId = e.id;

  const labelPos = new Vector3(
    e.position[0],
    e.position[1],
    e.position[2] + Math.max(e.size[2] * e.scale[2], 0.5) + 0.4,
  );

  let ready: Promise<void> | undefined;
  if (e.geometry?.mapUrl) {
    ready = loadSceneObject(e.geometry).then((object) => {
      if (disposed) {
        disposeObject(object);
        return;
      }
      group.add(object);
    }).catch((error) => {
      if (!disposed) {
        console.warn(`Failed to load child scene map: ${e.name}`, error);
      }
    });
  }

  return {
    group,
    ready,
    gizmoTarget: group,
    labelPos,
    setSelected: () => {},
    applyPose: (ent) => {
      if (ent.type !== "child") {
        return;
      }
      applyChildPoseToObject(group, {
        position: ent.position,
        rotation: ent.rotation,
        scale: ent.scale,
      });
      labelPos.set(
        ent.position[0],
        ent.position[1],
        ent.position[2] + Math.max(ent.size[2] * ent.scale[2], 0.5) + 0.4,
      );
    },
    dispose: () => {
      disposed = true;
      disposeObject(group);
    },
  };
}

/**
 * Proxy nodes for the scene map GLB. The mesh lives on `world.mapRoot`;
 * gizmoTarget points at that mesh so TransformControls edit its TRS.
 */
export function createMapNodes(mesh: Object3D, e: MapEntity): EntityNodes {
  const group = new Group();
  group.name = "map-entity-proxy";
  group.userData.entityId = e.id;
  const labelPos = new Vector3();
  const box = new Box3();
  const syncLabel = () => {
    box.setFromObject(mesh);
    if (!box.isEmpty()) {
      box.getCenter(labelPos);
    } else {
      labelPos.copy(mesh.position);
    }
  };
  syncLabel();
  return {
    group,
    gizmoTarget: mesh,
    labelPos,
    setSelected: () => {
      /* selection chrome is the gizmo */
    },
    applyPose: (ent) => {
      if (ent.type !== "map") {
        return;
      }
      applyScenePoseToObject(mesh, {
        translation: ent.position,
        rotation: ent.rotation,
        scale: ent.scale,
      });
      syncLabel();
    },
    dispose: () => {
      /* never dispose the scene mesh */
    },
  };
}

function buildEntityNodes(e: ViewportEntity): EntityNodes | null {
  switch (e.type) {
    case "region":
      return buildRegion(e);
    case "tripwire":
      return buildTripwire(e);
    case "camera":
      return buildCamera(e);
    case "sensor":
      return buildSensor(e);
    case "child":
      return buildChild(e);
    case "map":
      return null; // registered against world.sceneMesh in reconcile
    default:
      return null; // marks render via attachMarkRenderer
  }
}

function isFiniteTriple(v: unknown): v is [number, number, number] {
  return (
    Array.isArray(v) &&
    v.length >= 3 &&
    v.slice(0, 3).every((n) => Number.isFinite(Number(n)))
  );
}

function parseCameraPose(row: Record<string, unknown>): {
  position: [number, number, number];
  /** Rig euler for frustum meshes. */
  rotation: [number, number, number];
  /** REST / solve euler for map projection. */
  restRotation: [number, number, number];
  fov: number;
} | null {
  if (!isFiniteTriple(row.translation)) {
    return null;
  }
  const restRot: [number, number, number] = isFiniteTriple(row.rotation)
    ? [
        Number(row.rotation[0]),
        Number(row.rotation[1]),
        Number(row.rotation[2]),
      ]
    : [0, 0, 0];
  const intr = row.intrinsics as
    | {
        fov?: unknown;
        hfov?: unknown;
        vfov?: unknown;
        fx?: unknown;
        fy?: unknown;
      }
    | null
    | undefined;
  const res = row.resolution as
    | { width?: unknown; height?: unknown; w?: unknown; h?: unknown }
    | unknown[]
    | null
    | undefined;
  const num = (v: unknown) =>
    typeof v === "number" && Number.isFinite(v) ? v : null;
  const resObj = Array.isArray(res) ? null : res;
  const height =
    num(resObj?.height) ??
    num(resObj?.h) ??
    (Array.isArray(res) ? num(res[1]) : null);
  const fy = num(intr?.fy) ?? num(intr?.fx);
  // Prefer fy→vfov (legacy scenecamera.js); fall back to explicit vfov/fov.
  let fov = num(intr?.vfov) ?? num(intr?.fov) ?? null;
  if (fov == null && fy != null && height != null && fy > 0 && height > 0) {
    fov = (2 * Math.atan(height / (2 * fy)) * 180) / Math.PI;
  }
  if (fov == null) {
    const hfov = num(intr?.hfov);
    fov = hfov != null ? hfov * 0.75 : 60;
  }
  // Guard against treating focal length (hundreds of px) as degrees.
  if (fov <= 1 || fov >= 170) {
    fov = 60;
  }
  return {
    position: [
      Number(row.translation[0]),
      Number(row.translation[1]),
      Number(row.translation[2]),
    ],
    rotation: restPoseToRigEuler(restRot),
    restRotation: restRot,
    fov,
  };
}

function defaultRingPose(
  index: number,
  count: number,
): { position: [number, number, number]; rotation: [number, number, number] } {
  const angle = (index / Math.max(count, 1)) * Math.PI * 2;
  const position: [number, number, number] = [
    Math.cos(angle) * 15,
    Math.sin(angle) * 15,
    3,
  ];
  return { position, rotation: lookAtOriginEuler(position) };
}

async function seedCameras(
  cameras: SceneCameraBootstrap[],
  authToken: string,
): Promise<CameraEntity[]> {
  return Promise.all(
    cameras.map(async (c, i) => {
      let pose: {
        position: [number, number, number];
        rotation: [number, number, number];
        restRotation: [number, number, number];
        fov: number;
      } | null = null;
      if (authToken) {
        try {
          pose = parseCameraPose(
            (await api.getCamera(authToken, c.sensorId)) ?? {},
          );
        } catch {
          pose = null;
        }
      }
      const fallback = defaultRingPose(i, cameras.length);
      return {
        id: c.id,
        type: "camera",
        name: c.name,
        visible: true,
        position: pose?.position ?? fallback.position,
        target: [0, 0, 0],
        rotation: pose?.rotation ?? fallback.rotation,
        restRotation: pose?.restRotation ?? [0, 0, 0],
        posed: pose != null,
        fov: pose?.fov ?? 60,
        color: "#0a84ff",
      } satisfies CameraEntity;
    }),
  );
}

type SensorArea = {
  area?: string;
  x?: unknown;
  y?: unknown;
  z?: unknown;
  color?: unknown;
  radius?: unknown;
  points?: unknown;
};

function parseSensorPolyPoints(area: SensorArea | null): [number, number][] {
  if (!area || !Array.isArray(area.points)) {
    return [];
  }
  const out: [number, number][] = [];
  for (const p of area.points) {
    if (
      Array.isArray(p) &&
      p.length >= 2 &&
      Number.isFinite(Number(p[0])) &&
      Number.isFinite(Number(p[1]))
    ) {
      out.push([Number(p[0]), Number(p[1])]);
    }
  }
  return out;
}

export async function seedSensors(
  sensors: SceneSensorBootstrap[],
  authToken: string,
): Promise<SensorEntity[]> {
  return Promise.all(
    sensors.map(async (s, i) => {
      let area: SensorArea | null = null;
      try {
        area = s.areaJson ? (JSON.parse(s.areaJson) as SensorArea) : null;
      } catch {
        area = null;
      }
      const areaRadius =
        area?.area === "circle" && Number.isFinite(Number(area.radius))
          ? Number(area.radius)
          : 5;
      let position: [number, number, number] | null = null;
      let color = typeof area?.color === "string" && /^#[0-9a-f]{6}$/i.test(area.color)
        ? area.color : "#bf5af2";
      let singletonType: SensorEntity["singletonType"] = "environmental";
      if (authToken) {
        try {
          const row = (await api.getSensor(authToken, s.sensorId)) ?? {};
          if (isFiniteTriple(row.translation)) {
            position = [
              Number(row.translation[0]),
              Number(row.translation[1]),
              Number(row.translation[2]),
            ];
          }
          if (typeof row.color === "string" && /^#[0-9a-f]{6}$/i.test(row.color)) {
            color = row.color;
          }
          if (
            row.singleton_type === "attribute" ||
            row.singleton_type === "environmental"
          ) {
            singletonType = row.singleton_type;
          }
        } catch {
          position = null;
        }
      }
      const areaMode =
        area?.area === "circle" || area?.area === "poly" ? area.area : "scene";
      const points = areaMode === "poly" ? parseSensorPolyPoints(area) : [];
      if (!position) {
        if (
          Number.isFinite(Number(area?.x)) &&
          Number.isFinite(Number(area?.y))
        ) {
          position = [Number(area?.x), Number(area?.y), Number.isFinite(Number(area?.z)) ? Number(area?.z) : 0];
        } else if (points.length >= 3) {
          const [cx, cy] = centroidOf(points);
          position = [cx, cy, 0];
        } else {
          const angle = (i / Math.max(sensors.length, 1)) * Math.PI * 2;
          position = [Math.cos(angle) * 10, Math.sin(angle) * 10, 0];
        }
      }
      return {
        id: s.id,
        type: "sensor",
        name: s.name,
        visible: true,
        position,
        areaMode,
        radius: areaRadius,
        points,
        singletonType,
        color,
      } satisfies SensorEntity;
    }),
  );
}

/** Signature to skip no-op reseeds from geometry notifications. */
function geometrySignature(): string {
  const { regions, tripwires } = seedEntitiesFromGeometry();
  return JSON.stringify([
    regions.map((r) => [r.id, r.points, r.height, r.volumetric, r.name]),
    tripwires.map((t) => [t.id, t.points, t.name]),
  ]);
}

/* ------------------------------------------------------------------ */
/* Reconciler hook                                                     */
/* ------------------------------------------------------------------ */

export interface EntitySeed {
  cameras: SceneCameraBootstrap[];
  sensors: SceneSensorBootstrap[];
  children?: SceneChildBootstrap[];
  authToken: string;
}

/**
 * Viewport key for a child link.
 *
 * Local children use the child scene UUID and remote children use
 * `remote_child_id`. The ChildScene integer pk is not safe here: camera
 * and sensor bootstrap ids are those rows' integer pks, and the viewport
 * store is a single map keyed only by `id`.
 */
export function childViewportId(child: {
  id: string;
  restUid?: string | null;
}): string {
  const uid = child.restUid?.trim();
  return uid ? uid : child.id;
}

async function seedChildren(
  children: SceneChildBootstrap[],
  authToken: string,
): Promise<ChildSceneEntity[]> {
  return Promise.all(
    children.map(async (c) => {
      let pose = parseChildPose({});
      let transformSource: string | null = null;
      let geometry = sceneGeometryFromRest({
        uid: c.restUid,
        name: c.name,
        map: c.mapUrl,
      });
      if (authToken && c.restUid) {
        try {
          const row = await api.getChild(authToken, c.restUid);
          pose = parseChildPose(row);
          transformSource =
            row.transform_source != null ? String(row.transform_source) : null;
        } catch {
          /* keep identity pose */
        }
      }
      if (authToken && c.restUid && c.childType !== "remote") {
        try {
          geometry = sceneGeometryFromRest(await api.getScene(authToken, c.restUid));
        } catch (error) {
          console.warn(`Failed to fetch child scene map metadata: ${c.name}`, error);
        }
      }
      return {
        id: childViewportId(c),
        type: "child",
        name: c.name,
        visible: true,
        position: pose.position,
        rotation: pose.rotation,
        scale: pose.scale,
        size: CHILD_FOOTPRINT_SIZE,
        color: CHILD_FOOTPRINT_COLOR,
        childType: c.childType || "local",
        restUid: c.restUid,
        remoteChildId: c.remoteChildId,
        detailUrl: c.detailUrl,
        editHref: c.editHref || `?ss=child-edit&id=${c.restUid}`,
        transformSource,
        deleteUrl: c.deleteUrl,
        geometry,
      } satisfies ChildSceneEntity;
    }),
  );
}

export function useViewportEntities(
  world: ViewportWorld | null,
  seed: EntitySeed,
): void {
  // Seed regions/tripwires from the geometry model + cameras/sensors.
  useEffect(() => {
    if (!world) {
      return;
    }
    let cancelled = false;
    let lastSignature = "";

    const seedFromModel = () => {
      if (cancelled) {
        return;
      }
      const signature = geometrySignature();
      if (signature === lastSignature) {
        return;
      }
      lastSignature = signature;
      const { regions, tripwires } = seedEntitiesFromGeometry();
      const keep = new Set<string>();
      for (const e of [...regions, ...tripwires]) {
        keep.add(e.id);
      }
      const state = getViewportState();
      // Drop regions/tripwires no longer in the geometry model. In-progress
      // drafts are written into the model on create, so they stay in `keep`.
      // Orphan `draft-*` rows left after a server rekey must be removed or
      // the Outliner shows the same region twice.
      for (const [id, e] of Object.entries(state.entities)) {
        if ((e.type === "region" || e.type === "tripwire") && !keep.has(id)) {
          state.removeEntity(id);
        }
      }
      getViewportState().upsertEntities([...regions, ...tripwires]);
      // Only seed baselines for new ids — don't clear dirty in-progress edits.
      getPersistState().captureBaselines(
        [...regions, ...tripwires].map((e) => e.id),
        { onlyMissing: true },
      );
    };

    seedFromModel();
    const unsubGeometry = subscribeGeometry(seedFromModel);

    return () => {
      cancelled = true;
      unsubGeometry();
      const state = getViewportState();
      state.removeEntitiesByType("camera");
      state.removeEntitiesByType("sensor");
      // Children / cameras / sensors are owned by their list-sync effects.
      // Clearing children here dropped them whenever this geometry effect re-ran.
      getPersistState().clearBaselines();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [world]);

  // Cameras: sync when the gallery/bootstrap list changes (sheet create/edit).
  // Only seed ids missing from the store so in-progress Properties edits stay.
  useEffect(() => {
    if (!world) {
      return;
    }
    let cancelled = false;
    (async () => {
      const state = getViewportState();
      const keep = new Set(seed.cameras.map((c) => c.id));
      for (const [id, e] of Object.entries(state.entities)) {
        if (
          e.type === "camera" &&
          !id.startsWith("draft-") &&
          !keep.has(id)
        ) {
          state.removeEntity(id);
        }
      }
      for (const c of seed.cameras) {
        const existing = state.entities[c.id];
        if (existing?.type === "camera" && existing.name !== c.name) {
          state.updateEntity(c.id, { name: c.name });
        }
      }
      const missing = seed.cameras.filter((c) => !state.entities[c.id]);
      if (missing.length === 0) {
        return;
      }
      const camEntities = await seedCameras(missing, seed.authToken);
      if (cancelled) {
        return;
      }
      getViewportState().upsertEntities(camEntities);
      getPersistState().captureBaselines(
        camEntities.map((e) => e.id),
        { onlyMissing: true },
      );
    })();
    return () => {
      cancelled = true;
    };
  }, [world, seed.cameras, seed.authToken]);

  // Sensors: same list-sync as cameras (gallery / sheet create).
  useEffect(() => {
    if (!world) {
      return;
    }
    let cancelled = false;
    (async () => {
      const state = getViewportState();
      const keep = new Set(seed.sensors.map((s) => s.id));
      for (const [id, e] of Object.entries(state.entities)) {
        if (
          e.type === "sensor" &&
          !id.startsWith("draft-") &&
          !keep.has(id)
        ) {
          state.removeEntity(id);
        }
      }
      for (const s of seed.sensors) {
        const existing = state.entities[s.id];
        if (existing?.type === "sensor" && existing.name !== s.name) {
          state.updateEntity(s.id, { name: s.name });
        }
      }
      const missing = seed.sensors.filter((s) => !state.entities[s.id]);
      if (missing.length === 0) {
        return;
      }
      const senEntities = await seedSensors(missing, seed.authToken);
      if (cancelled) {
        return;
      }
      getViewportState().upsertEntities(senEntities);
      getPersistState().captureBaselines(
        senEntities.map((e) => e.id),
        { onlyMissing: true },
      );
    })();
    return () => {
      cancelled = true;
    };
  }, [world, seed.sensors, seed.authToken]);

  // Child scenes: re-seed when the linked-children list reference changes
  // (create/edit/unlink via ChildSheet or Properties).
  useEffect(() => {
    if (!world) {
      return;
    }
    const children = seed.children ?? [];
    let cancelled = false;
    (async () => {
      const childEntities = await seedChildren(children, seed.authToken);
      if (cancelled) {
        return;
      }
      const keep = new Set(childEntities.map((e) => e.id));
      const state = getViewportState();
      for (const [id, e] of Object.entries(state.entities)) {
        if (e.type === "child" && !keep.has(id)) {
          state.removeEntity(id);
        }
      }
      if (childEntities.length) {
        state.upsertEntities(childEntities);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [world, seed.children, seed.authToken]);

  // Reconcile three nodes off structVersion; selection + frustums separately.
  useEffect(() => {
    if (!world) {
      return;
    }
    const labelEls = new Map<
      string,
      { entityId: string; el: HTMLDivElement; pos: Vector3 }
    >();
    const labelKey = (id: string, key: string) => `${id}\u0000${key}`;

    const removeLabels = (id: string) => {
      for (const [key, entry] of labelEls) {
        if (entry.entityId === id) {
          entry.el.remove();
          labelEls.delete(key);
        }
      }
    };

    const ensureLabel = (
      id: string,
      key: string,
      name: string,
      pos: Vector3,
      color?: string,
    ) => {
      const entryKey = labelKey(id, key);
      let entry = labelEls.get(entryKey);
      if (!entry) {
        const el = document.createElement("div");
        el.className = "ss-vlabel ss-vlabel--entity";
        if (!getViewportState().showLabels) {
          el.style.display = "none";
        }
        labelEls.set(
          entryKey,
          (entry = { entityId: id, el, pos: pos.clone() }),
        );
        world.labelLayer.appendChild(el);
      } else {
        entry.pos.copy(pos);
      }
      if (entry.el.textContent !== name) {
        entry.el.textContent = name;
      }
      entry.el.style.color = color ?? "";
      entry.el.style.borderColor = color ?? "";
      return entry.el;
    };

    const clearNodes = () => {
      for (const [, nodes] of world.entityNodes) {
        if (nodes.group.parent === world.entityRoot) {
          world.entityRoot.remove(nodes.group);
        }
        nodes.dispose();
      }
      world.entityNodes.clear();
      for (const { el } of labelEls.values()) {
        el.remove();
      }
      labelEls.clear();
    };

    const reconcileMap = (e: MapEntity, selectedId: string | null) => {
      const mesh = world.sceneMesh;
      if (!mesh) {
        return false;
      }
      mesh.visible = e.visible !== false;
      let nodes = world.entityNodes.get(e.id);
      if (!nodes || nodes.gizmoTarget !== mesh) {
        const prev = world.entityNodes.get(e.id);
        if (prev) {
          if (prev.group.parent === world.entityRoot) {
            world.entityRoot.remove(prev.group);
          }
          prev.dispose();
        }
        nodes = createMapNodes(mesh, e);
        world.entityNodes.set(e.id, nodes);
      }
      nodes.setSelected(e.id === selectedId);
      if (e.visible !== false) {
        nodes.applyPose?.(e);
        ensureLabel(e.id, "name", e.name, nodes.labelPos);
      } else {
        removeLabels(e.id);
      }
      return true;
    };

    const reconcile = () => {
      const { entities, selectedId, showFrustums } = getViewportState();
      const calibration = useCalibrateStore.getState();
      const seen = new Set<string>();
      for (const e of Object.values(entities)) {
        if (e.type === "map") {
          if (reconcileMap(e, selectedId)) {
            seen.add(e.id);
          }
          continue;
        }
        if (calibration.active ? e.id !== calibration.cameraId : e.visible === false) {
          continue;
        }
        const nodes = buildEntityNodes(e);
        if (!nodes) {
          continue;
        }
        seen.add(e.id);
        const prev = world.entityNodes.get(e.id);
        if (prev) {
          if (prev.group.parent === world.entityRoot) {
            world.entityRoot.remove(prev.group);
          }
          prev.dispose();
        }
        removeLabels(e.id);
        world.entityRoot.add(nodes.group);
        world.entityNodes.set(e.id, nodes);
        nodes.setSelected(e.id === selectedId);
        nodes.setFrustumVisible?.(showFrustums);
        ensureLabel(
          e.id,
          "name",
          e.name,
          entityLabelAnchor(nodes.labelRoot ?? nodes.group, nodes.labelPos),
        );
        void nodes.ready?.then(() => {
          if (world.entityNodes.get(e.id) === nodes) {
            ensureLabel(
              e.id,
              "name",
              e.name,
              entityLabelAnchor(nodes.labelRoot ?? nodes.group, nodes.labelPos),
            );
          }
        });
        for (const label of nodes.extraLabels ?? []) {
          ensureLabel(
            e.id,
            label.key,
            label.text,
            label.position,
            label.color,
          );
        }
      }
      for (const [id, nodes] of world.entityNodes) {
        if (!seen.has(id)) {
          if (nodes.group.parent === world.entityRoot) {
            world.entityRoot.remove(nodes.group);
          }
          nodes.dispose();
          world.entityNodes.delete(id);
          removeLabels(id);
        }
      }
    };

    const updateSelection = (selectedId: string | null) => {
      for (const [id, nodes] of world.entityNodes) {
        nodes.setSelected(id === selectedId);
      }
    };

    const applyPoses = () => {
      const { entities } = getViewportState();
      for (const [id, nodes] of world.entityNodes) {
        const e = entities[id];
        if (!e || !nodes.applyPose) {
          continue;
        }
        nodes.applyPose(e);
        const entry = labelEls.get(labelKey(id, "name"));
        entry?.pos.copy(entityLabelAnchor(nodes.labelRoot ?? nodes.group, nodes.labelPos));
      }
    };

    const unsubscribe = useViewportStore.subscribe((s, prev) => {
      if (s.structVersion !== prev.structVersion) {
        reconcile();
      }
      if (s.poseVersion !== prev.poseVersion) {
        applyPoses();
      }
      if (s.selectedId !== prev.selectedId) {
        updateSelection(s.selectedId);
      }
      if (s.showFrustums !== prev.showFrustums) {
        for (const [, nodes] of world.entityNodes) {
          nodes.setFrustumVisible?.(s.showFrustums);
        }
      }
      if (s.showLabels !== prev.showLabels) {
        for (const { el } of labelEls.values()) {
          el.style.display = s.showLabels ? "" : "none";
        }
      }
    });
    const unsubscribeCalibration = useCalibrateStore.subscribe((state, previous) => {
      if (state.active !== previous.active || state.cameraId !== previous.cameraId) {
        reconcile();
      }
    });
    reconcile();

    const projV = new Vector3();
    const updateLabels = () => {
      if (!getViewportState().showLabels) {
        return;
      }
      const cam = world.getActiveCamera();
      const w = world.host.clientWidth;
      const h = world.host.clientHeight;
      for (const { el, pos } of labelEls.values()) {
        if (useCalibrateStore.getState().active) {
          el.style.display = "none";
          continue;
        }
        projV.copy(pos).project(cam);
        if (projV.z > 1 || projV.z < -1) {
          el.style.display = "none";
          continue;
        }
        el.style.display = "";
        el.style.transform = `translate(-50%, -100%) translate(${(projV.x * 0.5 + 0.5) * w}px, ${(-projV.y * 0.5 + 0.5) * h - 4}px)`;
      }
    };
    world.onFrame.add(updateLabels);

    return () => {
      unsubscribe();
      unsubscribeCalibration();
      world.onFrame.delete(updateLabels);
      clearNodes();
    };
  }, [world]);
}
