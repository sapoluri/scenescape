// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from "react";
import {
  BoxGeometry,
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
  Line,
  LineBasicMaterial,
  LineSegments,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Object3D,
  PerspectiveCamera,
  Quaternion,
  RingGeometry,
  Shape,
  ShapeGeometry,
  SphereGeometry,
  Vector2,
  Vector3,
} from "three";
import { api } from "../lib/rest";
import { subscribeGeometry } from "../scene/map/geometryModel";
import type {
  SceneCameraBootstrap,
  SceneSensorBootstrap,
} from "../scene/types";
import {
  bboxOf,
  centroidOf,
  seedEntitiesFromGeometry,
} from "./geometrySync";
import { getViewportState, useViewportStore } from "./store";
import type {
  CameraEntity,
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
  /** Object the TransformControls gizmo attaches to. */
  gizmoTarget: Object3D;
  /** World-space anchor for the name label. */
  labelPos: Vector3;
  setSelected: (on: boolean) => void;
  setFrustumVisible?: (visible: boolean) => void;
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
      | { dispose?: () => void }
      | { dispose?: () => void }[]
      | undefined;
    if (Array.isArray(mat)) {
      mat.forEach((m) => m.dispose?.());
    } else {
      mat?.dispose?.();
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

/* ------------------------------------------------------------------ */
/* Regions                                                             */
/* ------------------------------------------------------------------ */

function buildRegion(e: RegionEntity): EntityNodes {
  const group = new Group();
  const baseColor = new Color(e.color);
  const [cx, cy] = centroidOf(e.points);
  const pivot = new Group();
  pivot.position.set(cx, cy, 0);

  const shape = new Shape(
    e.points.map(([x, y]) => new Vector2(x - cx, y - cy)),
  );
  let geo: BufferGeometry;
  if (e.volumetric && e.height > 0.01) {
    geo = new ExtrudeGeometry(shape, { depth: e.height, bevelEnabled: false });
  } else {
    geo = new ShapeGeometry(shape);
  }
  const fillMat = new MeshStandardMaterial({
    color: baseColor.clone(),
    transparent: true,
    opacity: 0.22,
    roughness: 0.85,
    side: DoubleSide,
    depthWrite: false,
  });
  const fill = new Mesh(geo, fillMat);
  const edgeMat = new LineBasicMaterial({ color: baseColor.clone() });
  const edges = new LineSegments(new EdgesGeometry(geo), edgeMat);
  if (!e.volumetric || e.height <= 0.01) {
    fill.position.z = 0.05;
    edges.position.z = 0.05;
  }
  const { min, max } = bboxOf(e.points);
  const proxy = new Mesh(
    new BoxGeometry(
      Math.max(max[0] - min[0], 0.5),
      Math.max(max[1] - min[1], 0.5),
      e.volumetric ? Math.max(e.height, 0.5) : 1,
    ),
    proxyMaterial(),
  );
  proxy.position.set(
    (min[0] + max[0]) / 2 - cx,
    (min[1] + max[1]) / 2 - cy,
    (e.volumetric ? Math.max(e.height, 0.5) : 1) / 2,
  );
  pivot.add(fill, edges, proxy);
  group.add(pivot);
  group.userData.entityId = e.id;

  return {
    group,
    gizmoTarget: pivot,
    labelPos: new Vector3(cx, cy, (e.volumetric ? e.height : 0) + 0.6),
    setSelected: (on: boolean) => {
      edgeMat.color.set(on ? "#ffffff" : e.color);
      fillMat.opacity = on ? 0.42 : 0.22;
    },
    dispose: () => disposeObject(group),
  };
}

/* ------------------------------------------------------------------ */
/* Tripwires                                                           */
/* ------------------------------------------------------------------ */

function buildTripwire(e: TripwireEntity): EntityNodes {
  const group = new Group();
  const baseColor = new Color(e.color);
  const [cx, cy] = centroidOf(e.points);
  const pivot = new Group();
  pivot.position.set(cx, cy, 0);

  const lineMat = new LineBasicMaterial({ color: baseColor.clone() });
  const pts = e.points.map(([x, y]) => new Vector3(x - cx, y - cy, 0.15));
  const line = new Line(new BufferGeometry().setFromPoints(pts), lineMat);
  pivot.add(line);

  const postMat = new MeshBasicMaterial({ color: baseColor.clone() });
  const postGeo = new CylinderGeometry(0.06, 0.06, 1.6, 8);
  postGeo.rotateX(Math.PI / 2); // axis Y -> Z (Z-up)
  for (const [x, y] of e.points) {
    const post = new Mesh(postGeo, postMat);
    post.position.set(x - cx, y - cy, 0.8);
    pivot.add(post);
  }

  // Direction cone on the last segment (mockup language).
  if (e.points.length >= 2) {
    const ra = e.points[e.points.length - 2];
    const rb = e.points[e.points.length - 1];
    const dir = new Vector3(rb[0] - ra[0], rb[1] - ra[1], 0);
    if (dir.lengthSq() > 1e-6) {
      dir.normalize();
      const cone = new Mesh(new ConeGeometry(0.35, 1, 12), postMat);
      cone.quaternion.setFromUnitVectors(new Vector3(0, 1, 0), dir);
      cone.position.set(rb[0] - cx, rb[1] - cy, 0.5);
      pivot.add(cone);
    }
  }

  const { min, max } = bboxOf(e.points);
  const proxy = new Mesh(
    new BoxGeometry(
      Math.max(max[0] - min[0], 0.5),
      Math.max(max[1] - min[1], 0.5),
      1.6,
    ),
    proxyMaterial(),
  );
  proxy.position.set(
    (min[0] + max[0]) / 2 - cx,
    (min[1] + max[1]) / 2 - cy,
    0.8,
  );
  pivot.add(proxy);
  group.add(pivot);
  group.userData.entityId = e.id;

  const mid: [number, number] = [
    (e.a[0] + e.b[0]) / 2,
    (e.a[1] + e.b[1]) / 2,
  ];
  return {
    group,
    gizmoTarget: pivot,
    labelPos: new Vector3(mid[0], mid[1], 1.9),
    setSelected: (on: boolean) => {
      const c = on ? "#ffffff" : e.color;
      lineMat.color.set(c);
      postMat.color.set(c);
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
    new Euler(d2r(rotationDeg[0]), d2r(rotationDeg[1]), d2r(rotationDeg[2]), "XYZ"),
  );
  q.multiply(
    new Quaternion().setFromAxisAngle(new Vector3(1, 0, 0), Math.PI),
  );
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
    new Euler(d2r(e.rotation[0]), d2r(e.rotation[1]), d2r(e.rotation[2]), "XYZ"),
  );

  // Frustum rig: a real PerspectiveCamera (never rendered) + CameraHelper,
  // exactly like the legacy 3D view. The lens looks down local -Z.
  // NOTE: CameraHelper replaces its own matrix with the camera's matrixWorld,
  // so it must live under the identity outer group — parenting it under the
  // transformed rig would apply the rig transform twice and the frustum
  // would not align with the camera body.
  const persp = new PerspectiveCamera(e.fov, 4 / 3, 0.5, 30);
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
  rig.add(body, lens);

  const proxy = new Mesh(new BoxGeometry(1.6, 1.6, 1.6), proxyMaterial());
  rig.add(proxy);
  group.add(rig);
  // Added after the rig: CameraHelper reads the camera's matrixWorld during
  // the scene-graph update, so it must be traversed after the rig subtree
  // to see a fresh transform (no one-frame lag).
  group.add(helper);
  group.userData.entityId = e.id;

  return {
    group,
    gizmoTarget: rig,
    labelPos: new Vector3(
      e.position[0],
      e.position[1],
      e.position[2] + 1.2,
    ),
    setSelected: (on: boolean) => {
      bodyMat.emissiveIntensity = on ? 0.9 : 0;
    },
    setFrustumVisible: (visible: boolean) => {
      helper.visible = visible;
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

function buildSensor(e: SensorEntity): EntityNodes {
  const group = new Group();
  const base = new Color(e.color);
  const [x, y, z] = e.position;
  const radius = Math.max(e.radius, 0.5);

  const poleGeo = new CylinderGeometry(0.09, 0.12, 2.2, 10);
  poleGeo.rotateX(Math.PI / 2); // axis Y -> Z (Z-up)
  const pole = new Mesh(
    poleGeo,
    new MeshStandardMaterial({ color: 0x3a3b45, roughness: 0.6 }),
  );
  pole.position.z = 1.1;
  const headMat = new MeshStandardMaterial({
    color: base.clone(),
    emissive: base.clone(),
    emissiveIntensity: 0.6,
    roughness: 0.4,
  });
  const head = new Mesh(new SphereGeometry(0.22, 14, 10), headMat);
  head.position.z = 2.3;
  // RingGeometry/CircleGeometry already lie in the XY plane = Z-up floor.
  const ring = new Mesh(
    new RingGeometry(radius - 0.25, radius, 48),
    new MeshBasicMaterial({
      color: base.clone(),
      transparent: true,
      opacity: 0.35,
      side: DoubleSide,
    }),
  );
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
  const proxy = new Mesh(new BoxGeometry(1.5, 1.5, 2.6), proxyMaterial());
  proxy.position.z = 1.3;
  group.add(pole, head, ring, disc, proxy);
  group.position.set(x, y, z);
  group.userData.entityId = e.id;

  return {
    group,
    gizmoTarget: group,
    labelPos: new Vector3(x, y, z + 2.9),
    setSelected: (on: boolean) => {
      headMat.emissiveIntensity = on ? 1.6 : 0.6;
    },
    dispose: () => disposeObject(group),
  };
}

/* ------------------------------------------------------------------ */
/* Dispatcher + seeding                                              */
/* ------------------------------------------------------------------ */

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
    default:
      return null; // marks + child scenes render elsewhere / later phases
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
  rotation: [number, number, number];
  fov: number;
} | null {
  if (!isFiniteTriple(row.translation)) {
    return null;
  }
  const restRot: [number, number, number] = isFiniteTriple(row.rotation)
    ? [Number(row.rotation[0]), Number(row.rotation[1]), Number(row.rotation[2])]
    : [0, 0, 0];
  const intr = row.intrinsics as
    | { fov?: unknown; hfov?: unknown; vfov?: unknown }
    | null
    | undefined;
  const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
  const fov =
    num(intr?.vfov) ?? num(intr?.fov) ?? (num(intr?.hfov) ?? 60) * 0.75;
  return {
    position: [
      Number(row.translation[0]),
      Number(row.translation[1]),
      Number(row.translation[2]),
    ],
    rotation: restPoseToRigEuler(restRot),
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
  radius?: unknown;
};

async function seedSensors(
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
      if (authToken) {
        try {
          const row = (await api.getSensor(authToken, s.sensorId)) ?? {};
          if (isFiniteTriple(row.translation)) {
            position = [
              Number(row.translation[0]),
              Number(row.translation[1]),
              0,
            ];
          }
        } catch {
          position = null;
        }
      }
      if (!position) {
        if (
          area?.area === "circle" &&
          Number.isFinite(Number(area.x)) &&
          Number.isFinite(Number(area.y))
        ) {
          position = [Number(area.x), Number(area.y), 0];
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
        radius: areaRadius,
        color: "#bf5af2",
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
  authToken: string;
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
      // Drop model-backed entities deleted in 2D; never in-progress drafts.
      for (const [id, e] of Object.entries(state.entities)) {
        if (
          (e.type === "region" || e.type === "tripwire") &&
          !id.startsWith("draft-") &&
          !keep.has(id)
        ) {
          state.removeEntity(id);
        }
      }
      getViewportState().upsertEntities([...regions, ...tripwires]);
    };

    seedFromModel();
    const unsubGeometry = subscribeGeometry(seedFromModel);

    (async () => {
      const camEntities = await seedCameras(seed.cameras, seed.authToken);
      if (cancelled) {
        return;
      }
      getViewportState().upsertEntities(camEntities);
      const senEntities = await seedSensors(seed.sensors, seed.authToken);
      if (cancelled) {
        return;
      }
      getViewportState().upsertEntities(senEntities);
    })();

    return () => {
      cancelled = true;
      unsubGeometry();
      const state = getViewportState();
      state.removeEntitiesByType("camera");
      state.removeEntitiesByType("sensor");
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [world]);

  // Reconcile three nodes off structVersion; selection + frustums separately.
  useEffect(() => {
    if (!world) {
      return;
    }
    const labelEls = new Map<string, { el: HTMLDivElement; pos: Vector3 }>();

    const ensureLabel = (id: string, name: string, pos: Vector3) => {
      let entry = labelEls.get(id);
      if (!entry) {
        const el = document.createElement("div");
        el.className = "ss-vlabel ss-vlabel--entity";
        labelEls.set(id, (entry = { el, pos: pos.clone() }));
        world.labelLayer.appendChild(el);
      } else {
        entry.pos.copy(pos);
      }
      if (entry.el.textContent !== name) {
        entry.el.textContent = name;
      }
      return entry.el;
    };

    const clearNodes = () => {
      for (const [, nodes] of world.entityNodes) {
        world.entityRoot.remove(nodes.group);
        nodes.dispose();
      }
      world.entityNodes.clear();
      for (const { el } of labelEls.values()) {
        el.remove();
      }
      labelEls.clear();
    };

    const reconcile = () => {
      const { entities, selectedId, showFrustums } = getViewportState();
      const seen = new Set<string>();
      for (const e of Object.values(entities)) {
        if (e.visible === false) {
          continue;
        }
        const nodes = buildEntityNodes(e);
        if (!nodes) {
          continue;
        }
        seen.add(e.id);
        const prev = world.entityNodes.get(e.id);
        if (prev) {
          world.entityRoot.remove(prev.group);
          prev.dispose();
        }
        world.entityRoot.add(nodes.group);
        world.entityNodes.set(e.id, nodes);
        nodes.setSelected(e.id === selectedId);
        nodes.setFrustumVisible?.(showFrustums);
        ensureLabel(e.id, e.name, nodes.labelPos);
      }
      for (const [id, nodes] of world.entityNodes) {
        if (!seen.has(id)) {
          world.entityRoot.remove(nodes.group);
          nodes.dispose();
          world.entityNodes.delete(id);
          const entry = labelEls.get(id);
          entry?.el.remove();
          labelEls.delete(id);
        }
      }
    };

    const updateSelection = (selectedId: string | null) => {
      for (const [id, nodes] of world.entityNodes) {
        nodes.setSelected(id === selectedId);
      }
    };

    const unsubscribe = useViewportStore.subscribe((s, prev) => {
      if (s.structVersion !== prev.structVersion) {
        reconcile();
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
        projV.copy(pos).project(cam);
        if (projV.z > 1 || projV.z < -1) {
          el.style.display = "none";
          continue;
        }
        el.style.display = "";
        el.style.transform = `translate(-50%, -100%) translate(${(projV.x * 0.5 + 0.5) * w}px, ${(-projV.y * 0.5 + 0.5) * h}px)`;
      }
    };
    world.onFrame.add(updateLabels);

    return () => {
      unsubscribe();
      world.onFrame.delete(updateLabels);
      clearNodes();
    };
  }, [world]);
}

