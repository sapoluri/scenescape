// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
  BoxGeometry,
  BufferGeometry,
  CapsuleGeometry,
  CylinderGeometry,
  Euler,
  Object3D,
} from "three";
import { api, restJson } from "../lib/rest";

/**
 * Object Library data layer (Phase 1.3).
 *
 * The library is the set of per-class `Asset3D` mark definitions. The 3D
 * view composes a live mark as:
 *
 *   tracker transform (scene controller) × default pose × (GLB | primitive)
 *
 * where the default pose is the Asset3D rotation/translation/scale axes.
 * When no GLB is set, `markGeometry` selects the placeholder primitive.
 * This module owns the REST mapping and the pure geometry helpers the mark
 * editor stage uses; it adds no new window bridges.
 */

/** Primitive drawn for a class when no GLB is configured. */
export type MarkGeometry = "box" | "capsule" | "cylinder";

export const MARK_GEOMETRY_VALUES: MarkGeometry[] = [
  "box",
  "capsule",
  "cylinder",
];

export const MARK_GEOMETRY_LABELS: Record<MarkGeometry, string> = {
  box: "Box",
  capsule: "Capsule",
  cylinder: "Cylinder",
};

/** Short UI copy: sizes are full extents (meters), never radii / half-sizes. */
export const MARK_GEOMETRY_HELP: Record<MarkGeometry, string> = {
  box: "X, Y, and Z are full extents in meters (width × depth × height), not half-sizes.",
  capsule:
    "Diameter is the full plan width (not radius). Height is the total length including both rounded ends (must be ≥ diameter).",
  cylinder:
    "Diameter is the full plan width (not radius). Height is the full cylinder length along Z.",
};

export function parseMarkGeometry(v: unknown): MarkGeometry {
  const s = String(v ?? "")
    .trim()
    .toLowerCase();
  if (s === "capsule" || s === "cylinder") {
    return s;
  }
  return "box";
}

/** Capsule/cylinder use a circular plan; box uses independent X/Y. */
export function isCircularMarkGeometry(kind: MarkGeometry): boolean {
  return kind === "capsule" || kind === "cylinder";
}

/**
 * Plan diameter for circular mark geometries. Stored as equal `xSize`/`ySize`
 * (legacy rows may differ — use the smaller axis so the circle still fits).
 */
export function markDiameter(
  rec: Pick<Asset3DRecord, "xSize" | "ySize">,
): number {
  const x = Number.isFinite(rec.xSize) ? rec.xSize : 0;
  const y = Number.isFinite(rec.ySize) ? rec.ySize : 0;
  return Math.max(Math.min(x, y), 0);
}

/**
 * Normalize sizes/buffers when switching mark geometry so circular shapes
 * keep a single plan diameter (and matching plan buffer).
 */
export function withMarkGeometry(
  rec: Asset3DRecord,
  kind: MarkGeometry,
): Asset3DRecord {
  const markGeometry = parseMarkGeometry(kind);
  if (!isCircularMarkGeometry(markGeometry)) {
    return { ...rec, markGeometry };
  }
  const diameter = markDiameter(rec) || Math.max(rec.xSize, rec.ySize, 0.01);
  const planBuffer = Math.max(rec.xBuffer, rec.yBuffer, 0);
  return {
    ...rec,
    markGeometry,
    xSize: diameter,
    ySize: diameter,
    xBuffer: planBuffer,
    yBuffer: planBuffer,
  };
}

function fmtSize(n: number): string {
  if (!Number.isFinite(n)) {
    return "0";
  }
  const t = Math.round(n * 1000) / 1000;
  return String(t);
}

/** Compact size label for library cards (⌀×H for circular, W×D×H for box). */
export function markSizeLabel(rec: Asset3DRecord): string {
  const kind = parseMarkGeometry(rec.markGeometry);
  if (isCircularMarkGeometry(kind)) {
    const d = markDiameter(rec);
    return `⌀${fmtSize(d)}×${fmtSize(rec.zSize)} m`;
  }
  return `${fmtSize(rec.xSize)}×${fmtSize(rec.ySize)}×${fmtSize(rec.zSize)} m`;
}

export type Asset3DRecord = {
  uid: string;
  name: string;
  xSize: number;
  ySize: number;
  zSize: number;
  xBuffer: number;
  yBuffer: number;
  zBuffer: number;
  markColor: string;
  /** Placeholder primitive when `modelUrl` is null. */
  markGeometry: MarkGeometry;
  modelUrl: string | null;
  rotation: [number, number, number];
  translation: [number, number, number];
  /**
   * Default pose scale for the GLB. Stored as XYZ triple for the API, but the
   * Mark editor treats it as uniform (all axes equal). Unused for primitives.
   */
  scale: [number, number, number];
  rotationFromVelocity: boolean;
  trackingRadius: number;
  shiftType: number;
  projectToMap: boolean;
  geometricCenter: [number, number, number];
  mass: number;
  centerOfMass: [number, number, number];
  isStatic: boolean;
  ttl: number;
  linearDamping: number;
  angularDamping: number;
  restitution: number;
  friction: [number, number];
};

const num = (v: unknown, fallback: number): number => {
  const n = typeof v === "string" && v.trim() !== "" ? Number(v) : v;
  return typeof n === "number" && Number.isFinite(n) ? n : fallback;
};

const bool = (v: unknown): boolean => v === true || v === "True" || v === "true" || v === 1;

const vec3 = (v: unknown, fallback: [number, number, number]): [number, number, number] => {
  if (Array.isArray(v)) {
    return [num(v[0], fallback[0]), num(v[1], fallback[1]), num(v[2], fallback[2])];
  }
  return [...fallback] as [number, number, number];
};

const str = (v: unknown, fallback = ""): string =>
  v == null ? fallback : String(v);

/** Normalize one Asset3D serializer row into an Asset3DRecord. */
export function parseAsset3D(row: Record<string, unknown>): Asset3DRecord {
  const modelUrl =
    typeof row.model_3d === "string" && row.model_3d.trim()
      ? row.model_3d.trim()
      : null;
  return {
    uid: str(row.uid ?? row.pk ?? row.id),
    name: str(row.name),
    xSize: num(row.x_size, 1),
    ySize: num(row.y_size, 1),
    zSize: num(row.z_size, 1),
    xBuffer: num(row.x_buffer_size, 0),
    yBuffer: num(row.y_buffer_size, 0),
    zBuffer: num(row.z_buffer_size, 0),
    markColor: str(row.mark_color, "#888888") || "#888888",
    markGeometry: parseMarkGeometry(row.mark_geometry),
    modelUrl,
    rotation: [num(row.rotation_x, 0), num(row.rotation_y, 0), num(row.rotation_z, 0)],
    translation: [
      num(row.translation_x, 0),
      num(row.translation_y, 0),
      num(row.translation_z, 0),
    ],
    scale: (() => {
      const legacy = num(row.scale, NaN);
      const sx = num(row.scale_x, NaN);
      const sy = num(row.scale_y, NaN);
      const sz = num(row.scale_z, NaN);
      const axes = [sx, sy, sz].map((v) =>
        Number.isFinite(v) && v > 0
          ? v
          : Number.isFinite(legacy) && legacy > 0
            ? legacy
            : 1,
      ) as [number, number, number];
      // GLB scale is uniform — collapse legacy per-axis rows to one factor.
      return uniformScaleTriple(uniformScaleOf(axes));
    })(),
    rotationFromVelocity: bool(row.rotation_from_velocity),
    trackingRadius: num(row.tracking_radius, 2),
    shiftType: num(row.shift_type, 1),
    projectToMap: bool(row.project_to_map),
    geometricCenter: vec3(row.geometric_center, [0, 0, 0]),
    mass: num(row.mass, 1),
    centerOfMass: vec3(row.center_of_mass, [0, 0, 0]),
    isStatic: bool(row.is_static),
    ttl: num(row.ttl, 0),
    linearDamping: num(row.linear_damping, 0.05),
    angularDamping: num(row.angular_damping, 0.05),
    restitution: num(row.coefficient_of_restitution, 0.5),
    friction: (() => {
      const f = vec3(row.friction_coefficients, [0.5, 0.4, 0]);
      return [f[0], f[1]] as [number, number];
    })(),
  };
}

/** Defaults for the "new class" form (mirror the Django model defaults). */
export function defaultAsset3D(): Asset3DRecord {
  return parseAsset3D({ uid: "", name: "" });
}

/** GET /api/v1/assets — the per-class library rows. */
export async function listAssets(token: string): Promise<Asset3DRecord[]> {
  const res = await restJson<{ results?: unknown[] } | unknown[]>(
    "GET",
    "/assets",
    token,
  );
  const rows = Array.isArray(res) ? res : (res.results ?? []);
  return (rows as Record<string, unknown>[]).map(parseAsset3D);
}

/** GET /api/v1/asset/:uid — one class definition. */
export async function getAssetRecord(
  token: string,
  uid: string,
): Promise<Asset3DRecord> {
  return parseAsset3D(await api.getAsset(token, uid));
}

/** Django/DRF BooleanField+choices only matches str(True)/str(False). */
function formChoiceBool(value: boolean): "True" | "False" {
  return value ? "True" : "False";
}

/**
 * Encode an Asset3D record as multipart FormData for PUT /api/v1/asset/:uid
 * (or POST /api/v1/asset). Mirrors the serializer: booleans as "True"/"False"
 * strings, list fields as JSON strings, GLB as a file part.
 */
export function buildAssetFormData(
  rec: Asset3DRecord,
  opts: { modelFile?: File | null; clearModel?: boolean } = {},
): FormData {
  const form = new FormData();
  const f = (k: string, v: string) => form.append(k, v);
  f("name", rec.name.trim());
  f("mark_color", rec.markColor);
  f("mark_geometry", parseMarkGeometry(rec.markGeometry));
  {
    const s = uniformScaleOf(rec.scale);
    f("scale_x", String(s));
    f("scale_y", String(s));
    f("scale_z", String(s));
    f("scale", String(s));
  }
  f("x_size", String(rec.xSize));
  f("y_size", String(rec.ySize));
  f("z_size", String(rec.zSize));
  f("tracking_radius", String(rec.trackingRadius));
  f("shift_type", String(rec.shiftType));
  f("project_to_map", formChoiceBool(rec.projectToMap));
  f("rotation_from_velocity", formChoiceBool(rec.rotationFromVelocity));
  f("x_buffer_size", String(rec.xBuffer));
  f("y_buffer_size", String(rec.yBuffer));
  f("z_buffer_size", String(rec.zBuffer));
  f("rotation_x", String(rec.rotation[0]));
  f("rotation_y", String(rec.rotation[1]));
  f("rotation_z", String(rec.rotation[2]));
  f("translation_x", String(rec.translation[0]));
  f("translation_y", String(rec.translation[1]));
  f("translation_z", String(rec.translation[2]));
  f("mass", String(rec.mass));
  f("is_static", formChoiceBool(rec.isStatic));
  f("ttl", String(rec.ttl));
  f("linear_damping", String(rec.linearDamping));
  f("angular_damping", String(rec.angularDamping));
  f("coefficient_of_restitution", String(rec.restitution));
  f("geometric_center", JSON.stringify(rec.geometricCenter));
  f("center_of_mass", JSON.stringify(rec.centerOfMass));
  f("friction_coefficients", JSON.stringify(rec.friction));
  if (opts.modelFile) {
    form.append("model_3d", opts.modelFile);
  } else if (opts.clearModel) {
    form.append("clear_model_3d", "True");
  }
  return form;
}

/** PUT /api/v1/asset/:uid — save the edited class definition. */
export async function saveAssetRecord(
  token: string,
  uid: string,
  rec: Asset3DRecord,
  opts: { modelFile?: File | null; clearModel?: boolean } = {},
): Promise<Asset3DRecord> {
  const saved = await api.updateAsset(
    token,
    uid,
    buildAssetFormData(rec, opts),
  );
  return parseAsset3D(saved as Record<string, unknown>);
}

/** POST /api/v1/asset — create a new class definition. */
export async function createAssetRecord(
  token: string,
  rec: Asset3DRecord,
  opts: { modelFile?: File | null } = {},
): Promise<Asset3DRecord> {
  const saved = await api.createAsset(token, buildAssetFormData(rec, opts));
  return parseAsset3D(saved as Record<string, unknown>);
}

export const SHIFT_TYPE_LABELS: Record<number, string> = {
  1: "Center",
  2: "Bottom",
};

export function shiftTypeLabel(shiftType: number): string {
  return SHIFT_TYPE_LABELS[shiftType] ?? `Unknown (${shiftType})`;
}

/* ------------------------------------------------------------------ */
/* Stage geometry helpers (pure; unit-tested)                          */
/* ------------------------------------------------------------------ */

export type Footprint = {
  /** Object plan rectangle (x_size × y_size), meters. */
  inner: { x: number; y: number };
  /** Buffer-expanded plan rectangle, meters. Buffer applies per side. */
  outer: { x: number; y: number };
  /** Object height (z_size), meters. */
  height: number;
};

/**
 * Footprint rectangles for the stage floor overlay. The buffer zone expands
 * the plan rectangle by the buffer amount on each side.
 */
export function footprintOf(rec: Asset3DRecord): Footprint {
  const ix = Math.max(rec.xSize, 0.01);
  const iy = Math.max(rec.ySize, 0.01);
  return {
    inner: { x: ix, y: iy },
    outer: {
      x: ix + 2 * Math.max(rec.xBuffer, 0),
      y: iy + 2 * Math.max(rec.yBuffer, 0),
    },
    height: Math.max(rec.zSize, 0.01),
  };
}

/**
 * Build a Z-up mark primitive sized to the class AABB (meters).
 * `size.x/y/z` are full extents (not half-sizes / radii). Capsule/cylinder
 * use plan diameter = min(x, y). Place the mesh at `z = height / 2` so the
 * bottom sits on the floor. Do not also apply Asset3D pose scale to these
 * meshes — scale is for GLBs only.
 */
export function createMarkPrimitiveGeometry(
  kind: MarkGeometry,
  size: { x: number; y: number; z: number },
): BufferGeometry {
  const x = Math.max(size.x, 0.01);
  const y = Math.max(size.y, 0.01);
  const z = Math.max(size.z, 0.01);
  const shape = parseMarkGeometry(kind);
  if (shape === "cylinder") {
    const radius = Math.min(x, y) * 0.5;
    const geo = new CylinderGeometry(radius, radius, z, 24);
    geo.rotateX(Math.PI / 2); // Y-up cylinder → Z-up
    return geo;
  }
  if (shape === "capsule") {
    const radius = Math.min(x, y) * 0.5;
    const length = Math.max(z - 2 * radius, 0);
    const geo = new CapsuleGeometry(radius, length, 6, 16);
    geo.rotateX(Math.PI / 2); // Y-up capsule → Z-up
    return geo;
  }
  // BoxGeometry args are full width/height/depth in meters.
  return new BoxGeometry(x, y, z);
}

const DEG = Math.PI / 180;

export type ApplyDefaultPoseOpts = {
  /**
   * When false, leave scale at (1,1,1). Mark primitives are already sized in
   * meters via x_size/y_size/z_size — applying GLB pose scale would re-scale
   * those absolute extents (often ~2×). Default true for GLB pose frames.
   */
  applyScale?: boolean;
};

/**
 * Apply an Asset3D default pose to a three.js object (Z-up, XYZ euler in
 * degrees, per-axis scale) — the same convention the placement canvas uses
 * for SceneEulerPose.
 */
export function applyDefaultPose(
  obj: Object3D,
  rec: Asset3DRecord,
  opts: ApplyDefaultPoseOpts = {},
): void {
  obj.position.set(rec.translation[0], rec.translation[1], rec.translation[2]);
  obj.rotation.order = "XYZ";
  obj.rotation.set(
    rec.rotation[0] * DEG,
    rec.rotation[1] * DEG,
    rec.rotation[2] * DEG,
  );
  if (opts.applyScale === false) {
    obj.scale.set(1, 1, 1);
  } else {
    const s = uniformScaleOf(rec.scale);
    obj.scale.set(s, s, s);
  }
  obj.updateMatrix();
}

/** Single positive scale factor from a stored XYZ triple (GLB is uniform). */
export function uniformScaleOf(scale: [number, number, number]): number {
  const axes = scale.filter((v) => Number.isFinite(v) && v > 0);
  if (axes.length === 0) {
    return 1;
  }
  if (axes.every((v) => Math.abs(v - axes[0]) < 1e-9)) {
    return axes[0];
  }
  // Legacy non-uniform rows: geometric mean so volume is preserved-ish.
  return Math.exp(axes.reduce((a, v) => a + Math.log(v), 0) / axes.length);
}

export function uniformScaleTriple(s: number): [number, number, number] {
  const v = Number.isFinite(s) && s > 0 ? s : 1;
  return [v, v, v];
}

/** Read an Asset3D default pose back off a three.js object. */
export function readDefaultPose(obj: Object3D): {
  rotation: [number, number, number];
  translation: [number, number, number];
  scale: [number, number, number];
} {
  const e = new Euler().setFromQuaternion(obj.quaternion, "XYZ");
  const axis = (v: number) => (Number.isFinite(v) && v > 0 ? v : 1);
  return {
    rotation: [
      (e.x / DEG + 540) % 360 - 180,
      (e.y / DEG + 540) % 360 - 180,
      (e.z / DEG + 540) % 360 - 180,
    ],
    translation: [obj.position.x, obj.position.y, obj.position.z],
    scale: [axis(obj.scale.x), axis(obj.scale.y), axis(obj.scale.z)],
  };
}

/* ------------------------------------------------------------------ */
/* "Simulate tracker" path (pure; unit-tested)                         */
/* ------------------------------------------------------------------ */

export type TrackerSample = {
  x: number;
  y: number;
  /** Velocity heading about Z, radians, 0 = +X (matches live-mark deriveHeading). */
  heading: number;
};

/**
 * Circular tracker path for the "Simulate tracker" toggle: radius R meters,
 * one lap every `periodSec` seconds, counter-clockwise. Heading is the
 * velocity direction so `rotation_from_velocity` can be verified visually.
 */
export function simulateTrackerPath(
  tSec: number,
  radius = 4,
  periodSec = 12,
): TrackerSample {
  const theta = ((tSec % periodSec) / periodSec) * Math.PI * 2;
  const x = radius * Math.cos(theta);
  const y = radius * Math.sin(theta);
  // d/dt (R cos θ, R sin θ) = Rω (−sin θ, cos θ).
  const heading = Math.atan2(Math.cos(theta), -Math.sin(theta));
  return { x, y, heading };
}

/** Basename of a GLB URL for the library cards. */
export function glbBasename(url: string | null): string {
  if (!url) {
    return "No GLB";
  }
  try {
    const path = new URL(url, "http://x/").pathname;
    const base = path.split("/").pop() || url;
    return decodeURIComponent(base);
  } catch {
    return url;
  }
}
