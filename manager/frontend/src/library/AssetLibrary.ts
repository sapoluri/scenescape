// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Euler, Object3D } from "three";
import { api, restJson } from "../lib/rest";

/**
 * Object Library data layer (Phase 1.3).
 *
 * The library is the set of per-class `Asset3D` mark definitions. The 3D
 * view composes a live mark as:
 *
 *   tracker transform (scene controller) × default pose × GLB
 *
 * where the default pose is the Asset3D rotation/translation/scale triple.
 * This module owns the REST mapping and the pure geometry helpers the mark
 * editor stage uses; it adds no new window bridges.
 */

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
  modelUrl: string | null;
  rotation: [number, number, number];
  translation: [number, number, number];
  scale: number;
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
    modelUrl,
    rotation: [num(row.rotation_x, 0), num(row.rotation_y, 0), num(row.rotation_z, 0)],
    translation: [
      num(row.translation_x, 0),
      num(row.translation_y, 0),
      num(row.translation_z, 0),
    ],
    scale: num(row.scale, 1) || 1,
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
  f("scale", String(rec.scale));
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

const DEG = Math.PI / 180;

/**
 * Apply an Asset3D default pose to a three.js object (Z-up, XYZ euler in
 * degrees, uniform scale) — the same convention the placement canvas uses
 * for SceneEulerPose.
 */
export function applyDefaultPose(obj: Object3D, rec: Asset3DRecord): void {
  obj.position.set(rec.translation[0], rec.translation[1], rec.translation[2]);
  obj.rotation.order = "XYZ";
  obj.rotation.set(
    rec.rotation[0] * DEG,
    rec.rotation[1] * DEG,
    rec.rotation[2] * DEG,
  );
  const s = Number.isFinite(rec.scale) && rec.scale > 0 ? rec.scale : 1;
  obj.scale.setScalar(s);
  obj.updateMatrix();
}

/** Read an Asset3D default pose back off a three.js object. */
export function readDefaultPose(obj: Object3D): {
  rotation: [number, number, number];
  translation: [number, number, number];
  scale: number;
} {
  const e = new Euler().setFromQuaternion(obj.quaternion, "XYZ");
  return {
    rotation: [
      (e.x / DEG + 540) % 360 - 180,
      (e.y / DEG + 540) % 360 - 180,
      (e.z / DEG + 540) % 360 - 180,
    ],
    translation: [obj.position.x, obj.position.y, obj.position.z],
    scale: obj.scale.x,
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
