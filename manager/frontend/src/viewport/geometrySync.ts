// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Bridge between the 3D viewport entity store and the frozen 2D geometry
 * model (`src/scene/map/geometryModel.ts`).
 *
 * The geometry model stays the single source of truth for region/tripwire
 * persistence: the existing `persistSceneGeometry` path (Save buttons,
 * `window.ssPersistGeometry`) reads it, and the 2D editors write it. The 3D
 * tools write through these helpers so both views stay consistent, and every
 * write publishes the frozen `ss-roi-dirty` / `ss-trip-dirty` events.
 */

import {
  getRoiList,
  getTripwireList,
  removeRoi,
  removeTripwire,
  replaceRoiPoints,
  replaceTripPoints,
  resetGeometryModel,
  upsertRoiMeta,
  upsertTripMeta,
  type GeometryPoint,
  type RoiGeometry,
  type TripwireGeometry,
} from "../scene/map/geometryModel";
import type { RegionEntity, TripwireEntity } from "./types";

export type { RoiGeometry, TripwireGeometry };

export interface GeometrySnapshot {
  rois: RoiGeometry[];
  trips: TripwireGeometry[];
}

export function snapshotGeometry(): GeometrySnapshot {
  return {
    rois: getRoiList().map((r) => ({
      ...r,
      points: r.points.map((p) => [...p] as GeometryPoint),
      sectors: r.sectors.map((s) => ({ ...s })),
    })),
    trips: getTripwireList().map((t) => ({
      ...t,
      points: t.points.map((p) => [...p] as GeometryPoint),
    })),
  };
}

/** Replace the whole geometry model (undo/redo). Dispatches dirty events. */
export function restoreGeometry(snap: GeometrySnapshot): void {
  resetGeometryModel();
  for (const r of snap.rois) {
    upsertRoiMeta(r.uuid, {
      title: r.title,
      points: r.points,
      volumetric: r.volumetric,
      height: r.height,
      buffer_size: r.buffer_size,
      range_max: r.range_max,
      sectors: r.sectors,
    });
  }
  for (const t of snap.trips) {
    upsertTripMeta(t.uuid, { title: t.title, points: t.points });
  }
  publishGeometryDirty("roi", true);
  publishGeometryDirty("trip", true);
}

export function publishGeometryDirty(
  kind: "roi" | "trip",
  dirty: boolean,
): void {
  const eventName = kind === "roi" ? "ss-roi-dirty" : "ss-trip-dirty";
  window.dispatchEvent(new CustomEvent(eventName, { detail: dirty }));
}

/** Bounding box of floor-plane points. */
export function bboxOf(
  points: [number, number][],
): { min: [number, number]; max: [number, number] } {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const [x, y] of points) {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }
  if (!Number.isFinite(minX)) {
    minX = minY = maxX = maxY = 0;
  }
  return {
    min: [minX, minY],
    max: [maxX, maxY],
  };
}

export function centroidOf(points: [number, number][]): [number, number] {
  if (points.length === 0) {
    return [0, 0];
  }
  let sx = 0;
  let sy = 0;
  for (const [x, y] of points) {
    sx += x;
    sy += y;
  }
  return [sx / points.length, sy / points.length];
}

const DEFAULT_REGION_COLOR = "#30d158";
const DEFAULT_TRIPWIRE_COLOR = "#ff9f0a";

/** Geometry model → viewport entities (seed + 2D-edit sync). */
export function seedEntitiesFromGeometry(): {
  regions: RegionEntity[];
  tripwires: TripwireEntity[];
} {
  const regions = getRoiList().map((r): RegionEntity => {
    const points = r.points.map(
      (p) => [Number(p[0]), Number(p[1])] as [number, number],
    );
    const { min, max } = bboxOf(points);
    return {
      id: r.uuid,
      type: "region",
      name: r.title || r.uuid.slice(0, 8),
      visible: true,
      points,
      min,
      max,
      height: Number(r.height) || 1,
      color: r.sectors?.[0]?.color
        ? sectorColorToHex(r.sectors[0].color)
        : DEFAULT_REGION_COLOR,
      volumetric: Boolean(r.volumetric),
      bufferSize: Number(r.buffer_size) || 0,
    };
  });
  const tripwires = getTripwireList().map((t): TripwireEntity => {
    const points = t.points.map(
      (p) => [Number(p[0]), Number(p[1])] as [number, number],
    );
    return {
      id: t.uuid,
      type: "tripwire",
      name: t.title || t.uuid.slice(0, 8),
      visible: true,
      points,
      a: points[0] ?? [0, 0],
      b: points[points.length - 1] ?? [0, 0],
      color: DEFAULT_TRIPWIRE_COLOR,
    };
  });
  return { regions, tripwires };
}

function sectorColorToHex(color: string): string {
  switch (color) {
    case "green":
      return "#30d158";
    case "yellow":
      return "#ffd60a";
    case "red":
      return "#ff453a";
    default:
      return DEFAULT_REGION_COLOR;
  }
}

/** Viewport entity → geometry model (3D tool edits). */
export function writeRegionToModel(e: RegionEntity): void {
  upsertRoiMeta(e.id, {
    title: e.name,
    volumetric: e.volumetric,
    height: e.height,
    buffer_size: e.bufferSize,
  });
  replaceRoiPoints(e.id, e.points);
  publishGeometryDirty("roi", true);
}

export function writeTripwireToModel(e: TripwireEntity): void {
  upsertTripMeta(e.id, { title: e.name });
  replaceTripPoints(e.id, e.points);
  publishGeometryDirty("trip", true);
}

export function removeRegionFromModel(id: string): void {
  removeRoi(id);
  publishGeometryDirty("roi", true);
}

export function removeTripwireFromModel(id: string): void {
  removeTripwire(id);
  publishGeometryDirty("trip", true);
}
