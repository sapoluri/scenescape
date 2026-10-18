// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Client-side region / tripwire feed events from live marks + map geometry.
 *
 * Analytics MQTT `event/#` only publishes on enter/exit transitions and is
 * often silent in demos; the map already has region/tripwire polygons and
 * mark positions, so Telemetry derives the same transitions locally.
 */

import { useEffect, useRef, useState } from "react";
import { useViewportMarksList } from "../viewport/marks";
import { getViewportState, useViewportStore } from "../viewport/store";
import type {
  MarkEntity,
  RegionEntity,
  TripwireEntity,
} from "../viewport/types";
import {
  capTelemetryEvents,
  TOPIC_KEY_REGION,
  TOPIC_KEY_TRIPWIRE,
  type TelemetryEvent,
} from "./telemetryModel";

function listGeometry(): {
  regions: RegionEntity[];
  tripwires: TripwireEntity[];
} {
  const regions: RegionEntity[] = [];
  const tripwires: TripwireEntity[] = [];
  for (const e of Object.values(getViewportState().entities)) {
    if (e.type === "region") {
      regions.push(e);
    } else if (e.type === "tripwire") {
      tripwires.push(e);
    }
  }
  return { regions, tripwires };
}

/** Ray-cast point-in-polygon (floor x/y meters). */
export function pointInPolygon(
  x: number,
  y: number,
  poly: readonly [number, number][],
): boolean {
  if (poly.length < 3) {
    return false;
  }
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i]![0];
    const yi = poly[i]![1];
    const xj = poly[j]![0];
    const yj = poly[j]![1];
    const intersect =
      yi > y !== yj > y &&
      x < ((xj - xi) * (y - yi)) / (yj - yi || Number.EPSILON) + xi;
    if (intersect) {
      inside = !inside;
    }
  }
  return inside;
}

/** True if segments AB and CD properly intersect (not just touch endpoints). */
export function segmentsIntersect(
  a: readonly [number, number],
  b: readonly [number, number],
  c: readonly [number, number],
  d: readonly [number, number],
): boolean {
  const cross = (
    o: readonly [number, number],
    p: readonly [number, number],
    q: readonly [number, number],
  ) => (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0]);
  const d1 = cross(c, d, a);
  const d2 = cross(c, d, b);
  const d3 = cross(a, b, c);
  const d4 = cross(a, b, d);
  if (
    ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
    ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))
  ) {
    return true;
  }
  return false;
}

export type GeometryOccupancy = Map<string, Set<string>>;

/**
 * Diff mark positions against regions/tripwires and emit feed rows.
 * `occupancy` maps regionId → set of mark ids currently inside.
 * `prevPos` maps mark id → last [x,y] for tripwire segment tests.
 */
export function diffGeometryTelemetry(
  marks: MarkEntity[],
  regions: RegionEntity[],
  tripwires: TripwireEntity[],
  occupancy: GeometryOccupancy,
  prevPos: Map<string, [number, number]>,
  now: number,
  seqStart: number,
): {
  events: TelemetryEvent[];
  occupancy: GeometryOccupancy;
  prevPos: Map<string, [number, number]>;
  nextSeq: number;
} {
  let seq = seqStart;
  const out: TelemetryEvent[] = [];
  const nextOcc: GeometryOccupancy = new Map();
  const nextPos = new Map<string, [number, number]>();
  const markById = new Map(marks.map((m) => [m.id, m]));

  for (const m of marks) {
    nextPos.set(m.id, [m.position[0], m.position[1]]);
  }

  for (const region of regions) {
    if (!region.visible || region.points.length < 3) {
      continue;
    }
    const inside = new Set<string>();
    for (const m of marks) {
      if (pointInPolygon(m.position[0], m.position[1], region.points)) {
        inside.add(m.id);
      }
    }
    nextOcc.set(region.id, inside);
    const prev = occupancy.get(region.id) ?? new Set<string>();
    for (const id of inside) {
      if (!prev.has(id)) {
        const m = markById.get(id);
        out.push({
          seq: seq++,
          at: now,
          kind: "region-enter",
          topicKey: TOPIC_KEY_REGION,
          trackId: id,
          className: m?.className ?? "",
          entityId: region.id,
          entityName: region.name,
          position: m
            ? [m.position[0], m.position[1]]
            : undefined,
        });
      }
    }
    for (const id of prev) {
      if (!inside.has(id)) {
        const m = markById.get(id);
        out.push({
          seq: seq++,
          at: now,
          kind: "region-exit",
          topicKey: TOPIC_KEY_REGION,
          trackId: id,
          className: m?.className ?? id.replace(/^mark:/, ""),
          entityId: region.id,
          entityName: region.name,
        });
      }
    }
  }

  for (const tw of tripwires) {
    if (!tw.visible) {
      continue;
    }
    const a = tw.a ?? tw.points[0];
    const b = tw.b ?? tw.points[1];
    if (!a || !b) {
      continue;
    }
    for (const m of marks) {
      const prev = prevPos.get(m.id);
      if (!prev) {
        continue;
      }
      const cur: [number, number] = [m.position[0], m.position[1]];
      if (segmentsIntersect(prev, cur, a, b)) {
        out.push({
          seq: seq++,
          at: now,
          kind: "tripwire-cross",
          topicKey: TOPIC_KEY_TRIPWIRE,
          trackId: m.id,
          className: m.className,
          entityId: tw.id,
          entityName: tw.name,
          position: cur,
        });
      }
    }
  }

  // Marks that vanished: treat as exit from any region they occupied.
  for (const [regionId, prev] of occupancy) {
    if (nextOcc.has(regionId)) {
      continue;
    }
    const region = regions.find((r) => r.id === regionId);
    for (const id of prev) {
      out.push({
        seq: seq++,
        at: now,
        kind: "region-exit",
        topicKey: TOPIC_KEY_REGION,
        trackId: id,
        className: id.replace(/^mark:/, ""),
        entityId: regionId,
        entityName: region?.name,
      });
    }
  }

  out.reverse();
  return { events: out, occupancy: nextOcc, prevPos: nextPos, nextSeq: seq };
}

/** Live geometry-derived region/tripwire feed for the active scene. */
export function useGeometryEventTelemetry(sceneId: string): TelemetryEvent[] {
  const marks = useViewportMarksList();
  // Stable string key — avoid returning new arrays from the selector.
  const geometryKey = useViewportStore((s) => {
    const parts: string[] = [];
    for (const e of Object.values(s.entities)) {
      if (e.type === "region" || e.type === "tripwire") {
        parts.push(`${e.id}:${e.points.map((p) => p.join(",")).join(";")}`);
      }
    }
    return parts.sort().join("|");
  });
  const [events, setEvents] = useState<TelemetryEvent[]>([]);
  const occRef = useRef<GeometryOccupancy>(new Map());
  const posRef = useRef<Map<string, [number, number]>>(new Map());
  const seqRef = useRef(0);

  useEffect(() => {
    occRef.current = new Map();
    posRef.current = new Map();
    seqRef.current = 0;
    setEvents([]);
  }, [sceneId]);

  useEffect(() => {
    const { regions, tripwires } = listGeometry();
    const now = Date.now();
    const { events: fresh, occupancy, prevPos, nextSeq } = diffGeometryTelemetry(
      marks,
      regions,
      tripwires,
      occRef.current,
      posRef.current,
      now,
      seqRef.current,
    );
    occRef.current = occupancy;
    posRef.current = prevPos;
    seqRef.current = nextSeq;
    if (fresh.length > 0) {
      setEvents((prev) => capTelemetryEvents([...fresh, ...prev]));
    }
  }, [marks, geometryKey]);

  return events;
}
