// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef } from "react";
import { useShallow } from "zustand/react/shallow";
import type { SceneObjectMark } from "../scene/map/liveMarks";
import { getViewportState, useViewportStore } from "./store";
import type { MarkEntity } from "./types";

/**
 * Live tracked-object marks for the 3D viewport.
 *
 * Consumes the frozen `ss-scene-objects` window event dispatched by the
 * legacy sscape.js MQTT layer — the same event the 2D MarksLayer uses — so
 * the 3D viewport needs no new plumbing (ADR-19: no new window.ss* bridges).
 * Marks are stored as MarkEntity rows in the zustand scene-graph store; the
 * three.js layer reconciles meshes imperatively.
 *
 * Clearing matches marks.js / liveMarks: IDs absent from the current
 * regulated frame are removed immediately. A short TTL prune remains as a
 * safety net when MQTT goes silent without an empty frame.
 */

export const MARK_ID_PREFIX = "mark:";
/** Backup clear when regulated frames stop arriving entirely. */
const STALE_AFTER_MS = 5000;
const PRUNE_INTERVAL_MS = 2000;
const MIN_MOVE_M = 0.05;
export const DEFAULT_MARK_COLOR = "#64d2ff";

/** Normalize a SceneObjectMark translation (meters, x/y plane) to [x, y]. */
export function parseMarkPosition(
  t: SceneObjectMark["translation"],
): [number, number] | null {
  if (!t) {
    return null;
  }
  if (Array.isArray(t) && t.length >= 2) {
    const x = Number(t[0]);
    const y = Number(t[1]);
    return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
  }
  if (typeof t === "object") {
    const x = Number((t as { x?: number }).x);
    const y = Number((t as { y?: number }).y);
    if (Number.isFinite(x) && Number.isFinite(y)) {
      return [x, y];
    }
  }
  return null;
}

/**
 * Heading about the Z axis (radians, 0 = +X, CCW positive) derived from
 * successive positions. Returns `prev` when the mark barely moved, so
 * stationary marks don't spin.
 */
export function deriveHeading(
  prevX: number,
  prevY: number,
  x: number,
  y: number,
  prev: number,
): number {
  const dx = x - prevX;
  const dy = y - prevY;
  if (Math.hypot(dx, dy) < MIN_MOVE_M) {
    return prev;
  }
  return Math.atan2(dy, dx);
}

function markKey(id: string | number): string {
  return `${MARK_ID_PREFIX}${id}`;
}

/** Label text matching legacy 3D UI: `category #objectId`. */
export function markLabel(className: string, objectId: string | number): string {
  return `${className} #${objectId}`;
}

/** Raw tracker id from a `mark:` store key (or pass-through). */
export function trackIdFromMarkKey(key: string): string {
  return key.startsWith(MARK_ID_PREFIX)
    ? key.slice(MARK_ID_PREFIX.length)
    : key;
}

/**
 * Track color from object id — same rule as the 2D map marks
 * (`marks.js` / `liveMarks.ts`: `#` + first 6 characters of the id).
 */
export function colorFromTrackId(id: string | number): string {
  return `#${String(id).substring(0, 6)}`;
}

interface MarkHookOptions {
  /** Asset3D class name → mark color, from scene bootstrap. */
  assetMarkColors?: Record<string, string>;
}

interface SeenMark {
  x: number;
  y: number;
  heading: number;
  lastSeen: number;
}

/**
 * Frame-diff: store keys in `seen` that are not present in this MQTT frame.
 * Presence is by object id (even if translation fails to parse), matching
 * marks.js so a bad sample does not drop an active track.
 */
export function markKeysLeftScene(
  seenKeys: Iterable<string>,
  objects: SceneObjectMark[],
): string[] {
  const present = new Set<string>();
  for (const m of objects) {
    if (m == null || m.id === undefined || m.id === null) {
      continue;
    }
    present.add(markKey(m.id));
  }
  const gone: string[] = [];
  for (const id of seenKeys) {
    if (!present.has(id)) {
      gone.push(id);
    }
  }
  return gone;
}

/**
 * Build MarkEntity rows for one regulated frame and update `seen` in place.
 */
export function buildMarkEntities(
  objects: SceneObjectMark[],
  seen: Map<string, SeenMark>,
  colors: Record<string, string>,
  now: number,
): MarkEntity[] {
  const entities: MarkEntity[] = [];
  for (const m of objects) {
    if (m == null || m.id === undefined || m.id === null) {
      continue;
    }
    const pos = parseMarkPosition(m.translation);
    if (!pos) {
      continue;
    }
    const [x, y] = pos;
    const id = markKey(m.id);
    const className =
      (typeof m.type === "string" && m.type) ||
      (m.tag_id !== undefined ? String(m.tag_id) : String(m.id));
    const prev = seen.get(id);
    const heading = prev
      ? deriveHeading(prev.x, prev.y, x, y, prev.heading)
      : Number.NaN;
    const dx = prev ? x - prev.x : 0;
    const dy = prev ? y - prev.y : 0;
    seen.set(id, { x, y, heading, lastSeen: now });
    entities.push({
      id,
      type: "mark",
      name: markLabel(className, m.id),
      visible: true,
      className,
      color: colors[className] ?? DEFAULT_MARK_COLOR,
      position: [x, y, 0],
      heading,
      speed: Math.hypot(dx, dy),
    });
  }
  return entities;
}

/**
 * Subscribe to live marks and mirror them into the viewport store.
 * Safe to mount once per scene; unmounting removes this scene's marks.
 */
export function useViewportMarks({ assetMarkColors }: MarkHookOptions): void {
  const colorsRef = useRef(assetMarkColors);
  colorsRef.current = assetMarkColors;
  const seenRef = useRef(new Map<string, SeenMark>());

  useEffect(() => {
    const seen = seenRef.current;

    const onObjects = (ev: Event) => {
      const detail = (ev as CustomEvent<{ objects?: SceneObjectMark[] }>)
        .detail;
      const objects = detail?.objects;
      if (!Array.isArray(objects)) {
        return;
      }
      const now = Date.now();
      const colors = colorsRef.current ?? {};

      // Frame-diff clear + upsert in one store write (marks.js semantics).
      const gone = markKeysLeftScene(seen.keys(), objects);
      for (const id of gone) {
        seen.delete(id);
      }
      const entities = buildMarkEntities(objects, seen, colors, now);
      if (gone.length > 0 || entities.length > 0) {
        getViewportState().upsertEntities(entities, {
          fromMarks: true,
          removeIds: gone,
        });
      }
    };

    const prune = () => {
      const now = Date.now();
      const stale: string[] = [];
      for (const [id, s] of seen) {
        if (now - s.lastSeen > STALE_AFTER_MS) {
          stale.push(id);
          seen.delete(id);
        }
      }
      if (stale.length > 0) {
        getViewportState().removeEntities(stale, { fromMarks: true });
      }
    };
    const pruneTimer = window.setInterval(prune, PRUNE_INTERVAL_MS);

    // Stay in sync with the frozen trails checkbox (ss-show-trails).
    const onTrails = (ev: Event) => {
      const detail = (ev as CustomEvent<{ show?: boolean }>).detail;
      const show = Boolean(detail?.show);
      const cur = getViewportState().showTrails;
      if (show !== cur) {
        getViewportState().toggleOverlay("showTrails");
      }
    };

    window.addEventListener("ss-scene-objects", onObjects);
    window.addEventListener("ss-show-trails", onTrails);
    return () => {
      window.removeEventListener("ss-scene-objects", onObjects);
      window.removeEventListener("ss-show-trails", onTrails);
      window.clearInterval(pruneTimer);
      getViewportState().removeEntitiesByType("mark");
      seen.clear();
    };
  }, []);
}

/** Read-only access to the marks currently in the store (for tests/debug). */
export function useViewportMarksList(): MarkEntity[] {
  // useShallow is required: filtering Object.values yields a new array every
  // getSnapshot call; without it Telemetry hits React error #185 (max update depth).
  return useViewportStore(
    useShallow((s) =>
      Object.values(s.entities).filter(
        (e): e is MarkEntity => e.type === "mark",
      ),
    ),
  );
}
