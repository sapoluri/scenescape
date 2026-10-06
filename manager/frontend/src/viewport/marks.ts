// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef } from "react";
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
 */

export const MARK_ID_PREFIX = "mark:";
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
      const entities: MarkEntity[] = [];
      for (const m of objects) {
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
          name: className,
          visible: true,
          className,
          color: colors[className] ?? DEFAULT_MARK_COLOR,
          position: [x, y, 0],
          heading,
          speed: Math.hypot(dx, dy),
        });
      }
      if (entities.length > 0) {
        getViewportState().upsertEntities(entities);
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
      for (const id of stale) {
        getViewportState().removeEntity(id);
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
  return useViewportStore((s) =>
    Object.values(s.entities).filter(
      (e): e is MarkEntity => e.type === "mark",
    ),
  );
}
