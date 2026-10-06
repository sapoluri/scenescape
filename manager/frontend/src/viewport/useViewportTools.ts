// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef } from "react";
import {
  BufferGeometry,
  Euler,
  Group,
  Line,
  LineBasicMaterial,
  Mesh,
  MeshBasicMaterial,
  MOUSE,
  Plane,
  Raycaster,
  SphereGeometry,
  Vector2,
  Vector3,
} from "three";
import { lookAtOriginEuler } from "./entities";
import {
  bboxOf,
  centroidOf,
  removeRegionFromModel,
  removeTripwireFromModel,
  writeRegionToModel,
  writeTripwireToModel,
} from "./geometrySync";
import { getViewportState, useViewportStore } from "./store";
import { CREATION_TOOLS } from "./tools";
import type {
  CameraEntity,
  RegionEntity,
  SensorEntity,
  ToolId,
  TripwireEntity,
} from "./types";
import type { ViewportWorld } from "./world";

/**
 * Phase 1.2 tool interactions for the 3D viewport.
 *
 * - Selection: click (no-drag) raycasts the entity root; empty click clears.
 * - Transform tools (move/rotate/scale): attach TransformControls to the
 *   selection's gizmo target. Region/tripwire transforms bake into absolute
 *   floor points and write back to the geometry model (undoable, dirty).
 *   Camera/sensor moves are visual-only in Phase 1.2 — calibrated pose
 *   persistence arrives with the Phase 2.3 calibration wizard.
 * - Creation tools: floor-plane raycast gestures (region/tripwire polylines,
 *   camera/sensor placement, measure).
 * - Keyboard: Q/W/E/R/G/T/C/S/M/L tools, Delete, Ctrl+Z / Ctrl+Shift+Z,
 *   Esc (cancel), Enter (finish draft).
 */

const FLOOR = new Plane(new Vector3(0, 0, 1), 0);
const CLICK_TOLERANCE_PX = 6;
const DOUBLE_POINT_MS = 450;
const POINT_EPS = 1e-3;

const r2d = (r: number) => (r * 180) / Math.PI;

function floorPoint(
  world: ViewportWorld,
  ev: PointerEvent,
): [number, number] | null {
  const rect = world.host.getBoundingClientRect();
  const ndc = new Vector2(
    ((ev.clientX - rect.left) / rect.width) * 2 - 1,
    -((ev.clientY - rect.top) / rect.height) * 2 + 1,
  );
  const raycaster = new Raycaster();
  raycaster.setFromCamera(ndc, world.getActiveCamera());
  const hit = new Vector3();
  if (!raycaster.ray.intersectPlane(FLOOR, hit)) {
    return null;
  }
  return [hit.x, hit.y];
}

function pickEntity(
  world: ViewportWorld,
  ev: PointerEvent,
): string | null {
  const rect = world.host.getBoundingClientRect();
  const ndc = new Vector2(
    ((ev.clientX - rect.left) / rect.width) * 2 - 1,
    -((ev.clientY - rect.top) / rect.height) * 2 + 1,
  );
  const raycaster = new Raycaster();
  raycaster.setFromCamera(ndc, world.getActiveCamera());
  const hits = raycaster.intersectObject(world.entityRoot, true);
  for (const hit of hits) {
    let o: object | null = hit.object;
    while (o) {
      const id = (o as { userData?: { entityId?: string } }).userData
        ?.entityId;
      if (typeof id === "string") {
        return id;
      }
      o = (o as { parent?: object }).parent ?? null;
    }
  }
  return null;
}

/** What a pointer press hit: entity id, "gizmo", or null (empty space). */
function pickTarget(world: ViewportWorld, ev: PointerEvent): string | null {
  const id = pickEntity(world, ev);
  if (id) {
    return id;
  }
  // Gizmo handles live outside entityRoot — raycast the helper too.
  const rect = world.host.getBoundingClientRect();
  const ndc = new Vector2(
    ((ev.clientX - rect.left) / rect.width) * 2 - 1,
    -((ev.clientY - rect.top) / rect.height) * 2 + 1,
  );
  const raycaster = new Raycaster();
  raycaster.setFromCamera(ndc, world.getActiveCamera());
  try {
    const hits = raycaster.intersectObject(world.gizmo.getHelper(), true);
    if (hits.length > 0) {
      return "gizmo";
    }
  } catch {
    /* helper not ready */
  }
  return null;
}

function rotatePoint(
  [x, y]: [number, number],
  [cx, cy]: [number, number],
  angle: number,
): [number, number] {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c];
}

function pointsEqual(
  a: [number, number][],
  b: [number, number][],
): boolean {
  if (a.length !== b.length) {
    return false;
  }
  return a.every(
    ([x, y], i) =>
      Math.abs(x - b[i][0]) < POINT_EPS && Math.abs(y - b[i][1]) < POINT_EPS,
  );
}

interface Draft {
  tool: "region" | "tripwire";
  points: [number, number][];
  preview: Group;
  cursor: [number, number] | null;
  lastPointAt: number;
}

export function useViewportTools(world: ViewportWorld | null): void {
  const draftRef = useRef<Draft | null>(null);
  const finishDraftRef = useRef<(() => void) | null>(null);
  const measureRef = useRef<{
    a: [number, number] | null;
    visuals: Group | null;
    label: HTMLDivElement | null;
  }>({ a: null, visuals: null, label: null });
  const downRef = useRef<{ x: number; y: number; at: number } | null>(null);
  /** What pointerdown hit: entity id, "gizmo", or null (empty space). */
  const downPickRef = useRef<string | null>(null);

  /* ---------------- creation + selection pointer handling --------------- */
  useEffect(() => {
    if (!world) {
      return;
    }
    const host = world.host;

    const clearDraft = () => {
      const draft = draftRef.current;
      if (draft) {
        world.scene.remove(draft.preview);
        disposeGroup(draft.preview);
        draftRef.current = null;
      }
    };

    const clearMeasure = () => {
      const m = measureRef.current;
      if (m.visuals) {
        world.scene.remove(m.visuals);
        disposeGroup(m.visuals);
        m.visuals = null;
      }
      m.label?.remove();
      m.label = null;
      m.a = null;
    };

    const refreshPreview = () => {
      const draft = draftRef.current;
      if (!draft) {
        return;
      }
      world.scene.remove(draft.preview);
      disposeGroup(draft.preview);
      const g = new Group();
      const color = draft.tool === "region" ? 0x30d158 : 0xff9f0a;
      const mat = new MeshBasicMaterial({ color });
      for (const [x, y] of draft.points) {
        const s = new Mesh(new SphereGeometry(0.14, 10, 8), mat);
        s.position.set(x, y, 0.12);
        g.add(s);
      }
      const trail: [number, number][] = [...draft.points];
      if (draft.cursor) {
        trail.push(draft.cursor);
      }
      if (trail.length >= 2) {
        const line = new Line(
          new BufferGeometry().setFromPoints(
            trail.map(([x, y]) => new Vector3(x, y, 0.12)),
          ),
          new LineBasicMaterial({ color }),
        );
        g.add(line);
      }
      draft.preview = g;
      world.scene.add(g);
    };

    const startDraft = (tool: "region" | "tripwire") => {
      clearDraft();
      draftRef.current = {
        tool,
        points: [],
        preview: new Group(),
        cursor: null,
        lastPointAt: 0,
      };
      world.scene.add(draftRef.current.preview);
    };

    const finishDraft = () => {
      const draft = draftRef.current;
      if (!draft) {
        return;
      }
      const pts = draft.points;
      const valid =
        (draft.tool === "region" && pts.length >= 3) ||
        (draft.tool === "tripwire" && pts.length >= 2);
      clearDraft();
      if (!valid) {
        return;
      }
      const s = getViewportState();
      const id = `draft-${crypto.randomUUID()}`;
      s.commitHistory();
      if (draft.tool === "region") {
        const { min, max } = bboxOf(pts);
        const count = Object.values(s.entities).filter(
          (e) => e.type === "region",
        ).length;
        const entity: RegionEntity = {
          id,
          type: "region",
          name: `Region ${count + 1}`,
          visible: true,
          points: pts.map(([x, y]) => [x, y] as [number, number]),
          min,
          max,
          height: 1,
          color: "#30d158",
          volumetric: false,
          bufferSize: 0,
        };
        s.upsertEntities([entity]);
        writeRegionToModel(entity);
        s.select(id);
      } else {
        const count = Object.values(s.entities).filter(
          (e) => e.type === "tripwire",
        ).length;
        const entity: TripwireEntity = {
          id,
          type: "tripwire",
          name: `Tripwire ${count + 1}`,
          visible: true,
          points: pts.map(([x, y]) => [x, y] as [number, number]),
          a: pts[0],
          b: pts[pts.length - 1],
          color: "#ff9f0a",
        };
        s.upsertEntities([entity]);
        writeTripwireToModel(entity);
        s.select(id);
      }
    };

    const placeCamera = (pt: [number, number]) => {
      const s = getViewportState();
      const id = `draft-${crypto.randomUUID()}`;
      const count = Object.values(s.entities).filter(
        (e) => e.type === "camera",
      ).length;
      const position: [number, number, number] = [pt[0], pt[1], 2.5];
      const entity: CameraEntity = {
        id,
        type: "camera",
        name: `Camera ${count + 1}`,
        visible: true,
        position,
        target: [0, 0, 0],
        rotation: lookAtOriginEuler(position),
        fov: 60,
        color: "#0a84ff",
      };
      s.commitHistory();
      s.upsertEntities([entity]);
      s.select(id);
    };

    const placeSensor = (pt: [number, number]) => {
      const s = getViewportState();
      const id = `draft-${crypto.randomUUID()}`;
      const count = Object.values(s.entities).filter(
        (e) => e.type === "sensor",
      ).length;
      const entity: SensorEntity = {
        id,
        type: "sensor",
        name: `Sensor ${count + 1}`,
        visible: true,
        position: [pt[0], pt[1], 0],
        radius: 5,
        color: "#bf5af2",
      };
      s.commitHistory();
      s.upsertEntities([entity]);
      s.select(id);
    };

    const measureClick = (pt: [number, number]) => {
      const m = measureRef.current;
      if (!m.a) {
        m.a = pt;
        return;
      }
      const [ax, ay] = m.a;
      const [bx, by] = pt;
      const dist = Math.hypot(bx - ax, by - ay);
      clearMeasure();
      const g = new Group();
      const line = new Line(
        new BufferGeometry().setFromPoints([
          new Vector3(ax, ay, 0.1),
          new Vector3(bx, by, 0.1),
        ]),
        new LineBasicMaterial({ color: 0x0a84ff }),
      );
      g.add(line);
      for (const [x, y] of [
        [ax, ay],
        [bx, by],
      ] as [number, number][]) {
        const s = new Mesh(
          new SphereGeometry(0.12, 10, 8),
          new MeshBasicMaterial({ color: 0x0a84ff }),
        );
        s.position.set(x, y, 0.1);
        g.add(s);
      }
      world.scene.add(g);
      const label = document.createElement("div");
      label.className = "ss-vlabel ss-vlabel--measure";
      label.textContent = `${dist.toFixed(2)} m`;
      world.labelLayer.appendChild(label);
      // Project once (static overlay, not per-frame).
      const v = new Vector3((ax + bx) / 2, (ay + by) / 2, 0.4).project(
        world.getActiveCamera(),
      );
      const w = world.host.clientWidth;
      const h = world.host.clientHeight;
      label.style.transform = `translate(-50%, -100%) translate(${(v.x * 0.5 + 0.5) * w}px, ${(-v.y * 0.5 + 0.5) * h}px)`;
      measureRef.current = { a: null, visuals: g, label };
    };

    const onPointerDown = (ev: PointerEvent) => {
      if (ev.button !== 0) {
        return;
      }
      downRef.current = { x: ev.clientX, y: ev.clientY, at: performance.now() };
      const tool = getViewportState().activeTool;
      if (!CREATION_TOOLS.includes(tool)) {
        // Record the press target so a click that starts/ends on the gizmo
        // never deselects the entity being transformed.
        downPickRef.current = pickTarget(world, ev);
        return;
      }
      const pt = floorPoint(world, ev);
      if (!pt) {
        return;
      }
      if (tool === "region" || tool === "tripwire") {
        let draft = draftRef.current;
        if (!draft || draft.tool !== tool) {
          startDraft(tool);
          draft = draftRef.current;
        }
        if (!draft) {
          return;
        }
        // Swallow the second press of a double-click (finish uses dblclick).
        const now = performance.now();
        const last = draft.points[draft.points.length - 1];
        if (
          last &&
          now - draft.lastPointAt < DOUBLE_POINT_MS &&
          Math.hypot(last[0] - pt[0], last[1] - pt[1]) < 0.25
        ) {
          return;
        }
        draft.lastPointAt = now;
        draft.points.push(pt);
        refreshPreview();
      } else if (tool === "camera") {
        placeCamera(pt);
      } else if (tool === "sensor") {
        placeSensor(pt);
      } else if (tool === "measure") {
        measureClick(pt);
      }
    };

    const onPointerMove = (ev: PointerEvent) => {
      const draft = draftRef.current;
      if (!draft) {
        return;
      }
      const pt = floorPoint(world, ev);
      if (pt) {
        draft.cursor = pt;
        refreshPreview();
      }
    };

    const onPointerUp = (ev: PointerEvent) => {
      const down = downRef.current;
      const downPick = downPickRef.current;
      downRef.current = null;
      downPickRef.current = null;
      if (!down || ev.button !== 0) {
        return;
      }
      const moved = Math.hypot(ev.clientX - down.x, ev.clientY - down.y);
      if (moved > CLICK_TOLERANCE_PX) {
        return; // it was a drag, not a click
      }
      const tool = getViewportState().activeTool;
      if (CREATION_TOOLS.includes(tool)) {
        return; // creation tools act on pointerdown
      }
      const id = pickEntity(world, ev);
      if (id) {
        getViewportState().select(id);
      } else if (downPick !== "gizmo") {
        getViewportState().select(null);
      }
      // A click on a gizmo handle without dragging keeps the selection.
    };

    const onDblClick = (ev: MouseEvent) => {
      const tool = getViewportState().activeTool;
      if (tool === "region" || tool === "tripwire") {
        ev.preventDefault();
        finishDraft();
      }
    };

    host.addEventListener("pointerdown", onPointerDown);
    host.addEventListener("pointermove", onPointerMove);
    host.addEventListener("pointerup", onPointerUp);
    host.addEventListener("dblclick", onDblClick);
    finishDraftRef.current = finishDraft;
    return () => {
      host.removeEventListener("pointerdown", onPointerDown);
      host.removeEventListener("pointermove", onPointerMove);
      host.removeEventListener("pointerup", onPointerUp);
      host.removeEventListener("dblclick", onDblClick);
      finishDraftRef.current = null;
      clearDraft();
      clearMeasure();
    };
  }, [world]);

  /* ---------------- gizmo attach + transform baking -------------------- */
  const activeTool = useViewportStore((s) => s.activeTool);
  const selectedId = useViewportStore((s) => s.selectedId);
  const structVersion = useViewportStore((s) => s.structVersion);

  useEffect(() => {
    if (!world) {
      return;
    }
    const { gizmo } = world;
    const tool = getViewportState().activeTool as ToolId;
    const id = getViewportState().selectedId;
    const entity = id ? getViewportState().entities[id] : undefined;
    const nodes = id ? world.entityNodes.get(id) : undefined;

    const canTransform =
      nodes &&
      entity &&
      (tool === "move" ||
        (tool === "rotate" &&
          (entity.type === "region" ||
            entity.type === "tripwire" ||
            entity.type === "camera")) ||
        (tool === "scale" &&
          (entity.type === "region" || entity.type === "tripwire")));

    // Left-drag orbits only for navigation/transform tools; creation and
    // precise-select tools reserve it.
    const orbitLeftDrag =
      tool === "live" || tool === "move" || tool === "rotate" || tool === "scale";
    world.orbit.mouseButtons.LEFT = orbitLeftDrag ? MOUSE.ROTATE : null;

    if (!canTransform || !nodes || !entity) {
      gizmo.detach();
      return;
    }
    gizmo.setMode(tool === "move" ? "translate" : tool);
    // Regions/tripwires are floor-plan geometry: rotate about Z only, and
    // scale in the floor plane (+Z for volumetric height).
    const planarOnly =
      (tool === "rotate" || tool === "scale") &&
      (entity.type === "region" || entity.type === "tripwire");
    gizmo.showX = !(planarOnly && tool === "rotate");
    gizmo.showY = !(planarOnly && tool === "rotate");
    gizmo.showZ = true;
    gizmo.attach(nodes.gizmoTarget);

    const onDraggingChanged = (ev: { value: unknown }) => {
      if (ev.value) {
        getViewportState().commitHistory();
      } else {
        bakeTransform(world, entity.id, tool);
      }
    };
    gizmo.addEventListener("dragging-changed", onDraggingChanged);
    return () => {
      gizmo.removeEventListener("dragging-changed", onDraggingChanged);
      gizmo.detach();
      gizmo.showX = true;
      gizmo.showY = true;
      gizmo.showZ = true;
    };
  }, [world, activeTool, selectedId, structVersion]);

  /* ---------------- keyboard ------------------------------------------- */
  useEffect(() => {
    if (!world) {
      return;
    }
    const toolKeys: Record<string, ToolId> = {
      q: "select",
      w: "move",
      e: "rotate",
      r: "scale",
      g: "region",
      t: "tripwire",
      c: "camera",
      s: "sensor",
      m: "measure",
      l: "live",
    };
    const onKey = (ev: KeyboardEvent) => {
      const t = ev.target as HTMLElement | null;
      if (
        t &&
        (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)
      ) {
        return;
      }
      const s = getViewportState();
      const mod = ev.ctrlKey || ev.metaKey;
      if (mod && ev.key.toLowerCase() === "z") {
        ev.preventDefault();
        if (ev.shiftKey) {
          s.redo();
        } else {
          s.undo();
        }
        return;
      }
      if (mod && ev.key.toLowerCase() === "y") {
        ev.preventDefault();
        s.redo();
        return;
      }
      if (mod) {
        return;
      }
      const key = ev.key.toLowerCase();
      if (key === "escape") {
        if (draftRef.current) {
          const draft = draftRef.current;
          world.scene.remove(draft.preview);
          disposeGroup(draft.preview);
          draftRef.current = null;
        } else if (measureRef.current.visuals || measureRef.current.a) {
          const m = measureRef.current;
          if (m.visuals) {
            world.scene.remove(m.visuals);
            disposeGroup(m.visuals);
          }
          m.label?.remove();
          measureRef.current = { a: null, visuals: null, label: null };
        } else {
          s.select(null);
        }
        return;
      }
      if (key === "enter") {
        finishDraftRef.current?.();
        return;
      }
      if (key === "delete" || key === "backspace") {
        deleteSelection();
        return;
      }
      const tool = toolKeys[key];
      if (tool) {
        // Cancel any in-progress draft when switching tools.
        if (draftRef.current) {
          const draft = draftRef.current;
          world.scene.remove(draft.preview);
          disposeGroup(draft.preview);
          draftRef.current = null;
        }
        s.setTool(tool);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [world]);
}

/** Bake a finished gizmo drag into the entity + geometry model. */
function bakeTransform(
  world: ViewportWorld,
  id: string,
  tool: ToolId,
): void {
  const s = getViewportState();
  const e = s.entities[id];
  const nodes = world.entityNodes.get(id);
  if (!e || !nodes) {
    return;
  }
  world.gizmo.detach();
  const pivot = nodes.gizmoTarget;

  if (e.type === "region" || e.type === "tripwire") {
    const [cx, cy] = centroidOf(e.points);
    let newPoints: [number, number][];
    if (tool === "move") {
      const dx = pivot.position.x - cx;
      const dy = pivot.position.y - cy;
      newPoints = e.points.map(
        ([x, y]) => [x + dx, y + dy] as [number, number],
      );
    } else if (tool === "rotate") {
      const angle = pivot.rotation.z;
      newPoints = e.points.map(([x, y]) =>
        rotatePoint([x, y], [cx, cy], angle),
      );
    } else {
      const sx = pivot.scale.x;
      const sy = pivot.scale.y;
      newPoints = e.points.map(
        ([x, y]) =>
          [cx + (x - cx) * sx, cy + (y - cy) * sy] as [number, number],
      );
    }
    const { min, max } = bboxOf(newPoints);
    // Volumetric regions also scale in height.
    const heightScale =
      tool === "scale" && e.type === "region" && e.volumetric
        ? pivot.scale.z
        : 1;
    if (
      pointsEqual(newPoints, e.points) &&
      Math.abs(heightScale - 1) < 0.001
    ) {
      return;
    }
    if (e.type === "region") {
      const updated: RegionEntity = {
        ...e,
        points: newPoints,
        min,
        max,
        height:
          Number.isFinite(heightScale) && heightScale > 0.01
            ? e.height * heightScale
            : e.height,
      };
      s.upsertEntities([updated]);
      writeRegionToModel(updated);
    } else {
      const updated: TripwireEntity = {
        ...e,
        points: newPoints,
        a: newPoints[0],
        b: newPoints[newPoints.length - 1],
      };
      s.upsertEntities([updated]);
      writeTripwireToModel(updated);
    }
    return;
  }

  if (e.type === "camera") {
    if (tool === "move") {
      s.updateEntity(id, {
        position: [pivot.position.x, pivot.position.y, pivot.position.z],
      });
    } else if (tool === "rotate") {
      const eu = new Euler().setFromQuaternion(pivot.quaternion, "XYZ");
      s.updateEntity(id, {
        rotation: [r2d(eu.x), r2d(eu.y), r2d(eu.z)],
      });
    }
    // Visual-only in Phase 1.2; calibrated pose persistence ships with the
    // Phase 2.3 calibration wizard (REST camera PUT, dirty-gated).
    return;
  }

  if (e.type === "sensor" && tool === "move") {
    s.updateEntity(id, {
      position: [pivot.position.x, pivot.position.y, pivot.position.z],
    });
  }
}

/** Delete the selection (regions/tripwires also leave the geometry model). */
function deleteSelection(): void {
  const s = getViewportState();
  const id = s.selectedId;
  if (!id) {
    return;
  }
  const e = s.entities[id];
  if (!e) {
    return;
  }
  const isTempMarker = id.startsWith("draft-");
  if (e.type !== "region" && e.type !== "tripwire" && !isTempMarker) {
    return;
  }
  s.commitHistory();
  if (e.type === "region") {
    removeRegionFromModel(id);
  } else if (e.type === "tripwire") {
    removeTripwireFromModel(id);
  }
  s.removeEntity(id);
}

function disposeGroup(root: Group): void {
  root.traverse((o) => {
    const mesh = o as Mesh;
    const geo = (mesh as { geometry?: { dispose?: () => void } }).geometry;
    geo?.dispose?.();
    const mat = mesh.material as { dispose?: () => void } | undefined;
    mat?.dispose?.();
  });
}
