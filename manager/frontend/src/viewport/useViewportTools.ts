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
import { useCalibrateStore } from "./calibrate/calibrateStore";
import {
  addTripwireCurtain,
  lookAtOriginEuler,
  restPoseToRigEuler,
} from "./entities";
import {
  bboxOf,
  centroidOf,
  writeRegionToModel,
  writeTripwireToModel,
} from "./geometrySync";
import { scenePoseFromObject } from "../placement/poseThree";
import { getViewportState, useViewportStore } from "./store";
import { CREATION_TOOLS, entityCanScale } from "./tools";
import { attachShapeHandles, replaceShapePoints } from "./shapeHandles";
import type {
  CameraEntity,
  RegionEntity,
  SensorEntity,
  ToolId,
  TripwireEntity,
} from "./types";
import { SCENE_MAP_ENTITY_ID } from "./types";
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
/** World-meters snap distance to close a sensor polygon on the first vertex. */
const SENSOR_POLY_CLOSE_M = 0.45;

const r2d = (r: number) => (r * 180) / Math.PI;

/**
 * Floor hit under the cursor (z=0). Same NDC math as calibrate
 * `pickScenePoint` — keep them identical so tool clicks match wizard clicks.
 */
function floorPoint(
  world: ViewportWorld,
  ev: PointerEvent,
): [number, number] | null {
  const rect = world.host.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    return null;
  }
  const ndc = new Vector2(
    ((ev.clientX - rect.left) / rect.width) * 2 - 1,
    -(((ev.clientY - rect.top) / rect.height) * 2 - 1),
  );
  const raycaster = new Raycaster();
  raycaster.setFromCamera(ndc, world.getActiveCamera());
  const hit = new Vector3();
  if (!raycaster.ray.intersectPlane(FLOOR, hit)) {
    return null;
  }
  return [hit.x, hit.y];
}

function pickEntity(world: ViewportWorld, ev: PointerEvent): string | null {
  const rect = world.host.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    return null;
  }
  const ndc = new Vector2(
    ((ev.clientX - rect.left) / rect.width) * 2 - 1,
    -(((ev.clientY - rect.top) / rect.height) * 2 - 1),
  );
  const raycaster = new Raycaster();
  raycaster.setFromCamera(ndc, world.getActiveCamera());
  const hits = raycaster.intersectObject(world.entityRoot, true);
  for (const hit of hits) {
    let o: object | null = hit.object;
    while (o) {
      const id = (o as { userData?: { entityId?: string } }).userData?.entityId;
      if (typeof id === "string") {
        return id;
      }
      o = (o as { parent?: object }).parent ?? null;
    }
  }
  // Scene map GLB lives on mapRoot, not entityRoot.
  if (world.sceneMesh) {
    const mapHits = raycaster.intersectObject(world.sceneMesh, true);
    if (mapHits.length > 0) {
      const map = getViewportState().entities[SCENE_MAP_ENTITY_ID];
      if (map?.type === "map" && map.visible !== false) {
        return SCENE_MAP_ENTITY_ID;
      }
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
  if (rect.width <= 0 || rect.height <= 0) {
    return null;
  }
  const ndc = new Vector2(
    ((ev.clientX - rect.left) / rect.width) * 2 - 1,
    -(((ev.clientY - rect.top) / rect.height) * 2 - 1),
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

function pointsEqual(a: [number, number][], b: [number, number][]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  return a.every(
    ([x, y], i) =>
      Math.abs(x - b[i][0]) < POINT_EPS && Math.abs(y - b[i][1]) < POINT_EPS,
  );
}

interface Draft {
  tool: "region" | "tripwire" | "sensor-poly";
  /** Sensor entity id when `tool` is sensor-poly. */
  sensorId?: string;
  points: [number, number][];
  preview: Group;
  cursor: [number, number] | null;
  lastPointAt: number;
}

/** Selected poly sensor waiting for (or mid-draw) floor vertices. */
function sensorNeedsPolyDraw(): SensorEntity | null {
  const s = getViewportState();
  const e = s.selectedId ? s.entities[s.selectedId] : undefined;
  if (!e || e.type !== "sensor" || e.areaMode !== "poly") {
    return null;
  }
  if (e.points.length >= 3) {
    return null;
  }
  return e;
}

export function useViewportTools(world: ViewportWorld | null): void {
  useEffect(() => world ? attachShapeHandles(world) : undefined, [world]);
  const draftRef = useRef<Draft | null>(null);
  const finishDraftRef = useRef<(() => void) | null>(null);
  const measureRef = useRef<{
    /** First floor point (set on first click). */
    a: [number, number] | null;
    /** Locked end after the second click; null while rubber-banding. */
    b: [number, number] | null;
    /** Live cursor end while drawing (like tripwire draft.cursor). */
    cursor: [number, number] | null;
    done: boolean;
    visuals: Group | null;
    label: HTMLDivElement | null;
  }>({
    a: null,
    b: null,
    cursor: null,
    done: false,
    visuals: null,
    label: null,
  });
  const clearMeasureRef = useRef<() => void>(() => undefined);
  const downRef = useRef<{
    x: number;
    y: number;
    at: number;
    tool: ToolId;
  } | null>(null);
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
      }
      m.label?.remove();
      measureRef.current = {
        a: null,
        b: null,
        cursor: null,
        done: false,
        visuals: null,
        label: null,
      };
    };
    clearMeasureRef.current = clearMeasure;

    const refreshPreview = () => {
      const draft = draftRef.current;
      if (!draft) {
        return;
      }
      world.scene.remove(draft.preview);
      disposeGroup(draft.preview);
      const g = new Group();
      const sensor = draft.sensorId ? getViewportState().entities[draft.sensorId] : undefined;
      const color =
        draft.tool === "region"
          ? 0x30d158
          : draft.tool === "sensor-poly"
            ? sensor?.type === "sensor" ? sensor.color : 0xbf5af2
            : 0xff9f0a;
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
      if (draft.tool === "tripwire" && trail.length >= 2) {
        addTripwireCurtain(g, trail, color, { showLabel: false });
      } else if (trail.length >= 2) {
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

    const startDraft = (
      tool: "region" | "tripwire" | "sensor-poly",
      sensorId?: string,
    ) => {
      clearDraft();
      draftRef.current = {
        tool,
        sensorId,
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
        (draft.tool === "tripwire" && pts.length >= 2) ||
        (draft.tool === "sensor-poly" && pts.length >= 3);
      const sensorId = draft.sensorId;
      clearDraft();
      if (!valid) {
        return;
      }
      const s = getViewportState();
      if (draft.tool === "sensor-poly") {
        if (!sensorId) {
          return;
        }
        const cur = s.entities[sensorId];
        if (!cur || cur.type !== "sensor") {
          return;
        }
        s.commitHistory();
        s.upsertEntities([replaceShapePoints(cur, pts)]);
        s.select(sensorId);
        return;
      }
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
        s.beginEditing(id);
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
        s.beginEditing(id);
      }
    };

    const addSensorPolyPoint = (pt: [number, number]) => {
      const sensor = sensorNeedsPolyDraw();
      if (!sensor) {
        return;
      }
      let draft = draftRef.current;
      if (
        !draft ||
        draft.tool !== "sensor-poly" ||
        draft.sensorId !== sensor.id
      ) {
        startDraft("sensor-poly", sensor.id);
        draft = draftRef.current;
      }
      if (!draft) {
        return;
      }
      // Close by clicking near the first vertex (same as 2D SensorAreaMap).
      if (draft.points.length >= 3) {
        const first = draft.points[0];
        if (
          Math.hypot(first[0] - pt[0], first[1] - pt[1]) < SENSOR_POLY_CLOSE_M
        ) {
          finishDraft();
          return;
        }
      }
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
        restRotation: [0, 0, 0],
        posed: false,
        fov: 60,
        color: "#0a84ff",
      };
      s.commitHistory();
      s.upsertEntities([entity]);
      // Stay on Move so the gizmo remains available for pose editing.
      // Region/tripwire/sensor creation still uses beginEditing → Live.
      useViewportStore.setState((state) => ({
        selectedId: id,
        activeTool: "move",
        editFocus: state.editFocus + 1,
      }));
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
        areaMode: "circle",
        radius: 5,
        points: [],
        singletonType: "environmental",
        color: "#bf5af2",
      };
      s.commitHistory();
      s.upsertEntities([entity]);
      s.select(id);
    };

    const placeMeasureLabel = (
      label: HTMLDivElement,
      ax: number,
      ay: number,
      bx: number,
      by: number,
      dist: number,
    ) => {
      label.textContent = `${dist.toFixed(2)} m`;
      const v = new Vector3((ax + bx) / 2, (ay + by) / 2, 0.4).project(
        world.getActiveCamera(),
      );
      const w = world.host.clientWidth;
      const h = world.host.clientHeight;
      label.style.transform = `translate(-50%, -100%) translate(${(v.x * 0.5 + 0.5) * w}px, ${(-v.y * 0.5 + 0.5) * h}px)`;
    };

    /** Rebuild measure mesh + label from a → (b | cursor). */
    const refreshMeasure = () => {
      const m = measureRef.current;
      if (!m.a) {
        return;
      }
      const [ax, ay] = m.a;
      const end = m.done && m.b ? m.b : (m.cursor ?? m.a);
      const [bx, by] = end;
      const dist = Math.hypot(bx - ax, by - ay);

      if (m.visuals) {
        world.scene.remove(m.visuals);
        disposeGroup(m.visuals);
      }
      const g = new Group();
      const mat = new MeshBasicMaterial({ color: 0x0a84ff });
      const startDot = new Mesh(new SphereGeometry(0.12, 10, 8), mat);
      startDot.position.set(ax, ay, 0.1);
      g.add(startDot);
      if (dist > POINT_EPS) {
        g.add(
          new Line(
            new BufferGeometry().setFromPoints([
              new Vector3(ax, ay, 0.1),
              new Vector3(bx, by, 0.1),
            ]),
            new LineBasicMaterial({ color: 0x0a84ff }),
          ),
        );
        const endDot = new Mesh(
          new SphereGeometry(0.12, 10, 8),
          mat.clone(),
        );
        endDot.position.set(bx, by, 0.1);
        g.add(endDot);
      }
      world.scene.add(g);
      m.visuals = g;

      if (!m.label) {
        const label = document.createElement("div");
        label.className = "ss-vlabel ss-vlabel--measure";
        world.labelLayer.appendChild(label);
        m.label = label;
      }
      placeMeasureLabel(m.label, ax, ay, bx, by, dist);
    };

    const measureClick = (pt: [number, number]) => {
      const m = measureRef.current;
      // Finished segment: next click starts a new measurement.
      if (m.done) {
        clearMeasure();
      }
      const cur = measureRef.current;
      if (!cur.a) {
        cur.a = pt;
        cur.cursor = pt;
        cur.b = null;
        cur.done = false;
        refreshMeasure();
        return;
      }
      cur.b = pt;
      cur.cursor = pt;
      cur.done = true;
      refreshMeasure();
    };

    const onPointerDown = (ev: PointerEvent) => {
      if (useCalibrateStore.getState().active) {
        return;
      }
      if (ev.button !== 0) {
        return;
      }
      downRef.current = {
        x: ev.clientX,
        y: ev.clientY,
        at: performance.now(),
        tool: getViewportState().activeTool,
      };
      const tool = getViewportState().activeTool;
      // Sensor polygon drawing uses click (pointerup) so Ortho pan still works.
      if (sensorNeedsPolyDraw() && !CREATION_TOOLS.includes(tool)) {
        downPickRef.current = pickTarget(world, ev);
        return;
      }
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
      if (useCalibrateStore.getState().active) {
        return;
      }
      const draft = draftRef.current;
      const pt = floorPoint(world, ev);
      if (draft && pt) {
        draft.cursor = pt;
        refreshPreview();
      }
      // Rubber-band the open measure segment (tripwire-style).
      const m = measureRef.current;
      if (m.a && !m.done && pt) {
        m.cursor = pt;
        refreshMeasure();
      }
    };

    const onPointerUp = (ev: PointerEvent) => {
      if (useCalibrateStore.getState().active) {
        downRef.current = null;
        downPickRef.current = null;
        return;
      }
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
      // Use the tool from pointerdown. Placing a camera switches to Live
      // before pointerup; a live click would otherwise replace that selection.
      if (CREATION_TOOLS.includes(down.tool)) {
        return;
      }
      // Incomplete poly sensor: floor clicks add / close vertices.
      if (sensorNeedsPolyDraw()) {
        const pt = floorPoint(world, ev);
        if (pt) {
          addSensorPolyPoint(pt);
        }
        return;
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
      if (useCalibrateStore.getState().active) {
        return;
      }
      const tool = getViewportState().activeTool;
      const draft = draftRef.current;
      if (
        tool === "region" ||
        tool === "tripwire" ||
        draft?.tool === "sensor-poly"
      ) {
        ev.preventDefault();
        finishDraft();
      }
    };

    // Capture phase (same as calibrate): OrbitControls listens on the canvas
    // and would otherwise pan/orbit on bubble before we sample the camera,
    // shifting floor picks away from the click. Capture runs first.
    host.addEventListener("pointerdown", onPointerDown, true);
    host.addEventListener("pointermove", onPointerMove, true);
    host.addEventListener("pointerup", onPointerUp, true);
    host.addEventListener("dblclick", onDblClick, true);
    finishDraftRef.current = finishDraft;
    return () => {
      host.removeEventListener("pointerdown", onPointerDown, true);
      host.removeEventListener("pointermove", onPointerMove, true);
      host.removeEventListener("pointerup", onPointerUp, true);
      host.removeEventListener("dblclick", onDblClick, true);
      finishDraftRef.current = null;
      clearDraft();
      clearMeasure();
    };
  }, [world]);

  /* ---------------- gizmo attach + transform baking -------------------- */
  const activeTool = useViewportStore((s) => s.activeTool);
  const selectedId = useViewportStore((s) => s.selectedId);
  const structVersion = useViewportStore((s) => s.structVersion);
  const selectedSensorPolyPending = useViewportStore((s) => {
    const e = s.selectedId ? s.entities[s.selectedId] : undefined;
    return (
      !!e && e.type === "sensor" && e.areaMode === "poly" && e.points.length < 3
    );
  });
  // Calibrate uses double-click to add points; left-drag must still orbit.
  const calActive = useCalibrateStore((s) => s.active);

  // Leaving Measure dismisses the overlay (Esc / Delete also clear).
  useEffect(() => {
    if (activeTool !== "measure") {
      clearMeasureRef.current();
    }
  }, [activeTool]);

  useEffect(() => {
    if (!world) {
      return;
    }
    const { gizmo } = world;
    const tool = getViewportState().activeTool as ToolId;
    const id = getViewportState().selectedId;
    const entity = id ? getViewportState().entities[id] : undefined;
    const nodes = id ? world.entityNodes.get(id) : undefined;
    const drawingSensorPoly = selectedSensorPolyPending;

    const canTransform =
      !calActive &&
      nodes &&
      entity &&
      !drawingSensorPoly &&
      (tool === "move" ||
        (tool === "rotate" &&
          (entity.type === "region" ||
            entity.type === "tripwire" ||
            entity.type === "camera" ||
            entity.type === "map")) ||
        (tool === "scale" && entityCanScale(entity)));

    // Left-drag orbits/pans for navigation tools. During floor drawing
    // (region / tripwire / measure / sensor poly), disable LEFT so Ortho pan
    // or Persp orbit cannot move the camera before the floor pick — that was
    // shifting ROI/tripwire/measure points off the click (calibrate avoids
    // this via capture + no left-drag on markers). Middle/right still navigate.
    const drawingFloor = drawingSensorPoly || CREATION_TOOLS.includes(tool);
    const orbitLeftDrag =
      calActive || (!drawingFloor &&
      (
        tool === "live" ||
        tool === "move" ||
        tool === "rotate" ||
        tool === "scale"));
    if (orbitLeftDrag) {
      world.orbit.mouseButtons.LEFT = world.orbit.enableRotate
        ? MOUSE.ROTATE
        : MOUSE.PAN;
    } else {
      world.orbit.mouseButtons.LEFT = null;
    }

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
  }, [
    world,
    activeTool,
    selectedId,
    structVersion,
    calActive,
    selectedSensorPolyPending,
  ]);

  // Drop an in-progress sensor polygon draft when selection / mode changes.
  useEffect(() => {
    if (!world) {
      return;
    }
    const draft = draftRef.current;
    if (!draft || draft.tool !== "sensor-poly") {
      return;
    }
    if (!selectedSensorPolyPending || draft.sensorId !== selectedId) {
      world.scene.remove(draft.preview);
      disposeGroup(draft.preview);
      draftRef.current = null;
    }
  }, [world, selectedId, selectedSensorPolyPending]);

  // Scale is only meaningful for a region or tripwire. Leave the tool when
  // the selection cannot be scaled so the button does not stay pressed.
  useEffect(() => {
    if (activeTool !== "scale") {
      return;
    }
    const id = getViewportState().selectedId;
    const entity = id ? getViewportState().entities[id] : undefined;
    if (!entityCanScale(entity)) {
      getViewportState().setTool("live");
    }
  }, [activeTool, selectedId, structVersion]);

  /* ---------------- keyboard ------------------------------------------- */
  useEffect(() => {
    if (!world) {
      return;
    }
    const toolKeys: Record<string, ToolId> = {
      // Q kept as alias for Live (former Select tool merged into Live).
      q: "live",
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
      if (useCalibrateStore.getState().active) {
        return;
      }
      const t = ev.target as HTMLElement | null;
      if (
        t &&
        (t.tagName === "INPUT" ||
          t.tagName === "TEXTAREA" ||
          t.isContentEditable)
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
        } else if (
          measureRef.current.a ||
          measureRef.current.visuals ||
          measureRef.current.done
        ) {
          clearMeasureRef.current();
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
        if (
          measureRef.current.a ||
          measureRef.current.visuals ||
          measureRef.current.done
        ) {
          clearMeasureRef.current();
          return;
        }
        deleteSelection();
        return;
      }
      const tool = toolKeys[key];
      if (tool) {
        if (
          tool === "scale" &&
          !entityCanScale(s.selectedId ? s.entities[s.selectedId] : undefined)
        ) {
          return;
        }
        // Cancel any in-progress draft when switching tools.
        if (draftRef.current) {
          const draft = draftRef.current;
          world.scene.remove(draft.preview);
          disposeGroup(draft.preview);
          draftRef.current = null;
        }
        if (tool !== "measure") {
          clearMeasureRef.current();
        }
        s.setTool(tool);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [world]);
}

/** Bake a finished gizmo drag into the entity + geometry model. */
function bakeTransform(world: ViewportWorld, id: string, tool: ToolId): void {
  const s = getViewportState();
  const e = s.entities[id];
  const nodes = world.entityNodes.get(id);
  if (!e || !nodes) {
    return;
  }
  const pivot = nodes.gizmoTarget;

  if (e.type === "region" || e.type === "tripwire") {
    // Detach before struct rebuild — pose-only camera/sensor bakes keep the
    // gizmo attached so transform mode persists across drags.
    world.gizmo.detach();
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
    if (pointsEqual(newPoints, e.points) && Math.abs(heightScale - 1) < 0.001) {
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
        posed: true,
      });
    } else if (tool === "rotate") {
      const eu = new Euler().setFromQuaternion(pivot.quaternion, "XYZ");
      const rotation: [number, number, number] = [
        r2d(eu.x),
        r2d(eu.y),
        r2d(eu.z),
      ];
      s.updateEntity(id, {
        rotation,
        restRotation: restPoseToRigEuler(rotation),
        posed: true,
      });
    }
    return;
  }

  if (e.type === "sensor" && tool === "move") {
    s.updateEntity(id, {
      position: [pivot.position.x, pivot.position.y, pivot.position.z],
    });
    return;
  }

  if (e.type === "map") {
    try {
      const pose = scenePoseFromObject(pivot);
      s.updateEntity(id, {
        position: pose.translation,
        rotation: pose.rotation,
        scale: pose.scale,
      });
    } catch {
      /* degenerate matrix — leave entity unchanged */
    }
  }
}

/**
 * Ask the Outliner (or any listener) to run the shared delete confirm flow.
 * Keyboard Delete/Backspace and Outliner row trash use the same path.
 */
function requestEntityDelete(id: string): void {
  window.dispatchEvent(
    new CustomEvent("ss-request-entity-delete", { detail: { id } }),
  );
}

/** Delete key: open the shared delete flow for the current selection. */
function deleteSelection(): void {
  const id = getViewportState().selectedId;
  if (!id) {
    return;
  }
  const e = getViewportState().entities[id];
  if (!e) {
    return;
  }
  if (e.type === "mark") {
    return;
  }
  requestEntityDelete(id);
}

function disposeGroup(root: Group): void {
  root.traverse((o) => {
    const mesh = o as Mesh;
    const geo = (mesh as { geometry?: { dispose?: () => void } }).geometry;
    geo?.dispose?.();
    const mat = mesh.material as
      | { dispose?: () => void; map?: { dispose?: () => void } }
      | { dispose?: () => void; map?: { dispose?: () => void } }[]
      | undefined;
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
