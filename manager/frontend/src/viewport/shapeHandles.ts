// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
  CircleGeometry,
  DoubleSide,
  Group,
  Mesh,
  MeshBasicMaterial,
  Plane,
  Raycaster,
  RingGeometry,
  Vector2,
  Vector3,
} from "three";
import { useCalibrateStore } from "./calibrate/calibrateStore";
import { bboxOf, writeRegionToModel, writeTripwireToModel } from "./geometrySync";
import { getViewportState, useViewportStore } from "./store";
import { CREATION_TOOLS } from "./tools";
import type { RegionEntity, TripwireEntity } from "./types";
import type { ViewportWorld } from "./world";

type Shape = RegionEntity | TripwireEntity;
const HANDLE_Z = 0.04;
const HANDLE_RADIUS_PX = 7;
const PICK_RADIUS_PX = 11;

export function moveShapeVertex(shape: Shape, index: number, point: [number, number]): Shape {
  if (index < 0 || index >= shape.points.length || !point.every(Number.isFinite)) {
    return shape;
  }
  const points = shape.points.map(([x, y], pointIndex): [number, number] =>
    pointIndex === index ? [point[0], point[1]] : [x, y],
  );
  return shape.type === "region"
    ? { ...shape, points, ...bboxOf(points) }
    : { ...shape, points, a: points[0], b: points[points.length - 1] };
}

export function createShapeHandle(color: string): Group {
  const handle = new Group();
  handle.name = "shape-vertex-handle";
  const fill = new Mesh(new CircleGeometry(0.72, 32), new MeshBasicMaterial({
    color,
    side: DoubleSide,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  }));
  const ring = new Mesh(new RingGeometry(0.72, 1, 32), new MeshBasicMaterial({
    color: "#ffffff",
    side: DoubleSide,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  }));
  fill.renderOrder = 8;
  ring.renderOrder = 9;
  handle.add(fill, ring);
  return handle;
}

function disposeHandles(root: Group): void {
  root.traverse((object) => {
    if (object instanceof Mesh) {
      object.geometry.dispose();
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      materials.forEach((material) => material.dispose());
    }
  });
  root.clear();
}

export function attachShapeHandles(world: ViewportWorld): () => void {
  const { host, orbit } = world;
  const root = new Group();
  root.name = "shape-vertex-handles";
  world.scene.add(root);
  const priorCursor = host.style.cursor;
  let ownsCursor = false;
  let shape: Shape | null = null;
  let previousShape: Shape | null = null;
  let previousEnabled = false;
  type History = ReturnType<typeof getViewportState>;
  let drag: {
    original: Shape;
    index: number;
    pointerId: number;
    orbitEnabled: boolean;
    past: History["past"] | null;
    future: History["future"] | null;
  } | null = null;

  const resetCursor = () => {
    if (ownsCursor) {
      host.style.cursor = priorCursor;
      ownsCursor = false;
    }
  };
  const cursor = (value: string) => {
    host.style.cursor = value;
    ownsCursor = true;
  };
  const publish = (updated: Shape) => {
    getViewportState().upsertEntities([updated]);
    if (updated.type === "region") {
      writeRegionToModel(updated);
    } else {
      writeTripwireToModel(updated);
    }
  };
  const endDrag = (cancel: boolean) => {
    const current = drag;
    if (!current) {
      return;
    }
    drag = null;
    orbit.enabled = current.orbitEnabled;
    if (host.hasPointerCapture(current.pointerId)) {
      host.releasePointerCapture(current.pointerId);
    }
    if (cancel && current.past && getViewportState().entities[current.original.id]) {
      publish(current.original);
      if (getViewportState().past.length === current.past.length + 1) {
        useViewportStore.setState({ past: current.past, future: current.future! });
      }
    }
    resetCursor();
  };

  const layout = () => {
    if (!shape) {
      return;
    }
    const camera = world.getActiveCamera();
    camera.updateMatrixWorld();
    const height = host.getBoundingClientRect().height;
    if (height <= 0) {
      return;
    }
    root.children.forEach((handle, index) => {
      const point = shape!.points[index];
      if (!point) {
        return;
      }
      handle.position.set(point[0], point[1], HANDLE_Z);
      const cameraPoint = handle.position.clone().applyMatrix4(camera.matrixWorldInverse);
      const perspective = "isPerspectiveCamera" in camera && camera.isPerspectiveCamera;
      const depth = perspective ? Math.max(-cameraPoint.z, 0.01) : 1;
      const radius = HANDLE_RADIUS_PX * 2 * depth / (Math.abs(camera.projectionMatrix.elements[5]) * height);
      handle.scale.setScalar(radius);
    });
  };
  const sync = () => {
    const state = getViewportState();
    const selected = state.selectedId ? state.entities[state.selectedId] : undefined;
    const next = selected?.type === "region" || selected?.type === "tripwire" ? selected : null;
    const enabled = !!next && next.visible !== false &&
      !useCalibrateStore.getState().active && !CREATION_TOOLS.includes(state.activeTool);
    if (next === previousShape && enabled === previousEnabled) {
      return;
    }
    previousShape = next;
    previousEnabled = enabled;
    if (drag && (!enabled || next?.id !== drag.original.id)) {
      endDrag(true);
    }
    shape = enabled ? next : null;
    disposeHandles(root);
    if (shape) {
      shape.points.forEach(() => root.add(createShapeHandle(shape!.color)));
      layout();
    } else {
      resetCursor();
    }
  };
  const nearest = (event: PointerEvent): number => {
    if (!shape || world.gizmo.dragging) {
      return -1;
    }
    const rect = host.getBoundingClientRect();
    let best = -1;
    let distance = PICK_RADIUS_PX;
    root.children.forEach((handle, index) => {
      const point = handle.position.clone().project(world.getActiveCamera());
      if (point.z < -1 || point.z > 1) {
        return;
      }
      const dx = (point.x + 1) * rect.width / 2 + rect.left - event.clientX;
      const dy = (1 - point.y) * rect.height / 2 + rect.top - event.clientY;
      const pixels = Math.hypot(dx, dy);
      if (pixels < distance) {
        best = index;
        distance = pixels;
      }
    });
    return best;
  };
  const stop = (event: Event) => {
    event.preventDefault();
    event.stopImmediatePropagation();
  };
  const onDown = (event: PointerEvent) => {
    if (event.button !== 0 || drag) {
      return;
    }
    const index = nearest(event);
    if (index < 0 || !shape) {
      return;
    }
    drag = { original: shape, index, pointerId: event.pointerId, orbitEnabled: orbit.enabled, past: null, future: null };
    orbit.enabled = false;
    host.setPointerCapture(event.pointerId);
    cursor("grabbing");
    stop(event);
  };
  const onMove = (event: PointerEvent) => {
    if (!drag) {
      if (event.buttons === 0 && nearest(event) >= 0) {
        cursor("crosshair");
      } else {
        resetCursor();
      }
      return;
    }
    if (event.pointerId !== drag.pointerId) {
      return;
    }
    stop(event);
    const rect = host.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return;
    }
    const ray = new Raycaster();
    ray.setFromCamera(new Vector2(
      (event.clientX - rect.left) / rect.width * 2 - 1,
      1 - (event.clientY - rect.top) / rect.height * 2,
    ), world.getActiveCamera());
    const hit = ray.ray.intersectPlane(new Plane(new Vector3(0, 0, 1), 0), new Vector3());
    const selected = getViewportState().entities[drag.original.id];
    if (!hit || (selected?.type !== "region" && selected?.type !== "tripwire")) {
      return;
    }
    const point = selected.points[drag.index];
    if (Math.hypot(point[0] - hit.x, point[1] - hit.y) < 0.001) {
      return;
    }
    if (!drag.past) {
      drag.past = getViewportState().past;
      drag.future = getViewportState().future;
      getViewportState().commitHistory();
    }
    publish(moveShapeVertex(selected, drag.index, [hit.x, hit.y]));
  };
  const onUp = (event: PointerEvent) => {
    if (!drag || event.pointerId !== drag.pointerId) {
      return;
    }
    stop(event);
    endDrag(event.type !== "pointerup");
    if (nearest(event) >= 0) {
      cursor("crosshair");
    }
  };
  const onLeave = () => { if (!drag) resetCursor(); };
  const onKey = (event: KeyboardEvent) => {
    if (drag && event.key === "Escape") {
      stop(event);
      endDrag(true);
    }
  };
  const unsubscribe = useViewportStore.subscribe(sync);
  const unsubscribeCalibration = useCalibrateStore.subscribe(sync);
  host.addEventListener("pointerdown", onDown, true);
  host.addEventListener("pointermove", onMove, true);
  host.addEventListener("pointerup", onUp, true);
  host.addEventListener("pointercancel", onUp, true);
  host.addEventListener("lostpointercapture", onUp, true);
  host.addEventListener("pointerleave", onLeave);
  window.addEventListener("keydown", onKey, true);
  world.onFrame.add(layout);
  sync();
  return () => {
    unsubscribe();
    unsubscribeCalibration();
    endDrag(true);
    host.removeEventListener("pointerdown", onDown, true);
    host.removeEventListener("pointermove", onMove, true);
    host.removeEventListener("pointerup", onUp, true);
    host.removeEventListener("pointercancel", onUp, true);
    host.removeEventListener("lostpointercapture", onUp, true);
    host.removeEventListener("pointerleave", onLeave);
    window.removeEventListener("keydown", onKey, true);
    world.onFrame.delete(layout);
    world.scene.remove(root);
    disposeHandles(root);
    resetCursor();
  };
}