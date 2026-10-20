// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CircleGeometry, Mesh, MeshBasicMaterial, PerspectiveCamera, Scene, Vector3 } from "three";
import { useCalibrateStore } from "./calibrate/calibrateStore";
import { resetGeometryModel } from "../scene/map/geometryModel";
import { seedEntitiesFromGeometry, writeRegionToModel, writeTripwireToModel } from "./geometrySync";
import { attachShapeHandles, createShapeHandle, moveShapeVertex } from "./shapeHandles";
import { getViewportState, useViewportStore } from "./store";
import type { RegionEntity, SensorEntity, TripwireEntity } from "./types";
import type { ViewportWorld } from "./world";

function region(): RegionEntity {
  return {
    id: "region-1", type: "region", name: "Region 1", visible: true,
    points: [[0, 0], [4, 0], [4, 3], [0, 3]], min: [0, 0], max: [4, 3],
    color: "#30d158", height: 2, volumetric: false, bufferSize: 0.5,
  };
}

function tripwire(): TripwireEntity {
  return {
    id: "tripwire-1", type: "tripwire", name: "Tripwire 1", visible: true,
    points: [[0, 0], [2, 1], [4, 3]], a: [0, 0], b: [4, 3], color: "#ff9f0a",
  };
}

function sensor(): SensorEntity {
  return {
    id: "sensor-1", type: "sensor", name: "Sensor 1", visible: true,
    position: [20, 30, 2], areaMode: "poly", radius: 5,
    points: [[0, 0], [4, 0], [4, 3], [0, 3]],
    singletonType: "environmental", color: "#bf5af2",
  };
}

class FakeHost extends EventTarget {
  style = { cursor: "" };
  captures = new Set<number>();
  getBoundingClientRect() { return { left: 0, top: 0, width: 600, height: 600 }; }
  setPointerCapture(id: number) { this.captures.add(id); }
  hasPointerCapture(id: number) { return this.captures.has(id); }
  releasePointerCapture(id: number) { this.captures.delete(id); }
}

let cleanup: (() => void) | undefined;

beforeEach(() => {
  vi.stubGlobal("document", { getElementById: () => null });
  vi.stubGlobal("window", new EventTarget());
  vi.stubGlobal("CustomEvent", class extends Event {
    detail: unknown;
    constructor(type: string, options: { detail: unknown }) {
      super(type);
      this.detail = options.detail;
    }
  });
  resetGeometryModel();
  getViewportState().clearEntities();
  useViewportStore.setState({ activeTool: "live" });
  useCalibrateStore.setState({ active: false });
});

afterEach(() => {
  cleanup?.();
  cleanup = undefined;
  vi.unstubAllGlobals();
});

function harness(shape: RegionEntity | TripwireEntity | SensorEntity = region()) {
  const host = new FakeHost();
  const scene = new Scene();
  const camera = new PerspectiveCamera(60, 1, 0.1, 1000);
  camera.up.set(0, 0, 1);
  camera.position.set(4, -10, 12);
  camera.lookAt(2, 1, 0);
  camera.updateMatrixWorld(true);
  const orbit = { enabled: true };
  const world = {
    host, scene, orbit, gizmo: { dragging: false }, onFrame: new Set<() => void>(),
    getActiveCamera: () => camera,
  } as unknown as ViewportWorld;
  getViewportState().upsertEntities([shape]);
  if (shape.type === "region") writeRegionToModel(shape);
  else if (shape.type === "tripwire") writeTripwireToModel(shape);
  getViewportState().select(shape.id);
  cleanup = attachShapeHandles(world);
  const pointer = (type: string, point: [number, number], buttons = 0) => {
    const projected = new Vector3(point[0], point[1], 0.04).project(camera);
    const event = new Event(type, { cancelable: true });
    Object.assign(event, {
      clientX: (projected.x + 1) * 300, clientY: (1 - projected.y) * 300,
      pointerId: 1, button: 0, buttons,
    });
    host.dispatchEvent(event);
  };
  return { host, scene, camera, orbit, world, pointer };
}

describe("floor shape handles", () => {
  it("uses circular floor geometry rendered above translucent objects", () => {
    const handle = createShapeHandle("#30d158");
    const fill = handle.children[0] as Mesh;
    expect(fill.geometry).toBeInstanceOf(CircleGeometry);
    const material = fill.material as MeshBasicMaterial;
    expect(material.transparent).toBe(true);
    expect(material.depthWrite).toBe(false);
    expect(material.depthTest).toBe(false);
    expect(fill.renderOrder).toBeGreaterThan(4);
    handle.children.forEach((child) => {
      const mesh = child as Mesh;
      mesh.geometry.dispose();
      (mesh.material as MeshBasicMaterial).dispose();
    });
  });

  it("updates region bounds without changing its identity or settings", () => {
    const original = region();
    const updated = moveShapeVertex(original, 0, [-2, -1]) as RegionEntity;
    expect(updated.points[0]).toEqual([-2, -1]);
    expect(updated.min).toEqual([-2, -1]);
    expect(updated.max).toEqual([4, 3]);
    expect(updated.id).toBe(original.id);
    expect(updated.height).toBe(original.height);
    expect(original.points[0]).toEqual([0, 0]);
  });

  it("updates tripwire endpoints while preserving intermediate points and direction", () => {
    const updated = moveShapeVertex(tripwire(), 0, [-2, 0]) as TripwireEntity;
    expect(updated.a).toEqual([-2, 0]);
    expect(updated.b).toEqual([4, 3]);
    expect(updated.points[1]).toEqual([2, 1]);
    const last = moveShapeVertex(updated, 2, [6, 4]) as TripwireEntity;
    expect(last.b).toEqual([6, 4]);
  });

  it("rejects invalid point indexes or non-finite coordinates", () => {
    const original = region();
    expect(moveShapeVertex(original, -1, [1, 2])).toBe(original);
    expect(moveShapeVertex(original, 4, [1, 2])).toBe(original);
    expect(moveShapeVertex(original, 0, [NaN, 2])).toBe(original);
  });

  it.each([region, tripwire, sensor])("drags vertices with crosshair/grabbing cursors and one undo step", (makeShape) => {
    const original = makeShape();
    const { host, pointer, orbit } = harness(original);
    pointer("pointermove", [0, 0]);
    expect(host.style.cursor).toBe("crosshair");
    pointer("pointerdown", [0, 0], 1);
    expect(host.style.cursor).toBe("grabbing");
    expect(orbit.enabled).toBe(false);
    expect(host.hasPointerCapture(1)).toBe(true);
    pointer("pointermove", [-1, -1], 1);
    pointer("pointermove", [-2, -1], 1);
    pointer("pointerup", [-2, -1]);
    const updated = getViewportState().entities[original.id] as typeof original;
    expect(updated.points[0][0]).toBeLessThan(-1.9);
    expect(updated.points[1]).toEqual(original.points[1]);
    if (updated.type === "sensor") expect(updated.position).toEqual([20, 30, 2]);
    expect(getViewportState().past).toHaveLength(1);
    expect(orbit.enabled).toBe(true);
    expect(host.hasPointerCapture(1)).toBe(false);
    if (updated.type !== "sensor") {
      const model = seedEntitiesFromGeometry();
      expect([...model.regions, ...model.tripwires].find(item => item.id === original.id)?.points).toEqual(updated.points);
    }
    getViewportState().undo();
    expect((getViewportState().entities[original.id] as typeof original).points).toEqual(original.points);
  });

  it.each(["pointercancel", "Escape"])("cancels a drag with %s", (cancel) => {
    const original = region();
    const { pointer, orbit } = harness(original);
    pointer("pointerdown", [0, 0], 1);
    pointer("pointermove", [-2, -1], 1);
    if (cancel === "Escape") {
      const event = new Event("keydown", { cancelable: true });
      Object.assign(event, { key: "Escape" });
      window.dispatchEvent(event);
    } else pointer(cancel, [-2, -1]);
    expect((getViewportState().entities[original.id] as RegionEntity).points).toEqual(original.points);
    expect(getViewportState().past).toHaveLength(0);
    expect(orbit.enabled).toBe(true);
  });

  it("does not create undo history when a handle is clicked without moving", () => {
    const { pointer } = harness();
    pointer("pointerdown", [0, 0], 1);
    pointer("pointerup", [0, 0]);
    expect(getViewportState().past).toHaveLength(0);
  });

  it("hides handles during creation, calibration, hidden selection, and deselection", () => {
    const { scene } = harness();
    const root = scene.getObjectByName("shape-vertex-handles")!;
    expect(root.children).toHaveLength(4);
    getViewportState().setTool("region");
    expect(root.children).toHaveLength(0);
    getViewportState().setTool("live");
    expect(root.children).toHaveLength(4);
    useCalibrateStore.setState({ active: true });
    expect(root.children).toHaveLength(0);
    useCalibrateStore.setState({ active: false });
    getViewportState().updateEntity("region-1", { visible: false });
    expect(root.children).toHaveLength(0);
    getViewportState().select(null);
    expect(root.children).toHaveLength(0);
  });
});