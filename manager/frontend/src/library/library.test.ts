// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Object3D } from "three";
import { describe, expect, it } from "vitest";
import {
  applyDefaultPose,
  buildAssetFormData,
  defaultAsset3D,
  footprintOf,
  glbBasename,
  parseAsset3D,
  readDefaultPose,
  shiftTypeLabel,
  simulateTrackerPath,
} from "./AssetLibrary";

describe("parseAsset3D", () => {
  it("normalizes a serializer row with model defaults", () => {
    const rec = parseAsset3D({
      uid: "12",
      name: "forklift",
      x_size: 2.5,
      mark_color: "#ff0000",
      rotation_x: 90,
      translation_z: 0.4,
      scale: 1.5,
      rotation_from_velocity: true,
      model_3d: "/media/models/forklift.glb",
      geometric_center: [0.1, 0, 0.2],
      friction_coefficients: [0.6, 0.5],
    });
    expect(rec.uid).toBe("12");
    expect(rec.name).toBe("forklift");
    expect(rec.xSize).toBe(2.5);
    expect(rec.ySize).toBe(1); // model default
    expect(rec.markColor).toBe("#ff0000");
    expect(rec.rotation).toEqual([90, 0, 0]);
    expect(rec.translation).toEqual([0, 0, 0.4]);
    expect(rec.scale).toBe(1.5);
    expect(rec.rotationFromVelocity).toBe(true);
    expect(rec.modelUrl).toBe("/media/models/forklift.glb");
    expect(rec.geometricCenter).toEqual([0.1, 0, 0.2]);
    expect(rec.friction).toEqual([0.6, 0.5]);
    expect(rec.shiftType).toBe(1);
    expect(rec.projectToMap).toBe(false);
  });

  it("handles nulls and blanks like the Django defaults", () => {
    const rec = parseAsset3D({ uid: "3", name: "x", mark_color: "" });
    expect(rec.markColor).toBe("#888888");
    expect(rec.modelUrl).toBeNull();
    expect(rec.rotation).toEqual([0, 0, 0]);
    expect(rec.scale).toBe(1);
  });

  it("defaultAsset3D mirrors the model defaults", () => {
    const rec = defaultAsset3D();
    expect(rec.xSize).toBe(1);
    expect(rec.trackingRadius).toBe(2);
    expect(rec.linearDamping).toBe(0.05);
    expect(rec.restitution).toBe(0.5);
    expect(rec.ttl).toBe(0);
  });
});

describe("buildAssetFormData", () => {
  it("encodes booleans as True/False strings and lists as JSON", () => {
    const rec = {
      ...defaultAsset3D(),
      name: " pallet ",
      rotationFromVelocity: true,
      projectToMap: false,
      isStatic: true,
      geometricCenter: [0.1, 0.2, 0.3] as [number, number, number],
      friction: [0.6, 0.5] as [number, number],
    };
    const form = buildAssetFormData(rec);
    expect(form.get("name")).toBe("pallet");
    expect(form.get("rotation_from_velocity")).toBe("True");
    expect(form.get("project_to_map")).toBe("False");
    expect(form.get("is_static")).toBe("True");
    expect(form.get("geometric_center")).toBe("[0.1,0.2,0.3]");
    expect(form.get("friction_coefficients")).toBe("[0.6,0.5]");
    expect(form.get("shift_type")).toBe("1");
  });

  it("appends a GLB file part or the clear flag", () => {
    const rec = defaultAsset3D();
    const file = new File(["x"], "model.glb", { type: "model/gltf-binary" });
    const withFile = buildAssetFormData(rec, { modelFile: file });
    expect(withFile.get("model_3d")).toBe(file);
    const cleared = buildAssetFormData(rec, { clearModel: true });
    expect(cleared.get("clear_model_3d")).toBe("True");
    expect(cleared.get("model_3d")).toBeNull();
  });
});

describe("footprintOf", () => {
  it("expands the plan rectangle by the buffer on each side", () => {
    const rec = {
      ...defaultAsset3D(),
      xSize: 2,
      ySize: 1,
      zSize: 1.5,
      xBuffer: 0.5,
      yBuffer: 0.25,
    };
    const fp = footprintOf(rec);
    expect(fp.inner).toEqual({ x: 2, y: 1 });
    expect(fp.outer).toEqual({ x: 3, y: 1.5 });
    expect(fp.height).toBe(1.5);
  });

  it("clamps degenerate sizes to a visible minimum", () => {
    const fp = footprintOf({ ...defaultAsset3D(), xSize: 0, ySize: -1 });
    expect(fp.inner.x).toBeGreaterThan(0);
    expect(fp.inner.y).toBeGreaterThan(0);
  });
});

describe("default pose round-trip", () => {
  it("applyDefaultPose then readDefaultPose is stable", () => {
    const rec = {
      ...defaultAsset3D(),
      rotation: [10, -20, 30] as [number, number, number],
      translation: [1.5, -0.5, 0.25] as [number, number, number],
      scale: 2,
    };
    const obj = new Object3D();
    applyDefaultPose(obj, rec);
    expect(obj.position.x).toBeCloseTo(1.5, 9);
    expect(obj.scale.x).toBeCloseTo(2, 9);
    expect(obj.scale.y).toBeCloseTo(2, 9);
    const back = readDefaultPose(obj);
    expect(back.rotation[0]).toBeCloseTo(10, 6);
    expect(back.rotation[1]).toBeCloseTo(-20, 6);
    expect(back.rotation[2]).toBeCloseTo(30, 6);
    expect(back.translation).toEqual([1.5, -0.5, 0.25]);
    expect(back.scale).toBeCloseTo(2, 9);
  });
});

describe("simulateTrackerPath", () => {
  it("travels a circle with velocity-aligned heading", () => {
    const a = simulateTrackerPath(0, 4, 12);
    expect(a.x).toBeCloseTo(4, 9);
    expect(a.y).toBeCloseTo(0, 9);
    // Velocity at θ=0 is (0, +ωR): heading +Y = π/2.
    expect(a.heading).toBeCloseTo(Math.PI / 2, 9);
    const quarter = simulateTrackerPath(3, 4, 12);
    expect(quarter.x).toBeCloseTo(0, 9);
    expect(quarter.y).toBeCloseTo(4, 9);
    // Velocity at θ=π/2 is (−ωR, 0): heading π.
    expect(Math.abs(quarter.heading)).toBeCloseTo(Math.PI, 9);
  });

  it("wraps cleanly at the period boundary", () => {
    const a = simulateTrackerPath(0);
    const b = simulateTrackerPath(12);
    expect(b.x).toBeCloseTo(a.x, 9);
    expect(b.y).toBeCloseTo(a.y, 9);
    expect(b.heading).toBeCloseTo(a.heading, 9);
  });
});

describe("labels", () => {
  it("names shift types", () => {
    expect(shiftTypeLabel(1)).toBe("Center");
    expect(shiftTypeLabel(2)).toBe("Bottom");
  });

  it("basenames GLB urls", () => {
    expect(glbBasename(null)).toBe("No GLB");
    expect(glbBasename("/media/models/forklift.glb")).toBe("forklift.glb");
  });
});
