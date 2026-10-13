// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import {
  isGlbUrl,
  mapPlaneSizeMeters,
  sceneGeometryFromRest,
  sceneHasPlaceableGeometry,
} from "./sceneGeometry";

describe("sceneGeometry", () => {
  it("converts pixels-per-meter scale into floor meters", () => {
    // Queuing demo: 1200×1140 px at 157 px/m → ~7.64 × 7.26 m.
    const size = mapPlaneSizeMeters(1200, 1140, 157);
    expect(size.ppm).toBe(157);
    expect(size.widthM).toBeCloseTo(1200 / 157, 6);
    expect(size.heightM).toBeCloseTo(1140 / 157, 6);
    const fallback = mapPlaneSizeMeters(900, 643, null);
    expect(fallback.ppm).toBe(100);
    expect(fallback.widthM).toBe(9);
  });

  it("parses a map image and pixels-per-meter scale", () => {
    const spec = sceneGeometryFromRest({
      uid: "abc",
      name: "Hall",
      map: "/media/maps/hall.png",
      scale: 100,
    });
    expect(spec.mapUrl).toBe("/media/maps/hall.png");
    expect(spec.scale).toBe(100);
    expect(spec.isGlb).toBe(false);
    expect(spec.meshPose).toEqual({
      translation: [0, 0, 0],
      rotation: [0, 0, 0],
      scale: [1, 1, 1],
    });
    expect(sceneHasPlaceableGeometry(spec)).toBe(true);
  });

  it("detects GLB maps", () => {
    expect(isGlbUrl("/media/maps/campus.glb")).toBe(true);
    expect(isGlbUrl("/media/maps/campus.glb?v=2")).toBe(true);
    expect(isGlbUrl("/media/maps/floor.png")).toBe(false);
    const spec = sceneGeometryFromRest({
      id: "1",
      map: "/media/maps/campus.glb",
      scale: 1,
    });
    expect(spec.isGlb).toBe(true);
  });

  it("reads mesh pose from REST (with scalar fallbacks)", () => {
    const fromArrays = sceneGeometryFromRest({
      map: "/media/maps/campus.glb",
      mesh_translation: [4, 5, 6],
      mesh_rotation: [90, 0, 0],
      mesh_scale: [1, 1, 1],
    });
    expect(fromArrays.meshPose).toEqual({
      translation: [4, 5, 6],
      rotation: [90, 0, 0],
      scale: [1, 1, 1],
    });
    const fromScalars = sceneGeometryFromRest({
      map: "/media/maps/campus.glb",
      rotation_x: 90,
      translation_x: 2,
      translation_y: 3,
      translation_z: 1,
      scale_x: 1,
      scale_y: 1,
      scale_z: 1,
    });
    expect(fromScalars.meshPose.rotation[0]).toBe(90);
    expect(fromScalars.meshPose.translation).toEqual([2, 3, 1]);
  });

  it("treats a missing map as not placeable", () => {
    const spec = sceneGeometryFromRest({ uid: "x", name: "Empty" });
    expect(sceneHasPlaceableGeometry(spec)).toBe(false);
  });
});
