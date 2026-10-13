// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { Group } from "three";
import { applyDefaultPose, defaultAsset3D } from "../library/AssetLibrary";
import { LIBRARY_CHANGED_EVENT, poseMarkModel } from "./markAssets";

describe("poseMarkModel", () => {
  it("applies Asset3D default pose like the mark editor", () => {
    const g = new Group();
    const rec = {
      ...defaultAsset3D(),
      name: "forklift",
      translation: [1, 2, 0.5] as [number, number, number],
      rotation: [0, 0, 90] as [number, number, number],
      scale: [2, 2, 2] as [number, number, number],
    };
    poseMarkModel(g, rec);
    expect(g.position.x).toBeCloseTo(1);
    expect(g.position.y).toBeCloseTo(2);
    expect(g.position.z).toBeCloseTo(0.5);
    expect(g.scale.x).toBeCloseTo(2);
    expect(g.scale.y).toBeCloseTo(2);
    expect(g.scale.z).toBeCloseTo(2);
    expect(g.rotation.z).toBeCloseTo(Math.PI / 2);
  });

  it("exposes a stable library-changed event name", () => {
    expect(LIBRARY_CHANGED_EVENT).toBe("ss-library-changed");
  });

  it("matches applyDefaultPose", () => {
    const a = new Group();
    const b = new Group();
    const rec = {
      ...defaultAsset3D(),
      translation: [0.25, -0.5, 0.1] as [number, number, number],
      rotation: [10, 20, 30] as [number, number, number],
      scale: [1.5, 1.5, 1.5] as [number, number, number],
    };
    poseMarkModel(a, rec);
    applyDefaultPose(b, rec);
    expect(a.position.toArray()).toEqual(b.position.toArray());
    expect(a.scale.toArray()).toEqual(b.scale.toArray());
    expect(a.rotation.x).toBeCloseTo(b.rotation.x);
    expect(a.rotation.y).toBeCloseTo(b.rotation.y);
    expect(a.rotation.z).toBeCloseTo(b.rotation.z);
  });
});
