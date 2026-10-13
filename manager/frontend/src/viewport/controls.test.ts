// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from "vitest";
import {
  OrthographicCamera,
  PerspectiveCamera,
  Vector3,
} from "three";
import { applyViewPreset } from "./controls";

function stubOrbit(target = new Vector3()) {
  return {
    target,
    update: vi.fn(),
  } as unknown as import("three/addons/controls/OrbitControls.js").OrbitControls;
}

describe("applyViewPreset", () => {
  it("places an orthographic camera looking straight down for top", () => {
    const camera = new OrthographicCamera(-10, 10, 10, -10, 0.1, 1000);
    camera.up.set(0, 0, 1);
    const target = new Vector3(4, 6, 0);
    const orbit = stubOrbit(target.clone());
    applyViewPreset(camera, orbit, "top", target, 20);
    expect(camera.position.x).toBeCloseTo(4);
    expect(camera.position.y).toBeCloseTo(6);
    expect(camera.position.z).toBeCloseTo(20);
    // View direction must be -Z; up cannot be parallel to it.
    expect(camera.up.z).toBe(0);
    expect(Math.abs(camera.up.y)).toBeGreaterThan(0.5);
  });

  it("keeps Z-up for the angled perspective preset", () => {
    const camera = new PerspectiveCamera(50, 1, 0.1, 1000);
    const target = new Vector3(0, 0, 0);
    const orbit = stubOrbit();
    applyViewPreset(camera, orbit, "persp", target, 28);
    expect(camera.up.z).toBe(1);
    expect(camera.position.z).toBeGreaterThan(0);
    expect(camera.position.y).toBeLessThan(0);
  });
});
