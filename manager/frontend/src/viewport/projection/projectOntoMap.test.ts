// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from "vitest";
import { Matrix4, Mesh, MeshBasicMaterial, Object3D, PlaneGeometry, Texture, TextureLoader, Vector3 } from "three";
import { createCameraProjection, shouldReprojectFrame } from "./projectOntoMap";
import { diagonalFovFromOptics, focalFromDiagonalFov, parseCameraOptics } from "../../lib/cameraOptics";
import { calibrationOpticsForFrame, fovFromOptics } from "../calibrate/refineIntrinsics";

const previous = {
  fov: 60,
  aspect: 4 / 3,
  position: [0, 0, 0] as [number, number, number],
  rotationDeg: [0, 0, 0] as [number, number, number],
};

describe("manual diagonal FOV constraints", () => {
  it("round-trips the diagonal angle and derives a distinct vertical angle", () => {
    const focal = focalFromDiagonalFov(90, 320, 240);
    expect(focal).toBeCloseTo(400, 10);
    const optics = { ...parseCameraOptics({}), fx: focal, fy: focal, cx: 320, cy: 240 };
    expect(diagonalFovFromOptics(optics)).toBeCloseTo(90, 10);
    expect(fovFromOptics(optics)).toBeCloseTo(61.927513, 5);
  });

  it("preserves the angle or focal pixels across resizing according to the constraint", () => {
    const optics = calibrationOpticsForFrame({ ...parseCameraOptics({}), fy: 400 }, 640, 480);
    const byAngle = calibrationOpticsForFrame(optics, 1280, 960, 90);
    const byFocal = calibrationOpticsForFrame(optics, 1280, 960);
    expect(byAngle.fy).toBeCloseTo(800, 10);
    expect(diagonalFovFromOptics(byAngle)).toBeCloseTo(90, 10);
    expect(byFocal.fy).toBe(400);
    expect(diagonalFovFromOptics(byFocal)).toBeGreaterThan(90);
  });
});

describe("camera projection frame updates", () => {
  it("reprojects texture-only updates when the decoded frame aspect changes", () => {
    expect(
      shouldReprojectFrame(false, { ...previous, aspect: 16 / 9 }, previous),
    ).toBe(true);
  });

  it("keeps same-aspect texture-only updates cheap", () => {
    expect(shouldReprojectFrame(false, previous, previous)).toBe(false);
  });
});

describe("calibrated projector pixel mapping", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it.each([[1920, 1080, 1200], [640, 480, 690.25], [640, 480, 100], [640, 480, 5000]])(
    "maps pixels using the actual %ix%i frame with focal length %i",
    async (width, height, focal) => {
    vi.stubGlobal("window", { addEventListener: vi.fn(), removeEventListener: vi.fn() });
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("Use fixture texture")));
    const texture = new Texture({ width, height } as HTMLImageElement);
    vi.spyOn(TextureLoader.prototype, "loadAsync").mockResolvedValue(texture);
    const geometry = new PlaneGeometry(10, 10);
    const material = new MeshBasicMaterial();
    const map = new Mesh(geometry, material);
    const handle = createCameraProjection(map, new Object3D());
    try {
      const stored = parseCameraOptics({
        resolution: { width: 1280, height: 720 },
        intrinsics: { fx: focal, fy: focal, cx: 320, cy: 240 },
      });
      const optics = calibrationOpticsForFrame(stored, width, height);
      expect(optics.fx).toBe(optics.fy);
      expect(optics.cx).toBe(width / 2);
      expect(optics.cy).toBe(height / 2);
      const pose = {
        fov: fovFromOptics(optics),
        aspect: width / height,
        position: previous.position,
        rotationDeg: previous.rotationDeg,
      };
      handle.updatePose(pose);
      await handle.updateFrame("/fixture.jpg", { ...pose, reproject: false });
      const uniforms = (handle.material as unknown as {
        uniforms: {
          projectionMatrixCamera: { value: Matrix4 };
          widthScaled: { value: number };
          heightScaled: { value: number };
        };
      }).uniforms;
      const projected = new Vector3(1, -0.5, -5).applyMatrix4(uniforms.projectionMatrixCamera.value);
      expect((projected.x * 0.5 + 0.5) * width).toBeCloseTo(focal / 5 + width / 2, 8);
      expect((0.5 - projected.y * 0.5) * height).toBeCloseTo(focal * 0.5 / 5 + height / 2, 8);
      expect(uniforms.widthScaled.value).toBeCloseTo(1, 8);
      expect(uniforms.heightScaled.value).toBeCloseTo(1, 8);
    } finally {
      handle.dispose();
      geometry.dispose();
      material.dispose();
    }
  });
});