// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from "vitest";
import { attachTelemetryCallbacks, formatCameraFps } from "./useLiveChrome";

afterEach(() => vi.unstubAllGlobals());

describe("telemetry callback lifetime", () => {
  it("releases callbacks across repeated mount/cleanup cycles", () => {
    vi.stubGlobal("window", {});
    for (let index = 0; index < 100; index++) {
      const stop = attachTelemetryCallbacks({ setCameraRate: vi.fn(), clearRates: vi.fn() });
      expect(Object.keys(window.ssSceneTelemetry!)).toHaveLength(2);
      stop();
      expect(window.ssSceneTelemetry).toBeUndefined();
    }
  });

  it("does not remove callbacks owned by newer instances or other telemetry surfaces", () => {
    vi.stubGlobal("window", {});
    const old = attachTelemetryCallbacks({ setCameraRate: vi.fn() });
    const replacement = vi.fn();
    const current = attachTelemetryCallbacks({ setCameraRate: replacement });
    const sceneRate = vi.fn();
    const scene = attachTelemetryCallbacks({ setSceneRate: sceneRate });
    old();
    expect(window.ssSceneTelemetry?.setCameraRate).toBe(replacement);
    expect(window.ssSceneTelemetry?.setSceneRate).toBe(sceneRate);
    current();
    expect(window.ssSceneTelemetry?.setCameraRate).toBeUndefined();
    expect(window.ssSceneTelemetry?.setSceneRate).toBe(sceneRate);
    scene();
    expect(window.ssSceneTelemetry).toBeUndefined();
  });
});

describe("camera FPS text", () => {
  it.each(["12.50", "12.50 FPS", "12.50 fps", "12.50 FPS fps"])("renders one unit for %s", value => {
    expect(formatCameraFps(value)).toBe("12.50 FPS");
  });
  it("handles unavailable rates", () => {
    expect(formatCameraFps()).toBe("-- FPS");
    expect(formatCameraFps("--")).toBe("-- FPS");
  });
});