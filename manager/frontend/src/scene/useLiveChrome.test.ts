// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { formatCameraFps } from "./useLiveChrome";

describe("camera FPS text", () => {
  it.each(["12.50", "12.50 FPS", "12.50 fps", "12.50 FPS fps"])("renders one unit for %s", value => {
    expect(formatCameraFps(value)).toBe("12.50 FPS");
  });
  it("handles unavailable rates", () => {
    expect(formatCameraFps()).toBe("-- FPS");
    expect(formatCameraFps("--")).toBe("-- FPS");
  });
});