// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { VisualizeFeedPane } from "./VisualizeFeedPane";

const state = vi.hoisted(() => ({
  sensorId: "camera1", cameraName: "Camera 1", showTelemetry: false,
  setFeedImageUrl: () => {},
}));
vi.mock("./visualizeStore", () => ({
  useVisualizeStore: (selector: (value: typeof state) => unknown) => selector(state),
}));
vi.mock("../store", () => ({
  useViewportStore: (selector: (value: typeof state) => unknown) => selector(state),
}));
vi.mock("./useLiveCameraImage", () => ({
  useLiveCameraImage: () => ({ imageUrl: "/frame.jpg", naturalSize: { w: 1280, h: 720 } }),
}));

describe("active camera telemetry", () => {
  it.each([false, true])("matches Telemetry=%s beneath the active image", enabled => {
    state.showTelemetry = enabled;
    const markup = renderToStaticMarkup(<VisualizeFeedPane cameraRates={{ camera1: "10.20 FPS" }} />);
    expect(markup.includes('aria-label="Active camera telemetry"')).toBe(enabled);
    expect(markup.includes("10.20 FPS")).toBe(enabled);
    expect(markup).not.toContain("FPS fps");
    if (enabled) {
      expect(markup.indexOf('class="ss-viz-feed-telemetry"')).toBeGreaterThan(markup.indexOf('class="ss-viz-feed-img"'));
    }
  });
});