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
  it.each([false, true])("matches Telemetry=%s in the header without repeating the camera name", enabled => {
    state.showTelemetry = enabled;
    const markup = renderToStaticMarkup(<VisualizeFeedPane cameraRates={{ camera1: "10.20 FPS" }} />);
    expect(markup.includes('aria-label="Active camera telemetry"')).toBe(enabled);
    expect(markup.includes("10.20 FPS")).toBe(enabled);
    expect(markup).not.toContain("FPS fps");
    expect(markup.match(/Camera 1/g)).toHaveLength(1);
    expect(markup).not.toContain("ss-viz-feed-telemetry");
    if (enabled) {
      const header = markup.match(/<div class="ss-viz-feed-bar">[\s\S]*?<\/div>/)![0];
      expect(header).toContain('class="ss-viz-feed-fps"');
      expect(header).toContain("10.20 FPS");
    }
  });
});