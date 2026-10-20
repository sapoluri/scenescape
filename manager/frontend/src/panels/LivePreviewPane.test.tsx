// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { LivePreviewPane } from "./LivePreviewPane";

const state = vi.hoisted(() => ({
  active: false, cameraId: "1", selectedId: "1", showTelemetry: false,
}));
vi.mock("../viewport/store", () => ({
  useViewportStore: (selector: (value: typeof state) => unknown) => selector(state),
}));
vi.mock("../viewport/visualize/visualizeStore", () => ({
  useVisualizeStore: (selector: (value: typeof state) => unknown) => selector(state),
}));
vi.mock("../viewport/visualize", () => ({ enterCameraVisualize: vi.fn() }));
vi.mock("../mqtt", () => ({ refreshCameraStrip: vi.fn() }));

const cameras = ["1", "2"].map(id => ({
  id, sensorId: `camera${id}`, name: `Camera ${id}`, calibrateHref: "",
  cmdTopic: `scenescape/cmd/camera/camera${id}`, deleteUrl: null,
}));

describe("active live camera previews", () => {
  it.each([false, true])("hides only the active camera while live view is active=%s", active => {
    state.active = active;
    const markup = renderToStaticMarkup(<LivePreviewPane cameras={cameras} />);
    const cards = markup.match(/<div[^>]*role="option"[^>]*>/g)!;
    expect(cards).toHaveLength(2);
    expect(cards[0].includes('hidden=""')).toBe(active);
    expect(cards[1]).not.toContain('hidden=""');
    expect(markup).toContain(`live of ${active ? 1 : 2} cameras`);
    expect(markup).toContain('id="live-preview-camera1"');
  });
});