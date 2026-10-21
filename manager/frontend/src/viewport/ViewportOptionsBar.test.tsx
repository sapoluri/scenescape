// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { ViewportOptionsBar } from "./ViewportOptionsBar";

const state = vi.hoisted(() => ({
  ortho: false, showGrid: true, showLabels: true, showTrails: true, showTelemetry: false, smoothTracks: false,
  setOrtho: () => {}, toggleOverlay: () => {},
}));

vi.mock("./store", () => ({
  useViewportStore: (selector: (value: typeof state) => unknown) => selector(state),
}));

describe("telemetry overlay visibility", () => {
  it.each([false, true])("shows an independent Smooth checkbox with state=%s", enabled => {
    state.smoothTracks = enabled;
    state.showTelemetry = false;
    const markup = renderToStaticMarkup(<ViewportOptionsBar />);
    expect(markup).toContain('title="Smooth tracks in viewer only"');
    expect(markup.indexOf("Telemetry")).toBeLessThan(markup.indexOf("Smooth"));
    const checkbox = markup.match(/<label[^>]*title="Smooth tracks in viewer only"[\s\S]*?<\/label>/)![0];
    expect(checkbox.includes('checked=""')).toBe(enabled);
    expect(checkbox).not.toContain('disabled=""');
  });

  it.each([false, true])("matches scene rate visibility to Telemetry=%s", enabled => {
    state.showTelemetry = enabled;
    const markup = renderToStaticMarkup(<ViewportOptionsBar sceneChrome={{ sceneName: "Queuing", sceneRate: "40.0" }} />);
    const rate = markup.match(/<div[^>]*class="scene-rate[^"]*"[^>]*>/)?.[0];
    expect(rate).toBeDefined();
    expect(rate!.includes('hidden=""')).toBe(!enabled);
    expect(markup).not.toContain("✓");
  });
});