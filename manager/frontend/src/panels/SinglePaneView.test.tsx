// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useViewportStore } from "../viewport/store";
import { SinglePaneView } from "./SinglePaneView";

vi.mock("../viewport/store", async importOriginal => {
  const original = await importOriginal<typeof import("../viewport/store")>();
  return { ...original, useViewportStore: Object.assign(
    (selector: (state: ReturnType<typeof original.getViewportState>) => unknown) => selector(original.getViewportState()),
    original.useViewportStore,
  ) };
});
vi.mock("../viewport/SceneViewport", () => ({ SceneViewport: () => null }));
vi.mock("../viewport/ViewportMenuBar", () => ({ ViewportMenuBar: () => null }));
vi.mock("../viewport/ViewportOptionsBar", () => ({ ViewportOptionsBar: () => null }));
vi.mock("./Outliner", () => ({ Outliner: () => null }));
vi.mock("./Telemetry", () => ({ Telemetry: () => null }));
vi.mock("./LivePreviewPane", () => ({ LivePreviewPane: () => null, ProjectionFeedHost: () => null }));

const initialViewport = useViewportStore.getState();
afterEach(() => {
  useViewportStore.setState(initialViewport);
  vi.unstubAllGlobals();
});

describe("side panel controls", () => {
  const cases = [false, true].flatMap(collapsed => [false, true].map(telemetry => ({ collapsed, telemetry })));
  it.each(cases)("keeps Telemetry visible with collapsed=$collapsed and enabled=$telemetry", ({ collapsed, telemetry }) => {
    vi.stubGlobal("sessionStorage", {
      getItem: (key: string) => key === "ss-sp-dock-collapsed" && collapsed ? "1" : null,
    });
    useViewportStore.setState({ showTelemetry: telemetry, selectedId: null });
    const markup = renderToStaticMarkup(<SinglePaneView
      sceneId="scene" sceneName="Scene" cameras={[]} sensors={[]} authToken=""
    />);
    const telemetryTabs = markup.match(/<button[^>]*role="tab"[^>]*>Telemetry<\/button>/g)!;
    expect(telemetryTabs).toHaveLength(collapsed ? 2 : 1);
    for (const tab of telemetryTabs) {
      expect(tab.includes('disabled=""')).toBe(!telemetry);
    }
    const toggles = markup.match(/<button[^>]*ss-sp-dock-collapse[^>]*>/g)!;
    expect(toggles).toHaveLength(1);
    expect(toggles[0]).toContain(`aria-label="${collapsed ? "Expand" : "Collapse"} panel"`);
    expect(toggles[0]).toContain(`aria-expanded="${!collapsed}"`);
    expect(markup.indexOf(toggles[0])).toBeLessThan(markup.indexOf('id="ss-sp-dock-main"'));
    if (collapsed) {
      expect(markup).toContain('id="ss-sp-dock-main" inert="" aria-hidden="true"');
      expect(markup).toContain("Expand panel");
      expect(markup).not.toContain("bi-three-dots-vertical");
    } else {
      expect(markup).toContain('aria-label="Resize side panel"');
      const resizeHandle = markup.match(/<div[^>]*aria-label="Resize side panel"[^>]*>/)![0];
      expect(resizeHandle).toContain('title="Resize side panel"');
      expect(resizeHandle).not.toContain("aria-keyshortcuts");
      expect(resizeHandle).not.toContain("tabindex");
      expect(markup).toContain("bi-three-dots-vertical");
      expect(markup).not.toContain('class="ss-sp-dock-toggle"');
    }
  });
});