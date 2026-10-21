// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { ToastProvider } from "../components/ToastProvider";
import { Outliner } from "./Outliner";
import type { MarkEntity } from "../viewport/types";

const state = vi.hoisted(() => ({
  entities: {} as Record<string, MarkEntity>, selectedId: null,
  projectAllActive: false, projectCameraIds: [] as string[],
}));
vi.mock("../viewport/store", () => ({
  useViewportStore: (selector: (value: typeof state) => unknown) => selector(state),
  getViewportState: () => state,
}));

describe("tracked objects section", () => {
  it.each([0, 1])("remains below Lighting with %s tracks", count => {
    state.entities = count ? {
      track: {
        id: "track", type: "mark", name: "person #123456", visible: true,
        className: "person", color: "#123456", position: [1, 2, 0], heading: 0, speed: 0,
      },
    } : {};
    const markup = renderToStaticMarkup(<ToastProvider><Outliner /></ToastProvider>);
    expect(markup.indexOf('id="ss-outliner-body-camera"')).toBeLessThan(markup.indexOf('id="ss-outliner-body-region"'));
    expect(markup).toContain('aria-label="Tracked objects"');
    expect(markup.indexOf('id="ss-outliner-body-lighting"')).toBeLessThan(markup.indexOf('id="ss-outliner-body-mark"'));
    expect(markup).toContain(`class="ss-outliner-badge">${count}</span>`);
  });
});