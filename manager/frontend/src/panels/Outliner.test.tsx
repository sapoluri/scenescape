// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ToastProvider } from "../components/ToastProvider";
import { Outliner } from "./Outliner";
import type { ViewportEntity } from "../viewport/types";
import { useEditModeStore } from "../lib/editMode";

vi.mock("../lib/editMode", async importOriginal => {
  const original = await importOriginal<typeof import("../lib/editMode")>();
  return { ...original, useWorkspaceEditing: () => original.workspaceEditingAllowed() };
});
const originalEditMode = useEditModeStore.getState();
afterEach(() => useEditModeStore.setState(originalEditMode));

const state = vi.hoisted(() => ({
  entities: {} as Record<string, ViewportEntity>, selectedId: null,
  projectAllActive: false, projectCameraIds: [] as string[],
}));
vi.mock("../viewport/store", () => ({
  useViewportStore: (selector: (value: typeof state) => unknown) => selector(state),
  getViewportState: () => state,
}));

describe("tracked objects section", () => {
  it("keeps projection and lighting but removes creation and deletion for viewers", () => {
    useEditModeStore.getState().enterWorkspace(false);
    state.entities = { camera: { id: "camera", type: "camera", name: "Camera 1", visible: true,
      position: [0, 0, 3], rotation: [0, 0, 0], restRotation: [0, 0, 0], target: [0, 0, 0],
      fov: 60, posed: true, color: "#0a84ff" } };
    const markup = renderToStaticMarkup(<ToastProvider><Outliner isSuperuser /></ToastProvider>);
    expect(markup).toContain('aria-label="Project Camera 1"');
    expect(markup).toContain("Ambient light intensity");
    expect(markup).not.toContain('aria-label="Add ');
    expect(markup).not.toContain('aria-label="Delete ');
    expect(markup).not.toContain('aria-label="Calibrate ');
  });

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