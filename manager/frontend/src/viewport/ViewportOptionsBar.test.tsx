// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ViewportOptionsBar } from "./ViewportOptionsBar";
import { useEditModeStore } from "../lib/editMode";

vi.mock("../lib/editMode", async importOriginal => {
  const original = await importOriginal<typeof import("../lib/editMode")>();
  return { ...original, useWorkspaceEditing: () => original.workspaceEditingAllowed() };
});
const initialEditMode = useEditModeStore.getState();
afterEach(() => useEditModeStore.setState(initialEditMode));

const state = vi.hoisted(() => ({
  ortho: false, showGrid: true, showLabels: true, showTrails: true, showTelemetry: false, smoothTracks: false,
  setOrtho: () => {}, toggleOverlay: () => {},
}));

vi.mock("./store", () => ({
  useViewportStore: (selector: (value: typeof state) => unknown) => selector(state),
}));

describe("telemetry overlay visibility", () => {
  it.each([false, true])("shows Edit only for a superuser=%s", isSuperuser => {
    const markup = renderToStaticMarkup(<ViewportOptionsBar isSuperuser={isSuperuser} onEditChange={vi.fn()} />);
    expect(markup.includes('title="Enable scene editing"')).toBe(isSuperuser);
    const edit = markup.match(/<label[^>]*title="Enable scene editing"[\s\S]*?<\/label>/);
    if (isSuperuser) expect(edit?.[0]).toContain('checked=""');
    else expect(edit).toBeNull();
  });

  it.each([false, true])("shows Edit only for superusers=%s and keeps export available while viewing", superuser => {
    useEditModeStore.getState().enterWorkspace(superuser);
    useEditModeStore.getState().setEditing(false);
    const markup = renderToStaticMarkup(<ViewportOptionsBar isSuperuser={superuser}
      sceneChrome={{ sceneName: "Scene", isSuperuser: superuser, canDelete: true }} />);
    expect(markup.includes("Enable scene editing")).toBe(superuser);
    expect(markup).toContain('id="export-scene"');
    expect(markup).not.toContain('id="scene-edit"');
    expect(markup).not.toContain('id="scene-delete"');
    expect(markup).toContain("Smooth");
    expect(markup).toContain("Telemetry");
    expect(markup).toContain("Ortho");
    expect(markup.indexOf("Ortho")).toBeLessThan(markup.indexOf("Enable scene editing") === -1 ? markup.length : markup.indexOf("Enable scene editing"));
  });

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