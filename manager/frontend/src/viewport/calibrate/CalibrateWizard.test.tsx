// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { ToastProvider } from "../../components/ToastProvider";
import { CalibrateWizard } from "./CalibrateWizard";

const state = vi.hoisted(() => ({
  cameraId: "1", sensorId: "camera1", cameraName: "Camera 1",
  camPoints: [[0, 0], [1, 0], [1, 1], [0, 1]],
  mapPoints: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
  dirty: true, optics: {}, priorPose: null, feedImageUrl: null, updatingPose: false,
  pastPoints: [] as unknown[], futurePoints: [] as unknown[], pointEditStart: null,
  exit: () => {}, undoPoint: () => {}, redoPoint: () => {},
  resetPoints: () => {}, markClean: () => {}, enter: () => {},
}));

vi.mock("./calibrateStore", () => ({
  useCalibrateStore: (selector: (value: typeof state) => unknown) => selector(state),
  getCalibrateState: () => state,
}));

vi.mock("../store", () => ({
  useViewportStore: (selector: (value: { projectOpacityById: Record<string, number>; projectAllOpacity: number }) => unknown) =>
    selector({ projectOpacityById: {}, projectAllOpacity: 0.8 }),
  getViewportState: () => ({}),
  projectOpacityOf: () => 0.8,
}));

describe("calibration toolbar", () => {
  it.each([false, true])("uses standard Undo/Redo icons and reflects available history=%s", available => {
    state.pastPoints = available ? [{}] : [];
    state.futurePoints = available ? [{}] : [];
    const markup = renderToStaticMarkup(<ToastProvider><CalibrateWizard
      sceneId="scene" authToken="test-token" cameras={[{ id: "1", sensorId: "camera1", name: "Camera 1" }]}
    /></ToastProvider>);
    expect(markup).toContain("bi bi-arrow-counterclockwise");
    expect(markup).toContain("bi bi-arrow-clockwise");
    expect(markup).toContain('aria-label="Undo calibration edit"');
    expect(markup).toContain('aria-label="Redo calibration edit"');
    const buttons = markup.match(/<button[^>]*aria-label="(?:Undo|Redo) calibration edit"[^>]*>/g)!;
    expect(buttons).toHaveLength(2);
    buttons.forEach(button => expect(button.includes('disabled=""')).toBe(!available));
    expect(markup).toContain(">Reset</button>");
    expect(markup).toContain("ss-cal-wizard-save ss-btn--dirty");
    expect(markup).toContain("<span>Save</span>");
  });
});