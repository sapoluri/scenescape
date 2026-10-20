// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { VisualizeBar } from "./VisualizeBar";

const state = vi.hoisted(() => ({
  cameraId: "camera",
  cameraName: "Camera 1",
  exit: () => {},
  entities: { camera: { type: "camera", posed: true } },
  projectCameraIds: [] as string[],
  projectOpacityById: {},
  projectAllOpacity: 0.8,
}));
vi.mock("./visualizeStore", () => ({
  useVisualizeStore: (selector: (value: typeof state) => unknown) =>
    selector(state),
}));
vi.mock("../store", () => ({
  useViewportStore: (selector: (value: typeof state) => unknown) =>
    selector(state),
  getViewportState: () => state,
  projectOpacityOf: () => 0.8,
}));
vi.mock("../calibrate", () => ({ enterCameraCalibrate: vi.fn() }));
vi.mock("./index", () => ({ enterCameraVisualize: vi.fn() }));

describe("Live camera toolbar", () => {
  it.each([false, true])(
    "keeps opacity left of calibration-style actions when projecting=%s",
    (projecting) => {
      state.projectCameraIds = projecting ? ["camera"] : [];
      const markup = renderToStaticMarkup(
        <VisualizeBar cameras={[]} isSuperuser />,
      );
      expect(markup).toContain("ss-cal-wizard--calibration");
      expect(markup).toContain("ss-btn--sm");
      expect(markup).toContain('aria-label="Close live view"');
      expect(markup).toContain("bi bi-x");
      expect(markup).not.toContain(">Exit<");
      expect(markup.includes('aria-label="Projection opacity"')).toBe(
        projecting,
      );
      if (projecting) {
        expect(markup.indexOf('aria-label="Projection opacity"')).toBeLessThan(
          markup.indexOf('class="ss-cal-wizard-actions"'),
        );
      }
    },
  );
});
