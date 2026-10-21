// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from "vitest";
import type { CameraEntity } from "./types";

vi.mock("../mqtt", () => ({ refreshCameraStrip: vi.fn() }));

import { liveProjectionOptions, projectionCameraIds } from "./useLiveCameraProjection";

const camera = (id: string, visible = true): CameraEntity => ({
  id,
  type: "camera",
  name: id,
  visible,
  position: [0, 0, 2],
  target: [0, 0, 0],
  rotation: [0, 0, 0],
  restRotation: [0, 0, 0],
  posed: true,
  fov: 60,
  color: "#0a84ff",
});

describe("projectionCameraIds", () => {
  it("keeps the calibrated aspect instead of deriving it from the live JPEG", () => {
    const entity = { ...camera("retail"), fov: 38.29, aspect: 4 / 3 };
    expect(liveProjectionOptions(entity)).toMatchObject({ fov: 38.29, aspect: 4 / 3 });
    expect(liveProjectionOptions(camera("legacy"))).not.toHaveProperty("aspect");
  });

  it("isolates calibration and restores normal projection choices on exit", () => {
    const selected = ["first", "second"];
    const entities = { first: camera("first"), second: camera("second") };
    expect(projectionCameraIds(selected, entities, true, "first")).toEqual(["first"]);
    expect(projectionCameraIds(selected, entities, true, "second")).toEqual(["second"]);
    expect(projectionCameraIds(selected, entities, false, null)).toEqual(selected);
    expect(selected).toEqual(["first", "second"]);
  });

  it("projects the calibration camera even if its normal projection is off", () => {
    expect(projectionCameraIds([], {}, true, "calibration")).toEqual(["calibration"]);
    expect(projectionCameraIds(["first"], {}, true, null)).toEqual([]);
  });

  it("excludes hidden and unposed cameras from normal projections", () => {
    const entities = {
      first: camera("first"),
      hidden: camera("hidden", false),
      unposed: { ...camera("unposed"), posed: false },
    };
    expect(projectionCameraIds(Object.keys(entities), entities, false, null)).toEqual(["first"]);
  });
});