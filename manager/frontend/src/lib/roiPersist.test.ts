// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from "vitest";

const apiMock = vi.hoisted(() => ({
  getRegions: vi.fn(),
  getTripwires: vi.fn(),
  updateRegion: vi.fn(),
  updateTripwire: vi.fn(),
  createRegion: vi.fn(),
  createTripwire: vi.fn(),
  deleteRegion: vi.fn(),
  deleteTripwire: vi.fn(),
}));

vi.mock("./rest", () => ({ api: apiMock }));
vi.stubGlobal("document", { getElementById: () => null });

import { resetGeometryModel, upsertRoiMeta, upsertTripMeta } from "../scene/map/geometryModel";
import { persistSceneGeometry } from "./roiPersist";

const REGION_ID = "11111111-2222-4333-8444-555555555555";
const TRIPWIRE_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee";

describe("persistSceneGeometry display colors", () => {
  beforeEach(() => {
    resetGeometryModel();
    vi.clearAllMocks();
    apiMock.getRegions.mockResolvedValue([{ uid: REGION_ID }]);
    apiMock.getTripwires.mockResolvedValue([{ uid: TRIPWIRE_ID }]);
    apiMock.updateRegion.mockResolvedValue({});
    apiMock.updateTripwire.mockResolvedValue({});
  });

  it("sends independently configured Region and Tripwire colors to REST", async () => {
    upsertRoiMeta(REGION_ID, {
      title: "Dock",
      color: "#64d2ff",
      points: [
        [0, 0],
        [4, 0],
        [4, 3],
      ],
    });
    upsertTripMeta(TRIPWIRE_ID, {
      title: "Gate",
      color: "#bf5af2",
      points: [
        [1, 1],
        [5, 5],
      ],
    });

    await persistSceneGeometry("token", "scene");

    expect(apiMock.updateRegion).toHaveBeenCalledWith(
      "token",
      REGION_ID,
      expect.objectContaining({ color: "#64d2ff" }),
    );
    expect(apiMock.updateTripwire).toHaveBeenCalledWith(
      "token",
      TRIPWIRE_ID,
      expect.objectContaining({ color: "#bf5af2" }),
    );
  });
});