// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from "vitest";

// geometryModel touches document/window; the repo's vitest runs in node.
const dispatchSpy = vi.fn();
vi.stubGlobal("document", { getElementById: () => null });
vi.stubGlobal("window", { dispatchEvent: dispatchSpy });

import {
  resetGeometryModel,
  upsertRoiMeta,
  upsertTripMeta,
} from "../scene/map/geometryModel";
import {
  bboxOf,
  centroidOf,
  seedEntitiesFromGeometry,
  snapshotGeometry,
  writeRegionToModel,
} from "./geometrySync";
import type { RegionEntity } from "./types";

beforeEach(() => {
  resetGeometryModel();
  dispatchSpy.mockClear();
});

describe("bboxOf / centroidOf", () => {
  it("computes bounds and centroid", () => {
    const pts: [number, number][] = [
      [0, 0],
      [4, 0],
      [4, 3],
    ];
    expect(bboxOf(pts)).toEqual({ min: [0, 0], max: [4, 3] });
    const [cx, cy] = centroidOf(pts);
    expect(cx).toBeCloseTo(8 / 3);
    expect(cy).toBeCloseTo(1);
  });

  it("handles empty input", () => {
    expect(bboxOf([])).toEqual({ min: [0, 0], max: [0, 0] });
    expect(centroidOf([])).toEqual([0, 0]);
  });
});

describe("seedEntitiesFromGeometry", () => {
  it("maps roi rows to region entities", () => {
    upsertRoiMeta("uuid-1", {
      title: "Dock",
      points: [
        [0, 0],
        [4, 0],
        [4, 3],
      ],
      volumetric: true,
      height: 2.5,
      buffer_size: 1,
    });
    upsertTripMeta("trip-1", {
      title: "Gate",
      points: [
        [1, 1],
        [5, 5],
      ],
    });
    const { regions, tripwires } = seedEntitiesFromGeometry();
    expect(regions).toHaveLength(1);
    expect(regions[0]).toMatchObject({
      id: "uuid-1",
      type: "region",
      name: "Dock",
      min: [0, 0],
      max: [4, 3],
      height: 2.5,
      volumetric: true,
      bufferSize: 1,
    });
    expect(tripwires).toHaveLength(1);
    expect(tripwires[0]).toMatchObject({
      id: "trip-1",
      type: "tripwire",
      name: "Gate",
      a: [1, 1],
      b: [5, 5],
    });
  });
});

describe("writeRegionToModel", () => {
  it("writes points to the model and publishes dirty", () => {
    const entity: RegionEntity = {
      id: "uuid-9",
      type: "region",
      name: "Yard",
      visible: true,
      points: [
        [2, 2],
        [6, 2],
        [6, 6],
      ],
      min: [2, 2],
      max: [6, 6],
      height: 1,
      color: "#30d158",
      volumetric: false,
      bufferSize: 0,
    };
    writeRegionToModel(entity);
    const snap = snapshotGeometry();
    expect(snap.rois).toHaveLength(1);
    expect(snap.rois[0].points).toEqual([
      [2, 2],
      [6, 2],
      [6, 6],
    ]);
    expect(dispatchSpy).toHaveBeenCalledWith(
      expect.objectContaining({ type: "ss-roi-dirty" }),
    );
  });
});
