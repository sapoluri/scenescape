// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import {
  filterScenes,
  formatModified,
  sceneStats,
  type SceneSummary,
} from "./OpenSceneDialog";

const A: SceneSummary = {
  uid: "aaa",
  name: "Loading Dock",
  cameras: [{}, {}],
  sensors: [{}],
  regions: [{}, {}, {}],
  tripwires: [{}],
  map_processed: "2026-09-15T10:00:00",
};
const B: SceneSummary = {
  uid: "bbb",
  name: "North Yard",
  cameras: [],
  map_processed: null,
};
const C: SceneSummary = { uid: "ccc", name: "dock staging" };

describe("filterScenes", () => {
  const scenes = [A, B, C];

  it("returns all scenes on empty query", () => {
    expect(filterScenes(scenes, "")).toEqual(scenes);
    expect(filterScenes(scenes, "   ")).toEqual(scenes);
  });

  it("matches case-insensitively on name", () => {
    const out = filterScenes(scenes, "DOCK");
    expect(out.map((s) => s.uid).sort()).toEqual(["aaa", "ccc"]);
  });

  it("matches substrings", () => {
    expect(filterScenes(scenes, "yard").map((s) => s.uid)).toEqual(["bbb"]);
  });

  it("returns an empty list when nothing matches", () => {
    expect(filterScenes(scenes, "zzz")).toEqual([]);
  });
});

describe("sceneStats", () => {
  it("counts cameras, sensors, and regions+tripwires", () => {
    expect(sceneStats(A)).toEqual({ cameras: 2, sensors: 1, rois: 4 });
  });

  it("treats missing arrays as zero", () => {
    expect(sceneStats(C)).toEqual({ cameras: 0, sensors: 0, rois: 0 });
  });
});

describe("formatModified", () => {
  it("formats a valid ISO date", () => {
    const out = formatModified("2026-09-15T10:00:00");
    expect(out).not.toBe("—");
    expect(out).toContain("2026");
  });

  it("falls back to an em dash for null or invalid input", () => {
    expect(formatModified(null)).toBe("—");
    expect(formatModified(undefined)).toBe("—");
    expect(formatModified("not-a-date")).toBe("—");
  });
});
