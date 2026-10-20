// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { MeshBasicMaterial } from "three";
import { createTrailRibbon } from "./renderMarks";
import {
  buildMarkEntities,
  colorFromTrackId,
  deriveHeading,
  markKeysLeftScene,
  markLabel,
  parseMarkPosition,
  trackIdFromMarkKey,
} from "./marks";

describe("trail rendering", () => {
  it("renders beneath object bodies and respects scene depth", () => {
    const trail = createTrailRibbon("#ff0000");
    const material = trail.material as MeshBasicMaterial;
    try {
      expect(material.depthTest).toBe(true);
      expect(material.depthWrite).toBe(false);
      expect(material.transparent).toBe(true);
      expect(trail.renderOrder).toBe(-1);
    } finally {
      trail.geometry.dispose();
      material.dispose();
    }
  });
});

describe("parseMarkPosition", () => {
  it("parses array translations", () => {
    expect(parseMarkPosition([1.5, -2.5])).toEqual([1.5, -2.5]);
    expect(parseMarkPosition([1.5, -2.5, 9])).toEqual([1.5, -2.5]);
  });

  it("parses object translations", () => {
    expect(parseMarkPosition({ x: 3, y: 4 })).toEqual([3, 4]);
  });

  it("rejects missing and non-finite translations", () => {
    expect(parseMarkPosition(undefined)).toBeNull();
    expect(parseMarkPosition(null as never)).toBeNull();
    expect(parseMarkPosition([1])).toBeNull();
    expect(parseMarkPosition([Number.NaN, 2])).toBeNull();
    expect(parseMarkPosition({ x: 1 })).toBeNull();
    expect(parseMarkPosition("1,2" as never)).toBeNull();
  });
});

describe("deriveHeading", () => {
  it("computes heading from movement (0 = +X, CCW)", () => {
    expect(deriveHeading(0, 0, 1, 0, 0)).toBeCloseTo(0);
    expect(deriveHeading(0, 0, 0, 1, 0)).toBeCloseTo(Math.PI / 2);
    expect(deriveHeading(0, 0, -1, 0, 0)).toBeCloseTo(Math.PI);
  });

  it("keeps the previous heading when barely moved", () => {
    expect(deriveHeading(0, 0, 0.01, 0.01, 1.23)).toBeCloseTo(1.23);
  });
});

describe("markLabel", () => {
  it("matches legacy 3D category #id format", () => {
    expect(markLabel("person", "a1b2c3d4-e5f6")).toBe(
      "person #a1b2c3d4-e5f6",
    );
    expect(markLabel("vehicle", 42)).toBe("vehicle #42");
  });
});

describe("colorFromTrackId", () => {
  it("matches 2D UI: # + first 6 characters of the id", () => {
    expect(colorFromTrackId("a1b2c3d4-e5f6-7890")).toBe("#a1b2c3");
    expect(colorFromTrackId("ff00aa")).toBe("#ff00aa");
  });
});

describe("trackIdFromMarkKey", () => {
  it("strips the mark: prefix", () => {
    expect(trackIdFromMarkKey("mark:a1b2c3d4")).toBe("a1b2c3d4");
    expect(trackIdFromMarkKey("a1b2c3d4")).toBe("a1b2c3d4");
  });
});

describe("markKeysLeftScene", () => {
  it("returns store keys omitted from the current frame", () => {
    expect(
      markKeysLeftScene(
        ["mark:a", "mark:b", "mark:c"],
        [{ id: "b", translation: [1, 2] }],
      ),
    ).toEqual(["mark:a", "mark:c"]);
  });

  it("clears all when the frame is empty", () => {
    expect(markKeysLeftScene(["mark:a", "mark:b"], [])).toEqual([
      "mark:a",
      "mark:b",
    ]);
  });

  it("keeps an id present even when translation is missing", () => {
    expect(
      markKeysLeftScene(["mark:a"], [{ id: "a" }]),
    ).toEqual([]);
  });
});

describe("buildMarkEntities", () => {
  it("upserts positions and records seen state", () => {
    const seen = new Map();
    const entities = buildMarkEntities(
      [{ id: "t1", type: "person", translation: [1, 2, 0] }],
      seen,
      { person: "#ff0000" },
      1000,
    );
    expect(entities).toHaveLength(1);
    expect(entities[0]).toMatchObject({
      id: "mark:t1",
      className: "person",
      color: "#ff0000",
      position: [1, 2, 0],
    });
    expect(seen.get("mark:t1")?.lastSeen).toBe(1000);
  });
});
