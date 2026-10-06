// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { deriveHeading, parseMarkPosition } from "./marks";

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
