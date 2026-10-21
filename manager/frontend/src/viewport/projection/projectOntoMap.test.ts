// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { shouldReprojectFrame } from "./projectOntoMap";

const previous = {
  fov: 60,
  aspect: 4 / 3,
  position: [0, 0, 0] as [number, number, number],
  rotationDeg: [0, 0, 0] as [number, number, number],
};

describe("camera projection frame updates", () => {
  it("reprojects texture-only updates when the decoded frame aspect changes", () => {
    expect(
      shouldReprojectFrame(false, { ...previous, aspect: 16 / 9 }, previous),
    ).toBe(true);
  });

  it("keeps same-aspect texture-only updates cheap", () => {
    expect(shouldReprojectFrame(false, previous, previous)).toBe(false);
  });
});