// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { replaceShapePoints } from "./shapeHandles";
import type { SensorEntity } from "./types";

/** Mirrors the close-hit threshold used by sensor polygon drawing. */
const SENSOR_POLY_CLOSE_M = 0.45;

function closesOnFirst(
  points: [number, number][],
  click: [number, number],
): boolean {
  if (points.length < 3) {
    return false;
  }
  const first = points[0];
  return Math.hypot(first[0] - click[0], first[1] - click[1]) < SENSOR_POLY_CLOSE_M;
}

describe("sensor polygon drawing helpers", () => {
  it("closes when the click is near the first vertex after 3+ points", () => {
    const pts: [number, number][] = [
      [0, 0],
      [4, 0],
      [4, 3],
    ];
    expect(closesOnFirst(pts, [0.1, 0.1])).toBe(true);
    expect(closesOnFirst(pts, [2, 2])).toBe(false);
    expect(closesOnFirst(pts.slice(0, 2), [0, 0])).toBe(false);
  });

  it("keeps sensor location independent when finishing or redrawing a polygon", () => {
    const pts: [number, number][] = [
      [0, 0],
      [4, 0],
      [4, 4],
      [0, 4],
    ];
    const sensor: SensorEntity = {
      id: "sensor", type: "sensor", name: "Infrared", visible: true,
      position: [15, 20, 2], areaMode: "poly", radius: 5, points: [],
      singletonType: "environmental", color: "#bf5af2",
    };
    const updated = replaceShapePoints(sensor, pts) as SensorEntity;
    expect(updated.points).toEqual(pts);
    expect(updated.position).toEqual([15, 20, 2]);
    expect(updated.id).toBe(sensor.id);
  });
});
