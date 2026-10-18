// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import type { MarkEntity, RegionEntity, TripwireEntity } from "../viewport/types";
import {
  diffGeometryTelemetry,
  pointInPolygon,
  segmentsIntersect,
} from "./geometryTelemetry";
import { TOPIC_KEY_REGION, TOPIC_KEY_TRIPWIRE } from "./telemetryModel";

function mark(
  id: string,
  x: number,
  y: number,
  className = "person",
): MarkEntity {
  return {
    id,
    type: "mark",
    name: className,
    visible: true,
    className,
    color: "#fff",
    position: [x, y, 0],
    heading: Number.NaN,
    speed: 0,
  };
}

const square: RegionEntity = {
  id: "reg-1",
  type: "region",
  name: "Box",
  visible: true,
  points: [
    [0, 0],
    [2, 0],
    [2, 2],
    [0, 2],
  ],
  min: [0, 0],
  max: [2, 2],
  height: 1,
  color: "#0f0",
  volumetric: false,
  bufferSize: 0,
};

const wire: TripwireEntity = {
  id: "tw-1",
  type: "tripwire",
  name: "Door",
  visible: true,
  points: [
    [1, 0],
    [1, 2],
  ],
  a: [1, 0],
  b: [1, 2],
  color: "#00f",
};

describe("pointInPolygon", () => {
  it("detects inside and outside", () => {
    expect(pointInPolygon(1, 1, square.points)).toBe(true);
    expect(pointInPolygon(3, 3, square.points)).toBe(false);
  });
});

describe("segmentsIntersect", () => {
  it("detects a crossing", () => {
    expect(segmentsIntersect([0, 1], [2, 1], [1, 0], [1, 2])).toBe(true);
    expect(segmentsIntersect([0, 0], [0, 1], [2, 0], [2, 1])).toBe(false);
  });
});

describe("diffGeometryTelemetry", () => {
  it("emits region-enter and region-exit", () => {
    const empty = new Map();
    const pos = new Map<string, [number, number]>();
    const enter = diffGeometryTelemetry(
      [mark("mark:1", 1, 1)],
      [square],
      [],
      empty,
      pos,
      1000,
      0,
    );
    expect(enter.events).toHaveLength(1);
    expect(enter.events[0]).toMatchObject({
      kind: "region-enter",
      topicKey: TOPIC_KEY_REGION,
      trackId: "mark:1",
      entityName: "Box",
    });

    const exit = diffGeometryTelemetry(
      [mark("mark:1", 5, 5)],
      [square],
      [],
      enter.occupancy,
      enter.prevPos,
      2000,
      enter.nextSeq,
    );
    expect(exit.events.some((e) => e.kind === "region-exit")).toBe(true);
  });

  it("emits tripwire-cross when a mark segment crosses the wire", () => {
    const pos = new Map<string, [number, number]>([["mark:1", [0, 1]]]);
    const { events } = diffGeometryTelemetry(
      [mark("mark:1", 2, 1)],
      [],
      [wire],
      new Map(),
      pos,
      1000,
      0,
    );
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({
      kind: "tripwire-cross",
      topicKey: TOPIC_KEY_TRIPWIRE,
      entityName: "Door",
    });
  });
});
