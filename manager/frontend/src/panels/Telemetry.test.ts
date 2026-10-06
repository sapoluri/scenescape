// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import type { MarkEntity } from "../viewport/types";
import {
  capTelemetryEvents,
  diffTelemetryBatch,
  MAX_TELEMETRY_EVENTS,
  summarizeMarks,
  type TelemetryEvent,
} from "./Telemetry";

function mark(
  id: string,
  className: string,
  opts: Partial<MarkEntity> = {},
): MarkEntity {
  return {
    id,
    type: "mark",
    name: className,
    visible: true,
    className,
    color: "#64d2ff",
    position: [1, 2, 0],
    heading: Number.NaN,
    speed: 0,
    ...opts,
  };
}

describe("diffTelemetryBatch", () => {
  it("emits track-new for unseen ids, newest first", () => {
    const { events, known, nextSeq } = diffTelemetryBatch(
      new Set(),
      [mark("mark:1", "person"), mark("mark:2", "forklift")],
      1000,
      0,
    );
    expect(events).toHaveLength(2);
    expect(events.every((e) => e.kind === "track-new")).toBe(true);
    // Newest first: mark:2 was later in the batch.
    expect(events[0]?.trackId).toBe("mark:2");
    expect(events[1]?.trackId).toBe("mark:1");
    expect(events[0]?.seq).toBeGreaterThan(events[1]?.seq ?? -1);
    expect(events[0]?.at).toBe(1000);
    expect(events[0]?.className).toBe("forklift");
    expect(events[0]?.position).toEqual([1, 2]);
    expect(known).toEqual(new Set(["mark:1", "mark:2"]));
    expect(nextSeq).toBe(2);
  });

  it("does not re-emit track-new for already-known ids", () => {
    const known = new Set(["mark:1"]);
    const { events } = diffTelemetryBatch(
      known,
      [mark("mark:1", "person", { position: [3, 4, 0], speed: 0.5 })],
      2000,
      7,
    );
    expect(events).toHaveLength(0);
  });

  it("emits track-lost for ids that disappeared", () => {
    const known = new Set(["mark:1", "mark:2"]);
    const { events, known: next } = diffTelemetryBatch(
      known,
      [mark("mark:2", "forklift")],
      3000,
      0,
    );
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({
      kind: "track-lost",
      trackId: "mark:1",
      at: 3000,
    });
    expect(next).toEqual(new Set(["mark:2"]));
  });

  it("omits speed when zero and includes it when positive", () => {
    const { events } = diffTelemetryBatch(
      new Set(),
      [
        mark("mark:1", "person", { speed: 0 }),
        mark("mark:2", "person", { speed: 1.25 }),
      ],
      1000,
      0,
    );
    const byId = new Map(events.map((e) => [e.trackId, e]));
    expect(byId.get("mark:1")?.speed).toBeUndefined();
    expect(byId.get("mark:2")?.speed).toBeCloseTo(1.25);
  });

  it("handles an empty batch by marking everything lost", () => {
    const { events } = diffTelemetryBatch(
      new Set(["mark:9"]),
      [],
      4000,
      0,
    );
    expect(events).toHaveLength(1);
    expect(events[0]?.kind).toBe("track-lost");
  });
});

describe("capTelemetryEvents", () => {
  it("keeps the newest MAX_TELEMETRY_EVENTS entries", () => {
    const events: TelemetryEvent[] = Array.from(
      { length: MAX_TELEMETRY_EVENTS + 50 },
      (_, i): TelemetryEvent => ({
        seq: i,
        at: i,
        kind: "track-new",
        trackId: `mark:${i}`,
        className: "person",
      }),
    ).reverse(); // newest first, seq desc
    const capped = capTelemetryEvents(events);
    expect(capped).toHaveLength(MAX_TELEMETRY_EVENTS);
    expect(capped[0]?.seq).toBe(MAX_TELEMETRY_EVENTS + 49);
    expect(capped[capped.length - 1]?.seq).toBe(50);
  });

  it("leaves short feeds untouched", () => {
    const events: TelemetryEvent[] = [
      { seq: 0, at: 0, kind: "track-new", trackId: "mark:1", className: "x" },
    ];
    expect(capTelemetryEvents(events)).toBe(events);
  });
});

describe("summarizeMarks", () => {
  it("counts per class, sorted by count desc", () => {
    const summary = summarizeMarks([
      mark("mark:1", "person", { color: "#ff0000" }),
      mark("mark:2", "forklift", { color: "#0000ff" }),
      mark("mark:3", "person", { color: "#ff0000" }),
    ]);
    expect(summary).toEqual([
      { className: "person", count: 2, color: "#ff0000" },
      { className: "forklift", count: 1, color: "#0000ff" },
    ]);
  });

  it("returns an empty list when there are no marks", () => {
    expect(summarizeMarks([])).toEqual([]);
  });
});
