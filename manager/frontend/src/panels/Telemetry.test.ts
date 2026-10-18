// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from "vitest";
import {
  parseSceneEventTopic,
  sceneEventPayloadToFeed,
} from "../mqtt/useSceneEventTelemetry";
import type { MarkEntity } from "../viewport/types";
import {
  brokerConnectDetails,
  capTelemetryEvents,
  defaultSelectedTopicKeys,
  diffTelemetryBatch,
  filterTelemetryByTopics,
  filterTelemetryEvents,
  MAX_TELEMETRY_EVENTS,
  mqttNativeUrl,
  sceneDataTopic,
  sceneTopicCatalog,
  summarizeMarks,
  toggleTopicKey,
  TOPIC_KEY_REGION,
  TOPIC_KEY_REGULATED,
  TOPIC_KEY_SCENE_DATA,
  TOPIC_KEY_TRIPWIRE,
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

function ev(
  partial: Omit<TelemetryEvent, "topicKey"> & { topicKey?: string },
): TelemetryEvent {
  return {
    topicKey: TOPIC_KEY_REGULATED,
    ...partial,
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
    expect(events.every((e) => e.topicKey === TOPIC_KEY_REGULATED)).toBe(true);
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
      topicKey: TOPIC_KEY_REGULATED,
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
      (_, i): TelemetryEvent =>
        ev({
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
      ev({
        seq: 0,
        at: 0,
        kind: "track-new",
        trackId: "mark:1",
        className: "x",
      }),
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

describe("filterTelemetryEvents", () => {
  const sample: TelemetryEvent[] = [
    ev({
      seq: 2,
      at: 2,
      kind: "track-new",
      trackId: "mark:2",
      className: "forklift",
    }),
    ev({
      seq: 1,
      at: 1,
      kind: "track-lost",
      trackId: "mark:1",
      className: "person",
    }),
    ev({
      seq: 0,
      at: 0,
      kind: "track-new",
      trackId: "mark:0",
      className: "person",
    }),
  ];

  it("returns all events when filter is null", () => {
    expect(filterTelemetryEvents(sample, null)).toBe(sample);
  });

  it("keeps only matching class names", () => {
    expect(filterTelemetryEvents(sample, "person")).toEqual([
      sample[1],
      sample[2],
    ]);
    expect(filterTelemetryEvents(sample, "forklift")).toEqual([sample[0]]);
  });

  it("returns an empty list when nothing matches", () => {
    expect(filterTelemetryEvents(sample, "bicycle")).toEqual([]);
  });

  it("keeps events without a class when a filter is active", () => {
    const bare = ev({
      seq: 9,
      at: 9,
      kind: "region-enter",
      topicKey: TOPIC_KEY_REGION,
      trackId: "",
      className: "",
      entityId: "reg-1",
      entityName: "Lobby",
    });
    expect(filterTelemetryEvents([...sample, bare], "person")).toEqual([
      sample[1],
      sample[2],
      bare,
    ]);
  });
});

describe("filterTelemetryByTopics", () => {
  const sample: TelemetryEvent[] = [
    ev({
      seq: 2,
      at: 2,
      kind: "track-new",
      trackId: "mark:2",
      className: "person",
    }),
    ev({
      seq: 1,
      at: 1,
      kind: "region-enter",
      topicKey: TOPIC_KEY_REGION,
      trackId: "mark:1",
      className: "person",
      entityId: "r1",
    }),
    ev({
      seq: 0,
      at: 0,
      kind: "tripwire-cross",
      topicKey: TOPIC_KEY_TRIPWIRE,
      trackId: "mark:0",
      className: "person",
      entityId: "t1",
    }),
  ];

  it("defaults to regulated, region, and tripwire feed topics", () => {
    expect(defaultSelectedTopicKeys()).toEqual([
      TOPIC_KEY_REGULATED,
      TOPIC_KEY_REGION,
      TOPIC_KEY_TRIPWIRE,
    ]);
    expect(
      filterTelemetryByTopics(sample, defaultSelectedTopicKeys()),
    ).toEqual(sample);
  });

  it("supports multi-select", () => {
    expect(
      filterTelemetryByTopics(sample, [
        TOPIC_KEY_REGULATED,
        TOPIC_KEY_REGION,
      ]),
    ).toEqual([sample[0], sample[1]]);
  });

  it("returns empty when nothing is selected", () => {
    expect(filterTelemetryByTopics(sample, [])).toEqual([]);
  });

  it("treats scene-data as an alias for the regulated track stream", () => {
    expect(filterTelemetryByTopics(sample, [TOPIC_KEY_SCENE_DATA])).toEqual([
      sample[0],
    ]);
  });

  it("toggles keys without forcing a selection", () => {
    expect(toggleTopicKey([TOPIC_KEY_REGULATED], TOPIC_KEY_REGION)).toEqual([
      TOPIC_KEY_REGULATED,
      TOPIC_KEY_REGION,
    ]);
    expect(toggleTopicKey([TOPIC_KEY_REGULATED], TOPIC_KEY_REGULATED)).toEqual(
      [],
    );
  });
});

describe("sceneTopicCatalog", () => {
  it("lists scene data/event topics and sensors (no camera/cmd/image)", () => {
    const catalog = sceneTopicCatalog("scene-1", [
      { name: "Temp", sensorId: "temp_1" },
    ]);
    const topics = catalog.map((c) => c.topic);
    const keys = catalog.map((c) => c.key);
    expect(topics).toContain("scenescape/regulated/scene/scene-1");
    expect(topics).toContain("scenescape/data/scene/scene-1/+");
    expect(topics).toContain("scenescape/event/region/scene-1/+/+");
    expect(topics).toContain("scenescape/event/tripwire/scene-1/+/+");
    expect(topics).toContain("scenescape/data/sensor/temp_1");
    expect(topics.some((t) => t.includes("/data/camera/"))).toBe(false);
    expect(topics.some((t) => t.includes("/cmd/"))).toBe(false);
    expect(topics.some((t) => t.includes("/image/"))).toBe(false);
    expect(keys[0]).toBe(TOPIC_KEY_REGULATED);
  });
});

describe("scene event MQTT parsing", () => {
  it("parses region and tripwire topics for the active scene", () => {
    expect(
      parseSceneEventTopic(
        "scenescape/event/region/scene-1/reg-9/objects",
        "scene-1",
      ),
    ).toEqual({
      kind: "region",
      sceneId: "scene-1",
      entityId: "reg-9",
    });
    expect(
      parseSceneEventTopic(
        "scenescape/event/tripwire/scene-1/tw-1/objects",
        "other",
      ),
    ).toBeNull();
  });

  it("emits enter/exit rows and skips count-only region updates", () => {
    const parsed = {
      kind: "region" as const,
      sceneId: "scene-1",
      entityId: "reg-9",
    };
    const empty = sceneEventPayloadToFeed(
      parsed,
      { counts: { person: 1 }, entered: [], exited: [] },
      1000,
      0,
    );
    expect(empty.events).toEqual([]);

    const { events, nextSeq } = sceneEventPayloadToFeed(
      parsed,
      {
        region_name: "Lobby",
        entered: [{ id: "42", category: "person" }],
        exited: [{ object: { id: "7", type: "person" } }],
      },
      2000,
      0,
    );
    expect(nextSeq).toBe(2);
    expect(events).toHaveLength(2);
    expect(events[0]?.kind).toBe("region-exit");
    expect(events[0]?.topicKey).toBe(TOPIC_KEY_REGION);
    expect(events[0]?.trackId).toBe("mark:7");
    expect(events[1]?.kind).toBe("region-enter");
    expect(events[1]?.trackId).toBe("mark:42");
    expect(events[1]?.entityName).toBe("Lobby");
  });

  it("emits tripwire-cross only when objects have direction", () => {
    const parsed = {
      kind: "tripwire" as const,
      sceneId: "scene-1",
      entityId: "tw-1",
    };
    expect(
      sceneEventPayloadToFeed(
        parsed,
        { objects: [{ id: "1", category: "person" }] },
        1000,
        0,
      ).events,
    ).toEqual([]);
    const { events } = sceneEventPayloadToFeed(
      parsed,
      {
        tripwire_name: "Door",
        objects: [{ id: "1", category: "person", direction: "+" }],
      },
      1000,
      0,
    );
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({
      kind: "tripwire-cross",
      topicKey: TOPIC_KEY_TRIPWIRE,
      trackId: "mark:1",
      entityName: "Door",
    });
  });
});

describe("broker connect helpers", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("builds the regulated scene topic", () => {
    expect(sceneDataTopic("abc")).toBe("scenescape/regulated/scene/abc");
    expect(sceneDataTopic("")).toBe("scenescape/regulated/scene/");
  });

  it("builds the native MQTT URL on 1883", () => {
    expect(mqttNativeUrl("broker.example")).toBe("mqtt://broker.example:1883");
    expect(mqttNativeUrl("  ")).toBe("mqtt://localhost:1883");
  });

  it("assembles websocket and mqtt URLs for display", () => {
    vi.stubGlobal("window", {
      location: {
        hostname: "scene.local",
        port: "",
        protocol: "https:",
      },
    });
    const details = brokerConnectDetails(
      "wss://localhost/mqtt",
      "scene.local",
    );
    expect(details.websocketUrl).toBe("wss://scene.local/mqtt");
    expect(details.mqttUrl).toBe("mqtt://scene.local:1883");
    expect("topic" in details).toBe(false);
  });
});
