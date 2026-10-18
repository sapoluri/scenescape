// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { rewriteBrokerUrl } from "../mqtt/client";
import {
  sceneDataOutputTopic,
  sceneRegionEventTopic,
  sceneRegulatedTopic,
  sceneTripwireEventTopic,
  sensorDataTopic,
} from "../mqtt/topics";
import type { SceneSensorBootstrap } from "../scene/types";
import type { MarkEntity } from "../viewport/types";

export const MAX_TELEMETRY_EVENTS = 200;

/** Catalog / feed key for the regulated scene track stream (default filter). */
export const TOPIC_KEY_REGULATED = "regulated";
export const TOPIC_KEY_SCENE_DATA = "scene-data";
export const TOPIC_KEY_REGION = "region-events";
export const TOPIC_KEY_TRIPWIRE = "tripwire-events";

export type TelemetryEventKind =
  | "track-new"
  | "track-lost"
  | "region-enter"
  | "region-exit"
  | "tripwire-cross";

export interface TelemetryEvent {
  /** Monotonic sequence number, unique within the feed. */
  seq: number;
  /** Epoch ms when the event was recorded. */
  at: number;
  kind: TelemetryEventKind;
  /** Catalog topic key this event belongs to (for multi-select filter). */
  topicKey: string;
  /** Store mark id, e.g. "mark:42". Empty when unknown. */
  trackId: string;
  /** Asset3D class name, e.g. "person". */
  className: string;
  /** Region / tripwire entity id when from scene events. */
  entityId?: string;
  /** Region / tripwire display name when known. */
  entityName?: string;
  /** Last known position [x, y] meters, when known. */
  position?: [number, number];
  /** Per-batch displacement in meters, when known and nonzero. */
  speed?: number;
}

export interface ClassSummary {
  className: string;
  count: number;
  color: string;
}

export type BrokerConnectDetails = {
  websocketUrl: string;
  mqttUrl: string;
};

export type TopicCatalogEntry = {
  /** Stable id used for multi-select feed filtering. */
  key: string;
  label: string;
  topic: string;
};

/** Regulated scene topic string (marks / track feed). */
export function sceneDataTopic(sceneId: string): string {
  return sceneId
    ? sceneRegulatedTopic(sceneId)
    : "scenescape/regulated/scene/";
}

/** Native MQTT URL for external clients (port 1883). */
export function mqttNativeUrl(hostname: string): string {
  const host = hostname.trim() || "localhost";
  return `mqtt://${host}:1883`;
}

export function brokerConnectDetails(
  wssConnection: string,
  hostname: string = typeof window !== "undefined"
    ? window.location.hostname
    : "localhost",
): BrokerConnectDetails {
  return {
    websocketUrl: rewriteBrokerUrl(wssConnection || "wss://localhost/mqtt"),
    mqttUrl: mqttNativeUrl(hostname),
  };
}

/**
 * Default topic filter: all topics that currently drive the feed.
 * Camera/sensor/scene-data rows stay unselected (copyable; no feed rows yet).
 */
export function defaultSelectedTopicKeys(): string[] {
  return [TOPIC_KEY_REGULATED, TOPIC_KEY_REGION, TOPIC_KEY_TRIPWIRE];
}

/** Topic key for a feed event kind. */
export function topicKeyForKind(kind: TelemetryEventKind): string {
  switch (kind) {
    case "track-new":
    case "track-lost":
      return TOPIC_KEY_REGULATED;
    case "region-enter":
    case "region-exit":
      return TOPIC_KEY_REGION;
    case "tripwire-cross":
      return TOPIC_KEY_TRIPWIRE;
    default:
      return TOPIC_KEY_REGULATED;
  }
}

/**
 * Integrator-facing scene topics (data + events). Omits cmd/image — those are
 * internal or binary and belong in an MQTT explorer for advanced use.
 */
export function sceneTopicCatalog(
  sceneId: string,
  sensors: Pick<SceneSensorBootstrap, "name" | "sensorId">[] = [],
): TopicCatalogEntry[] {
  const sid = sceneId || "";
  const out: TopicCatalogEntry[] = [
    {
      key: TOPIC_KEY_REGULATED,
      label: "Regulated scene",
      topic: sceneRegulatedTopic(sid),
    },
    {
      key: TOPIC_KEY_SCENE_DATA,
      label: "Scene data",
      topic: sceneDataOutputTopic(sid),
    },
    {
      key: TOPIC_KEY_REGION,
      label: "Region events",
      topic: sceneRegionEventTopic(sid),
    },
    {
      key: TOPIC_KEY_TRIPWIRE,
      label: "Tripwire events",
      topic: sceneTripwireEventTopic(sid),
    },
  ];
  for (const s of sensors) {
    const id = s.sensorId || s.name;
    if (!id) {
      continue;
    }
    out.push({
      key: `sensor-data:${id}`,
      label: `${s.name || id} · data`,
      topic: sensorDataTopic(id),
    });
  }
  return out;
}

export function toggleTopicKey(
  selected: readonly string[],
  key: string,
): string[] {
  if (selected.includes(key)) {
    return selected.filter((k) => k !== key);
  }
  return [...selected, key];
}

export function capTelemetryEvents(events: TelemetryEvent[]): TelemetryEvent[] {
  return events.length > MAX_TELEMETRY_EVENTS
    ? events.slice(0, MAX_TELEMETRY_EVENTS)
    : events;
}

/**
 * Filter feed by class; `null` means all classes. Events without a className
 * always pass (region/tripwire rows may omit an object class).
 */
export function filterTelemetryEvents(
  events: TelemetryEvent[],
  className: string | null,
): TelemetryEvent[] {
  if (!className) {
    return events;
  }
  return events.filter((e) => !e.className || e.className === className);
}

/**
 * Keep events whose topicKey is in the multi-select set. Empty set → [].
 * Scene-data is treated as an alias for the regulated track stream (same
 * objects; unregulated topic is for integrators copying the pattern).
 */
export function filterTelemetryByTopics(
  events: TelemetryEvent[],
  selectedKeys: readonly string[],
): TelemetryEvent[] {
  if (selectedKeys.length === 0) {
    return [];
  }
  const keys = new Set(selectedKeys);
  if (keys.has(TOPIC_KEY_SCENE_DATA)) {
    keys.add(TOPIC_KEY_REGULATED);
  }
  return events.filter((e) => keys.has(e.topicKey));
}

export function summarizeMarks(marks: MarkEntity[]): ClassSummary[] {
  const acc = new Map<string, ClassSummary>();
  for (const m of marks) {
    const cur = acc.get(m.className);
    if (cur) {
      cur.count += 1;
    } else {
      acc.set(m.className, {
        className: m.className,
        count: 1,
        color: m.color,
      });
    }
  }
  return [...acc.values()].sort(
    (a, b) => b.count - a.count || a.className.localeCompare(b.className),
  );
}

export function kindLabel(kind: TelemetryEventKind): string {
  switch (kind) {
    case "track-new":
      return "appeared";
    case "track-lost":
      return "lost";
    case "region-enter":
      return "entered";
    case "region-exit":
      return "exited";
    case "tripwire-cross":
      return "crossed";
    default:
      return kind;
  }
}
