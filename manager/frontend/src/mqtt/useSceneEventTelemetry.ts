// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from "react";
import {
  capTelemetryEvents,
  TOPIC_KEY_REGION,
  TOPIC_KEY_TRIPWIRE,
  type TelemetryEvent,
} from "../panels/telemetryModel";
import { getViewportState } from "../viewport/store";
import { markKey } from "../viewport/marks";
import { mqttPayloadToJson, type MqttClientLike } from "./client";
import { APP_NAME } from "./topics";

type ParsedTopic = {
  kind: "region" | "tripwire";
  sceneId: string;
  entityId: string;
};

/** Pure: parse `scenescape/event/{region|tripwire}/{sceneId}/{entityId}/…`. */
export function parseSceneEventTopic(
  topic: string,
  expectSceneId: string,
): ParsedTopic | null {
  const parts = topic.split("/");
  if (parts.length < 6 || parts[0] !== APP_NAME || parts[1] !== "event") {
    return null;
  }
  const kind = parts[2];
  if (kind !== "region" && kind !== "tripwire") {
    return null;
  }
  if (parts[3] !== expectSceneId) {
    return null;
  }
  const entityId = parts[4];
  if (!entityId) {
    return null;
  }
  return { kind, sceneId: parts[3], entityId };
}

function asRecord(v: unknown): Record<string, unknown> | null {
  return v && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : null;
}

function objectClass(obj: Record<string, unknown>): string {
  if (typeof obj.category === "string" && obj.category) {
    return obj.category;
  }
  if (typeof obj.type === "string" && obj.type) {
    return obj.type;
  }
  return "object";
}

function objectTrackId(obj: Record<string, unknown>): string {
  if (obj.id === undefined || obj.id === null) {
    return "";
  }
  return markKey(obj.id as string | number);
}

function unwrapExited(item: unknown): Record<string, unknown> | null {
  const row = asRecord(item);
  if (!row) {
    return null;
  }
  const nested = asRecord(row.object);
  return nested ?? row;
}

function entityDisplayName(
  entityId: string,
  payloadName: unknown,
): string | undefined {
  const ent = getViewportState().entities[entityId];
  if (ent && (ent.type === "region" || ent.type === "tripwire")) {
    return ent.name;
  }
  return typeof payloadName === "string" && payloadName
    ? payloadName
    : undefined;
}

/**
 * Turn a region/tripwire MQTT payload into feed rows (newest first).
 * Count-only updates (empty entered/exited/objects) yield [].
 */
export function sceneEventPayloadToFeed(
  parsed: ParsedTopic,
  payload: unknown,
  now: number,
  seqStart: number,
): { events: TelemetryEvent[]; nextSeq: number } {
  const msg = asRecord(payload);
  if (!msg) {
    return { events: [], nextSeq: seqStart };
  }
  let seq = seqStart;
  const out: TelemetryEvent[] = [];
  const entityName =
    parsed.kind === "region"
      ? entityDisplayName(parsed.entityId, msg.region_name)
      : entityDisplayName(parsed.entityId, msg.tripwire_name);

  if (parsed.kind === "region") {
    const entered = Array.isArray(msg.entered) ? msg.entered : [];
    const exited = Array.isArray(msg.exited) ? msg.exited : [];
    for (const item of entered) {
      const obj = asRecord(item);
      if (!obj) {
        continue;
      }
      out.push({
        seq: seq++,
        at: now,
        kind: "region-enter",
        topicKey: TOPIC_KEY_REGION,
        trackId: objectTrackId(obj),
        className: objectClass(obj),
        entityId: parsed.entityId,
        entityName,
      });
    }
    for (const item of exited) {
      const obj = unwrapExited(item);
      if (!obj) {
        continue;
      }
      out.push({
        seq: seq++,
        at: now,
        kind: "region-exit",
        topicKey: TOPIC_KEY_REGION,
        trackId: objectTrackId(obj),
        className: objectClass(obj),
        entityId: parsed.entityId,
        entityName,
      });
    }
  } else {
    const objects = Array.isArray(msg.objects) ? msg.objects : [];
    for (const item of objects) {
      const obj = asRecord(item);
      if (!obj || obj.direction === undefined || obj.direction === null) {
        continue;
      }
      out.push({
        seq: seq++,
        at: now,
        kind: "tripwire-cross",
        topicKey: TOPIC_KEY_TRIPWIRE,
        trackId: objectTrackId(obj),
        className: objectClass(obj),
        entityId: parsed.entityId,
        entityName,
      });
    }
  }

  out.reverse();
  return { events: out, nextSeq: seq };
}

function detachMessage(
  client: MqttClientLike,
  fn: (...args: unknown[]) => void,
): void {
  if (typeof client.removeListener === "function") {
    client.removeListener("message", fn);
  } else if (typeof client.off === "function") {
    client.off("message", fn);
  }
}

/**
 * Listen on the shared scene MQTT client for region/tripwire events.
 * Does not subscribe (legacy attachLegacySceneHandlers already does).
 */
export function useSceneEventTelemetry(sceneId: string): TelemetryEvent[] {
  const [events, setEvents] = useState<TelemetryEvent[]>([]);
  const seqRef = useRef(0);

  useEffect(() => {
    seqRef.current = 0;
    setEvents([]);
  }, [sceneId]);

  useEffect(() => {
    if (!sceneId) {
      return;
    }

    let attached: MqttClientLike | null = null;
    let onMessage: ((...args: unknown[]) => void) | null = null;

    const attach = (client: MqttClientLike) => {
      if (client === attached) {
        return;
      }
      if (attached && onMessage) {
        detachMessage(attached, onMessage);
      }
      onMessage = (topic: unknown, data: unknown) => {
        const parsed = parseSceneEventTopic(String(topic || ""), sceneId);
        if (!parsed) {
          return;
        }
        const { events: fresh, nextSeq } = sceneEventPayloadToFeed(
          parsed,
          mqttPayloadToJson(data),
          Date.now(),
          seqRef.current,
        );
        seqRef.current = nextSeq;
        if (fresh.length === 0) {
          return;
        }
        setEvents((prev) => capTelemetryEvents([...fresh, ...prev]));
      };
      client.on("message", onMessage);
      attached = client;
    };

    const tryAttach = () => {
      const client = window.ssMqttClient as MqttClientLike | undefined;
      if (client) {
        attach(client);
      }
    };

    tryAttach();
    window.addEventListener("ss-mqtt-connected", tryAttach);
    window.addEventListener("ss-mqtt-status", tryAttach);
    const poll = window.setInterval(tryAttach, 1000);

    return () => {
      window.clearInterval(poll);
      window.removeEventListener("ss-mqtt-connected", tryAttach);
      window.removeEventListener("ss-mqtt-status", tryAttach);
      if (attached && onMessage) {
        detachMessage(attached, onMessage);
      }
    };
  }, [sceneId]);

  return events;
}
