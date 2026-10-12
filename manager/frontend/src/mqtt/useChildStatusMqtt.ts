// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from "react";
import {
  mqttPayloadToString,
  type MqttClientLike,
} from "./client";
import {
  getChildStatusState,
  reduceChildStatusPayload,
} from "./childStatusStore";
import { APP_NAME, SYS_CHILDSCENE_STATUS, childStatusTopic } from "./topics";

/**
 * Subscribe to `scenescape/sys/child/status/+` and probe remote children the
 * same way legacy sscape.js did (`publish(..., "isConnected")`). Updates
 * `useChildStatusStore` for Outliner / Properties badges.
 */
export function useChildStatusMqtt(remoteChildIds: string[]): void {
  const remoteKey = remoteChildIds.filter(Boolean).sort().join(",");

  useEffect(() => {
    if (!remoteKey) {
      return;
    }
    const ids = remoteKey.split(",").filter(Boolean);
    let client: MqttClientLike | null = null;
    let attached: MqttClientLike | null = null;

    const onMessage = (topic: unknown, payload: unknown) => {
      const t = typeof topic === "string" ? topic : String(topic ?? "");
      if (!t.includes(SYS_CHILDSCENE_STATUS)) {
        return;
      }
      const remoteId = t.slice(t.lastIndexOf("/") + 1);
      if (!remoteId) {
        return;
      }
      const text = mqttPayloadToString(payload);
      // Ignore our own probe publishes echoed back if the broker reflects them
      // as the literal probe string only when not yet answered.
      getChildStatusState().setStatus(remoteId, reduceChildStatusPayload(text));
    };

    const probe = (c: MqttClientLike) => {
      for (const id of ids) {
        try {
          c.publish(`${APP_NAME}${SYS_CHILDSCENE_STATUS}/${id}`, "isConnected");
        } catch {
          /* ignore */
        }
      }
    };

    const attach = (c: MqttClientLike) => {
      if (attached === c) {
        return;
      }
      if (attached) {
        attached.removeListener?.("message", onMessage);
        attached.off?.("message", onMessage);
      }
      attached = c;
      try {
        c.subscribe(childStatusTopic());
      } catch {
        /* already subscribed */
      }
      c.on("message", onMessage);
      if (c.connected) {
        probe(c);
      } else {
        c.on("connect", () => probe(c));
      }
    };

    const tryAttach = () => {
      const c = window.ssMqttClient;
      if (c && c !== client) {
        client = c;
        attach(c);
      }
    };

    tryAttach();
    const onMqttUp = () => tryAttach();
    window.addEventListener("ss-mqtt-connected", onMqttUp);
    window.addEventListener("ss-mqtt-status", onMqttUp);
    const poll = window.setInterval(tryAttach, 1000);

    return () => {
      window.clearInterval(poll);
      window.removeEventListener("ss-mqtt-connected", onMqttUp);
      window.removeEventListener("ss-mqtt-status", onMqttUp);
      if (attached) {
        attached.removeListener?.("message", onMessage);
        attached.off?.("message", onMessage);
      }
    };
  }, [remoteKey]);
}
