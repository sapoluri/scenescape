// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Thin wrapper around the global mqtt.js (mqtt.min.js on base.html).
 * Scene detail owns connect; Snap still receives messages via ssAttachSceneMqttClient.
 */

export type MqttClientLike = {
  subscribe: (topic: string) => void;
  publish: (topic: string, payload: string) => void;
  on: (ev: string, fn: (...args: unknown[]) => void) => void;
  removeListener?: (ev: string, fn: (...args: unknown[]) => void) => void;
  off?: (ev: string, fn: (...args: unknown[]) => void) => void;
  end?: (force?: boolean) => void;
};

type MqttGlobal = {
  connect: (url: string) => MqttClientLike;
};

function mqttGlobal(): MqttGlobal | null {
  const g = (window as unknown as { mqtt?: MqttGlobal }).mqtt;
  return g && typeof g.connect === "function" ? g : null;
}

export function rewriteBrokerUrl(raw: string): string {
  let broker = raw || "";
  const host = window.location.hostname;
  const port = window.location.port;
  const protocol = window.location.protocol;
  if (port && protocol === "https:") {
    broker = broker.replace("localhost", `${host}:${port}`);
  } else {
    broker = broker.replace("localhost", host);
  }
  if (protocol === "http:") {
    broker = broker.replace("wss:", "ws:");
    broker = broker.replace("/mqtt", ":1884");
  }
  return broker;
}

export function setMqttConnected(connected: boolean): void {
  const el = document.getElementById("mqtt_status");
  if (el) {
    el.classList.toggle("connected", connected);
  }
  document.querySelectorAll("[id^='mqtt_status']").forEach((node) => {
    if (!connected) {
      node.classList.remove("connected");
    }
  });
  window.dispatchEvent(
    new CustomEvent("ss-mqtt-status", { detail: { connected } }),
  );
}

export function endMqttClient(client: MqttClientLike | null | undefined): void {
  if (!client || typeof client.end !== "function") {
    return;
  }
  try {
    client.end(true);
  } catch {
    /* ignore */
  }
}

export function connectMqtt(brokerUrl: string): MqttClientLike | null {
  const api = mqttGlobal();
  if (!api) {
    console.error("mqtt.js not loaded");
    return null;
  }
  return api.connect(brokerUrl);
}

/** Ask legacy sscape.js to bind regulated/event/mark handlers on this client. */
export function attachLegacySceneHandlers(client: MqttClientLike): void {
  window.ssAttachSceneMqttClient?.(client);
}

declare global {
  interface Window {
    ssMqttClient?: MqttClientLike;
    ssAttachSceneMqttClient?: (client: MqttClientLike) => void;
    ssReactOwnsMqtt?: boolean;
    ssReactOwnsCameraStrip?: boolean;
  }
}
