// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { create } from "zustand";
import { mqttPayloadToJson, type MqttClientLike } from "../mqtt/client";
import type { SceneSensorBootstrap } from "../scene/types";

export const MIN_LIGHT_INTENSITY = 0.1;
export const MAX_LIGHT_INTENSITY = 3;
export const DEFAULT_LIGHT_INTENSITY = 1;
export const LIGHT_INTENSITY_STEP = 0.01;
const SENSOR_TOPIC = "scenescape/data/sensor/";

export const useLightingStore = create<{
  intensity: number;
  setIntensity: (intensity: number) => void;
}>()((set) => ({
  intensity: DEFAULT_LIGHT_INTENSITY,
  setIntensity: intensity => {
    if (!Number.isFinite(intensity)) return;
    set({ intensity: Math.max(MIN_LIGHT_INTENSITY, Math.min(MAX_LIGHT_INTENSITY, intensity)) });
  },
}));

export function lightIntensityFromSensor(
  topic: string,
  payload: unknown,
  sensors: SceneSensorBootstrap[],
): number | null {
  if (!topic.startsWith(SENSOR_TOPIC) || !payload || typeof payload !== "object" || Array.isArray(payload)) {
    return null;
  }
  const sensorId = topic.slice(SENSOR_TOPIC.length);
  const sensor = sensors.find(sensor => sensor.sensorId === sensorId);
  if (!sensor) return null;
  const message = payload as Record<string, unknown>;
  if (message.subtype !== "light" && !sensorId.toLowerCase().includes("_light")) return null;
  try {
    if (JSON.parse(sensor.areaJson).area !== "scene") return null;
  } catch {
    return null;
  }
  const lux = typeof message.value === "number"
    ? message.value : typeof message.value === "string" ? Number.parseFloat(message.value) : NaN;
  return Number.isFinite(lux)
    ? Math.max(MIN_LIGHT_INTENSITY, Math.min(MAX_LIGHT_INTENSITY, lux / 500))
    : null;
}

export function attachLightingMqtt(sensors: SceneSensorBootstrap[]): () => void {
  let attached: MqttClientLike | undefined;
  const onMessage = (topic: unknown, payload: unknown) => {
    if (typeof topic !== "string") return;
    const intensity = lightIntensityFromSensor(topic, mqttPayloadToJson(payload), sensors);
    if (intensity !== null) useLightingStore.getState().setIntensity(intensity);
  };
  const onConnect = () => attached?.subscribe(`${SENSOR_TOPIC}+`);
  const detach = () => {
    if (attached?.removeListener) {
      attached.removeListener("message", onMessage);
      attached.removeListener("connect", onConnect);
    } else {
      attached?.off?.("message", onMessage);
      attached?.off?.("connect", onConnect);
    }
  };
  const attach = () => {
    const client = window.ssMqttClient;
    if (!client || client === attached) return;
    detach();
    attached = client;
    client.on("message", onMessage);
    client.on("connect", onConnect);
    if (client.connected) onConnect();
  };
  window.addEventListener("ss-mqtt-connected", attach);
  window.addEventListener("ss-mqtt-status", attach);
  attach();
  return () => {
    window.removeEventListener("ss-mqtt-connected", attach);
    window.removeEventListener("ss-mqtt-status", attach);
    detach();
  };
}