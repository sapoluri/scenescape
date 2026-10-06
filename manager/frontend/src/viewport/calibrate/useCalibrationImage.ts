// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from "react";
import {
  mqttPayloadToJson,
  type MqttClientLike,
} from "../../mqtt/client";
import { APP_NAME, CMD_CAMERA, IMAGE_CALIBRATE } from "../../mqtt/topics";

/**
 * Subscribe to calibration JPEG frames for one camera and keep requesting
 * getcalibrationimage while active (same MQTT contract as the Django
 * calibrate page).
 */
export function useCalibrationImage(
  sensorId: string | null,
  enabled: boolean,
): { imageUrl: string | null; naturalSize: { w: number; h: number } | null } {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [naturalSize, setNaturalSize] = useState<{
    w: number;
    h: number;
  } | null>(null);
  const urlRef = useRef<string | null>(null);

  useEffect(() => {
    if (!enabled || !sensorId) {
      return;
    }
    let cancelled = false;
    let client: MqttClientLike | null = null;
    let interval: number | null = null;

    const revoke = () => {
      if (urlRef.current) {
        URL.revokeObjectURL(urlRef.current);
        urlRef.current = null;
      }
    };

    const request = (c: MqttClientLike) => {
      try {
        c.subscribe(`${APP_NAME}${IMAGE_CALIBRATE}+`);
      } catch {
        /* already subscribed */
      }
      c.publish(`${APP_NAME}${CMD_CAMERA}${sensorId}`, "getcalibrationimage");
    };

    const onMessage = (topic: unknown, data: unknown) => {
      const t = String(topic || "");
      if (!t.includes(`${IMAGE_CALIBRATE}${sensorId}`)) {
        return;
      }
      const parsed = mqttPayloadToJson(data);
      const b64 =
        parsed && typeof parsed === "object"
          ? (parsed as { image?: string }).image
          : null;
      if (!b64 || typeof b64 !== "string") {
        return;
      }
      const bin = atob(b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) {
        bytes[i] = bin.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: "image/jpeg" });
      const url = URL.createObjectURL(blob);
      if (cancelled) {
        URL.revokeObjectURL(url);
        return;
      }
      revoke();
      urlRef.current = url;
      setImageUrl(url);
      const img = new Image();
      img.onload = () => {
        if (!cancelled) {
          setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
        }
      };
      img.src = url;
    };

    const bind = (c: MqttClientLike | undefined) => {
      if (!c || cancelled) {
        return;
      }
      client = c;
      c.on("message", onMessage);
      request(c);
      interval = window.setInterval(() => request(c), 1500);
    };

    const onConnected = () => bind(window.ssMqttClient);
    window.addEventListener("ss-mqtt-connected", onConnected);
    bind(window.ssMqttClient);

    return () => {
      cancelled = true;
      window.removeEventListener("ss-mqtt-connected", onConnected);
      if (interval != null) {
        window.clearInterval(interval);
      }
      if (client) {
        client.removeListener?.("message", onMessage);
        client.off?.("message", onMessage);
      }
      revoke();
      setImageUrl(null);
      setNaturalSize(null);
    };
  }, [enabled, sensorId]);

  return { imageUrl, naturalSize };
}
