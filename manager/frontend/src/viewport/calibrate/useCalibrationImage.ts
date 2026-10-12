// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from "react";
import {
  mqttPayloadToJson,
  type MqttClientLike,
} from "../../mqtt/client";
import { APP_NAME, CMD_CAMERA, IMAGE_CALIBRATE } from "../../mqtt/topics";

/** Cap request rate (~10 fps). Pipeline demos commonly run near this. */
const MIN_REQUEST_INTERVAL_MS = 100;
/** Re-request if a frame is lost so the feed recovers without a remount. */
const RECOVERY_INTERVAL_MS = 2000;

/**
 * Subscribe to calibration JPEG frames for one camera and keep requesting
 * getcalibrationimage while active (same MQTT contract as the Django
 * calibrate page).
 *
 * Legacy chains the next request from each delivered frame
 * (`$("#snapshot").trigger("click")` in updateCalibrationView). A fixed
 * 1.5s poll felt choppy; we chain on receipt and throttle to ~10 fps.
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
  const sizedRef = useRef(false);

  useEffect(() => {
    if (!enabled || !sensorId) {
      return;
    }
    let cancelled = false;
    let client: MqttClientLike | null = null;
    let recovery: number | null = null;
    let throttleTimer: number | null = null;
    let lastRequestAt = 0;
    sizedRef.current = false;

    const revoke = () => {
      if (urlRef.current) {
        URL.revokeObjectURL(urlRef.current);
        urlRef.current = null;
      }
    };

    const publishRequest = (c: MqttClientLike) => {
      try {
        c.subscribe(`${APP_NAME}${IMAGE_CALIBRATE}+`);
      } catch {
        /* already subscribed */
      }
      c.publish(`${APP_NAME}${CMD_CAMERA}${sensorId}`, "getcalibrationimage");
      lastRequestAt = performance.now();
    };

    /** Request immediately, or after the remaining throttle window. */
    const requestSoon = (c: MqttClientLike) => {
      if (cancelled) {
        return;
      }
      const wait = MIN_REQUEST_INTERVAL_MS - (performance.now() - lastRequestAt);
      if (wait > 0) {
        if (throttleTimer == null) {
          throttleTimer = window.setTimeout(() => {
            throttleTimer = null;
            if (!cancelled) {
              publishRequest(c);
            }
          }, wait);
        }
        return;
      }
      publishRequest(c);
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
      if (!sizedRef.current) {
        const img = new Image();
        img.onload = () => {
          if (!cancelled) {
            sizedRef.current = true;
            setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
          }
        };
        img.src = url;
      }
      // Chain the next frame (legacy snapshot click + live-view strip pattern).
      if (client) {
        requestSoon(client);
      }
    };

    const bind = (c: MqttClientLike | undefined) => {
      if (!c || cancelled) {
        return;
      }
      client = c;
      c.on("message", onMessage);
      publishRequest(c);
      if (recovery != null) {
        window.clearInterval(recovery);
      }
      recovery = window.setInterval(() => {
        if (client) {
          requestSoon(client);
        }
      }, RECOVERY_INTERVAL_MS);
    };

    const onConnected = () => bind(window.ssMqttClient);
    window.addEventListener("ss-mqtt-connected", onConnected);
    bind(window.ssMqttClient);

    return () => {
      cancelled = true;
      window.removeEventListener("ss-mqtt-connected", onConnected);
      if (recovery != null) {
        window.clearInterval(recovery);
      }
      if (throttleTimer != null) {
        window.clearTimeout(throttleTimer);
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
