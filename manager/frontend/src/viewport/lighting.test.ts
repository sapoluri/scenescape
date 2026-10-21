// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { attachLightingMqtt, lightIntensityFromSensor, useLightingStore } from "./lighting";
import type { SceneSensorBootstrap } from "../scene/types";

function sensor(area = "scene", sensorId = "room_light"): SceneSensorBootstrap {
  return {
    id: "1", sensorId, name: "Room light", areaJson: JSON.stringify({ area }),
    iconUrl: null, calibrateHref: "", editHref: "", deleteUrl: null,
  };
}

beforeEach(() => useLightingStore.setState({ intensity: 1 }));
afterEach(() => vi.unstubAllGlobals());

describe("released lighting behavior", () => {
  it.each([[0, 0.1], [250, 0.5], [500, 1], [1500, 3], [5000, 3]])("maps %s lux to intensity %s", (lux, intensity) => {
    expect(lightIntensityFromSensor("scenescape/data/sensor/room_light", { value: lux }, [sensor()])).toBe(intensity);
  });

  it("accepts a light subtype and numeric string values", () => {
    expect(lightIntensityFromSensor("scenescape/data/sensor/luxmeter", {
      subtype: "light", value: "750",
    }, [sensor("scene", "luxmeter")])).toBe(1.5);
  });

  it.each(["circle", "poly", "unknown"])("ignores %s-area sensors", area => {
    expect(lightIntensityFromSensor("scenescape/data/sensor/room_light", { value: 500 }, [sensor(area)])).toBeNull();
  });

  it("ignores other scenes, non-light readings, malformed configuration, and invalid values", () => {
    expect(lightIntensityFromSensor("scenescape/data/sensor/unknown", { value: 500 }, [sensor()])).toBeNull();
    expect(lightIntensityFromSensor("scenescape/data/sensor/temperature", { value: 20 }, [sensor("scene", "temperature")])).toBeNull();
    for (const value of [null, undefined, "invalid", Infinity, NaN]) {
      expect(lightIntensityFromSensor("scenescape/data/sensor/room_light", { value }, [sensor()])).toBeNull();
    }
    expect(lightIntensityFromSensor("scenescape/data/sensor/room_light", { value: 500 }, [{ ...sensor(), areaJson: "invalid" }])).toBeNull();
  });

  it("shares clamped values between slider state and MQTT updates", () => {
    useLightingStore.getState().setIntensity(10);
    expect(useLightingStore.getState().intensity).toBe(3);
    useLightingStore.getState().setIntensity(-1);
    expect(useLightingStore.getState().intensity).toBe(0.1);
    useLightingStore.getState().setIntensity(NaN);
    expect(useLightingStore.getState().intensity).toBe(0.1);
  });

  it("subscribes on reconnect, updates the slider state, and removes listeners on cleanup", () => {
    const handlers = new Map<string, (...args: unknown[]) => void>();
    const client = {
      connected: true,
      subscribe: vi.fn(), publish: vi.fn(),
      on: (event: string, callback: (...args: unknown[]) => void) => { handlers.set(event, callback); },
      removeListener: vi.fn((event: string) => { handlers.delete(event); }),
    };
    vi.stubGlobal("window", Object.assign(new EventTarget(), { ssMqttClient: client }));
    const stop = attachLightingMqtt([sensor()]);
    expect(client.subscribe).toHaveBeenCalledWith("scenescape/data/sensor/+");
    handlers.get("message")!("scenescape/data/sensor/room_light", JSON.stringify({ value: 1000 }));
    expect(useLightingStore.getState().intensity).toBe(2);
    handlers.get("connect")!();
    expect(client.subscribe).toHaveBeenCalledTimes(2);
    stop();
    expect(handlers.size).toBe(0);
  });
});