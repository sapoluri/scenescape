// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { runInNewContext } from "node:vm";
import { afterEach, describe, expect, it, vi } from "vitest";
import legacySource from "../../../backend/manager/static/js/sscape.js?raw";
import { mqttPayloadToString, setMqttConnected } from "./client";

describe("legacy scene MQTT event routing", () => {
  function attach(reactMap: boolean) {
    const start = legacySource.indexOf("window.ssAttachSceneMqttClient = function");
    const end = legacySource.indexOf("async function checkBrokerConnections", start);
    expect(start).toBeGreaterThanOrEqual(0);
    expect(end).toBeGreaterThan(start);
    const handlers = new Map<string, (topic: string, data: string) => void>();
    const client = {
      connected: false,
      on: (event: string, handler: (topic: string, data: string) => void) => {
        handlers.set(event, handler);
      },
    };
    const window = {
      ssUseReactMap: reactMap,
      ssReactOwnsMqtt: true,
      ssAttachSceneMqttClient: undefined as ((client: unknown) => void) | undefined,
    };
    const schedulePlot = vi.fn();
    const drawSensor = vi.fn();
    const drawRoi = vi.fn();
    const newTripwire = vi.fn();
    runInNewContext(legacySource.slice(start, end), {
      window,
      refreshSceneId: () => {},
      $: () => ({ val: () => "", on: () => {} }),
      APP_NAME: "scenescape",
      CMD_CAMERA: "/cmd/camera/",
      DATA_REGULATED: "/regulated/scene/",
      show_telemetry: false,
      mqttPayloadToJson: JSON.parse,
      schedulePlot,
      drawSensor,
      drawRoi,
      newTripwire,
      setROIColor: vi.fn(),
    });
    window.ssAttachSceneMqttClient!(client);
    return { onMessage: handlers.get("message")!, schedulePlot, drawSensor, drawRoi, newTripwire };
  }

  it.each([
    ["region", { fromSensor: true, title: "Sensor 1" }],
    ["region", { title: "Region 1", uuid: "region-1" }],
    ["tripwire", { title: "Tripwire 1", uuid: "tripwire-1" }],
  ])("keeps tracks flowing after a React %s event", (kind, metadata) => {
    const bridge = attach(true);
    bridge.onMessage(`scenescape/event/${kind}/scene/entity/count`, JSON.stringify({ metadata }));
    const objects = [{ id: "person-1", translation: [1, 2] }];
    bridge.onMessage("scenescape/regulated/scene/scene", JSON.stringify({ objects }));
    expect(bridge.drawSensor).not.toHaveBeenCalled();
    expect(bridge.drawRoi).not.toHaveBeenCalled();
    expect(bridge.newTripwire).not.toHaveBeenCalled();
    expect(bridge.schedulePlot).toHaveBeenCalledWith(objects);
  });

  it("retains sensor event drawing for the legacy 2D map", () => {
    const bridge = attach(false);
    const metadata = { fromSensor: true, title: "Sensor 1", uuid: "sensor-1" };
    bridge.onMessage("scenescape/event/region/scene/sensor-1/count", JSON.stringify({ metadata }));
    expect(bridge.drawSensor).toHaveBeenCalledWith(metadata, "Sensor 1", "child_sensor");
  });
});

describe("mqttPayloadToString", () => {
  it("returns empty for nullish", () => {
    expect(mqttPayloadToString(null)).toBe("");
    expect(mqttPayloadToString(undefined)).toBe("");
  });

  it("passes through strings", () => {
    expect(mqttPayloadToString('{"ok":true}')).toBe('{"ok":true}');
  });

  it("decodes Uint8Array as UTF-8", () => {
    const bytes = new TextEncoder().encode('{"id":1}');
    expect(mqttPayloadToString(bytes)).toBe('{"id":1}');
  });

  it("decodes ArrayBuffer as UTF-8", () => {
    const bytes = new TextEncoder().encode("hello");
    expect(mqttPayloadToString(bytes.buffer)).toBe("hello");
  });

  it("fails closed on non-binary objects with toString", () => {
    const bogus = {
      toString(enc?: string) {
        return enc === "utf8" ? "[object Buffer]" : "[object Object]";
      },
    };
    expect(mqttPayloadToString(bogus)).toBe("");
  });

  it("decodes Buffer-like objects via toString(utf8)", () => {
    const payload = '{"ok":true}';
    const bufLike = {
      byteLength: payload.length,
      toString(enc?: string) {
        return enc === "utf8" ? payload : "";
      },
    };
    expect(mqttPayloadToString(bufLike)).toBe(payload);
  });

  it("fails closed on numbers and plain objects", () => {
    expect(mqttPayloadToString(42)).toBe("");
    expect(mqttPayloadToString({ a: 1 })).toBe("");
  });
});

describe("setMqttConnected", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("marks mqtt_status connected and dispatches ss-mqtt-status", () => {
    class FakeHTMLElement {}
    class FakeCustomEvent extends Event {
      detail: unknown;

      constructor(type: string, options: { detail: unknown }) {
        super(type);
        this.detail = options.detail;
      }
    }
    const classes = new Set<string>();
    const attrs: Record<string, string> = {};
    const el = Object.assign(Object.create(FakeHTMLElement.prototype), {
      id: "mqtt_status",
      classList: {
        toggle(name: string, force?: boolean) {
          if (force === true) {
            classes.add(name);
          } else if (force === false) {
            classes.delete(name);
          } else if (classes.has(name)) {
            classes.delete(name);
          } else {
            classes.add(name);
          }
        },
        contains(name: string) {
          return classes.has(name);
        },
        remove(name: string) {
          classes.delete(name);
        },
      },
      setAttribute(k: string, v: string) {
        attrs[k] = v;
      },
      getAttribute(k: string) {
        return attrs[k] ?? null;
      },
    });

    vi.stubGlobal("HTMLElement", FakeHTMLElement);
    vi.stubGlobal("CustomEvent", FakeCustomEvent);
    vi.stubGlobal("document", {
      getElementById: (id: string) => (id === "mqtt_status" ? el : null),
      querySelectorAll: () => [el],
    });

    const seen: boolean[] = [];
    const realWindow = globalThis.window;
    const dispatchEvent = vi.fn((ev: Event) => {
      if (ev.type === "ss-mqtt-status" && "detail" in ev) {
        seen.push(Boolean((ev as CustomEvent).detail?.connected));
      }
      return true;
    });
    vi.stubGlobal("window", {
      ...(realWindow ?? {}),
      dispatchEvent,
    });

    setMqttConnected(true);
    expect(classes.has("connected")).toBe(true);
    expect(attrs["data-ss-mqtt"]).toBe("connected");
    expect(seen).toEqual([true]);

    setMqttConnected(false);
    expect(classes.has("connected")).toBe(false);
    expect(attrs["data-ss-mqtt"]).toBe("disconnected");
    expect(seen).toEqual([true, false]);
  });
});
