// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from "vitest";

vi.stubGlobal("document", { getElementById: () => null });
vi.stubGlobal("window", { dispatchEvent: vi.fn() });

import { persistCameraEntity, persistSensorEntity, remapGeometryEntityIds } from "./entityPersist";
import { api } from "./rest";
import { getPersistState, usePersistStore } from "../viewport/persistStore";
import { getViewportState, useViewportStore } from "../viewport/store";
import type { CameraEntity, RegionEntity, SensorEntity } from "../viewport/types";

function region(id: string, name = "Region"): RegionEntity {
  return {
    id,
    type: "region",
    name,
    visible: true,
    points: [
      [0, 0],
      [1, 0],
      [1, 1],
    ],
    min: [0, 0],
    max: [1, 1],
    height: 1,
    color: "#30d158",
    volumetric: false,
    bufferSize: 0,
  };
}

beforeEach(() => {
  useViewportStore.setState({
    entities: {},
    selectedId: null,
    past: [],
    future: [],
    structVersion: 0,
  });
  usePersistStore.setState({ baselines: {} });
});

describe("remapGeometryEntityIds", () => {
  it("replaces draft ids so the outliner does not keep a duplicate", () => {
    const draft = "draft-abc";
    const uid = "11111111-2222-4333-8444-555555555555";
    getViewportState().upsertEntities([region(draft, "Dock")]);
    getViewportState().select(draft);
    getPersistState().captureBaselines([draft]);

    // Simulate geometry reseed after server create (new uid already present).
    getViewportState().upsertEntities([region(uid, "Dock")]);

    remapGeometryEntityIds({ [draft]: uid });

    const after = getViewportState();
    expect(after.entities[draft]).toBeUndefined();
    expect(after.entities[uid]?.name).toBe("Dock");
    expect(after.selectedId).toBe(uid);
    expect(getPersistState().baselines[draft]).toBeUndefined();
    expect(getPersistState().baselines[uid]).toBeDefined();
  });

  it("no-ops when old and new ids match", () => {
    const uid = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee";
    getViewportState().upsertEntities([region(uid)]);
    remapGeometryEntityIds({ [uid]: uid });
    expect(getViewportState().entities[uid]).toBeDefined();
  });
});

describe("camera scene persistence", () => {
  it("saves an existing camera to the selected scene without changing its sensor id", async () => {
    const camera: CameraEntity = {
      id: "42", type: "camera", name: "Camera", visible: true,
      position: [1, 2, 3], target: [0, 0, 0], rotation: [0, 0, 0],
      restRotation: [0, 0, 0], posed: true, fov: 60, color: "#0a84ff",
    };
    const update = vi.spyOn(api, "updateCamera").mockResolvedValueOnce({});
    try {
      await persistCameraEntity("test-token", "destination-scene", "camera-id", camera);
      expect(update).toHaveBeenCalledWith("test-token", "camera-id", expect.objectContaining({
        scene: "destination-scene", sensor_id: "camera-id", name: "Camera", translation: [1, 2, 3],
      }));
    } finally {
      update.mockRestore();
    }
  });
});

describe("sensor coverage persistence", () => {
  function sensor(): SensorEntity {
    return {
      id: "sensor", type: "sensor", name: "Infrared", visible: true,
      position: [15, 20, 3], areaMode: "poly", radius: 7.5,
      points: [[0, 0], [4, 0], [4, 3]],
      singletonType: "environmental", color: "#bf5af2",
    };
  }

  it("saves the sensor position independently from polygon coverage", async () => {
    const update = vi.spyOn(api, "updateSensor").mockResolvedValueOnce({});
    const entity = sensor();
    try {
      await persistSensorEntity("test-token", "scene", "infrared", entity);
      expect(update).toHaveBeenCalledWith("test-token", "infrared", expect.objectContaining({
        center: [15, 20],
        translation: [15, 20, 3],
        color: "#bf5af2",
        points: entity.points,
        area: "poly",
      }));
    } finally {
      update.mockRestore();
    }
  });

  it("saves circular radius and sensor center together", async () => {
    const update = vi.spyOn(api, "updateSensor").mockResolvedValueOnce({});
    try {
      await persistSensorEntity("test-token", "scene", "infrared", { ...sensor(), areaMode: "circle" });
      expect(update).toHaveBeenCalledWith("test-token", "infrared", expect.objectContaining({
        center: [15, 20], translation: [15, 20, 3], color: "#bf5af2", radius: 7.5, area: "circle",
      }));
    } finally {
      update.mockRestore();
    }
  });

  it("rejects incomplete polygon coverage without sending an update", async () => {
    const update = vi.spyOn(api, "updateSensor");
    try {
      await expect(persistSensorEntity("test-token", "scene", "infrared", {
        ...sensor(), points: [[0, 0], [1, 1]],
      })).rejects.toThrow("at least 3 vertices");
      expect(update).not.toHaveBeenCalled();
    } finally {
      update.mockRestore();
    }
  });
});
