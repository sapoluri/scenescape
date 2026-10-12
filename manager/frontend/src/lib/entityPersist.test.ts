// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from "vitest";

vi.stubGlobal("document", { getElementById: () => null });
vi.stubGlobal("window", { dispatchEvent: vi.fn() });

import { remapGeometryEntityIds } from "./entityPersist";
import { getPersistState, usePersistStore } from "../viewport/persistStore";
import { getViewportState, useViewportStore } from "../viewport/store";
import type { RegionEntity } from "../viewport/types";

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
