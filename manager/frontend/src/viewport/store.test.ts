// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from "vitest";

// The store's history snapshots the geometry model (node env: stub DOM).
vi.stubGlobal("document", { getElementById: () => null });
vi.stubGlobal("window", { dispatchEvent: () => undefined });

import { resetGeometryModel } from "../scene/map/geometryModel";
import { useViewportStore } from "./store";
import type { MarkEntity, RegionEntity } from "./types";

const region = (id: string): RegionEntity => ({
  id,
  type: "region",
  name: `Region ${id}`,
  visible: true,
  points: [
    [0, 0],
    [4, 0],
    [4, 3],
    [0, 3],
  ],
  min: [0, 0],
  max: [4, 3],
  height: 2,
  color: "#30d158",
  volumetric: false,
  bufferSize: 0,
});

describe("useViewportStore", () => {
  beforeEach(() => {
    useViewportStore.getState().clearEntities();
    useViewportStore.setState({
      selectedId: null,
      viewPreset: "persp",
      ortho: false,
      showGrid: true,
      showLabels: true,
      showTrails: true,
      showFrustums: true,
      activeTool: "live",
    });
  });

  it("upserts entities by id", () => {
    const s = useViewportStore.getState();
    s.upsertEntities([region("a"), region("b")]);
    expect(Object.keys(useViewportStore.getState().entities)).toHaveLength(2);
    s.upsertEntities([{ ...region("a"), name: "Renamed" }]);
    expect(useViewportStore.getState().entities["a"].name).toBe("Renamed");
  });

  it("removes one entity and clears selection", () => {
    const s = useViewportStore.getState();
    s.upsertEntities([region("a"), region("b")]);
    s.select("a");
    s.removeEntity("a");
    const after = useViewportStore.getState();
    expect(after.entities["a"]).toBeUndefined();
    expect(after.selectedId).toBeNull();
  });

  it("removes entities by type", () => {
    const s = useViewportStore.getState();
    s.upsertEntities([region("a"), region("b")]);
    s.removeEntitiesByType("region");
    expect(Object.keys(useViewportStore.getState().entities)).toHaveLength(0);
  });

  it("toggles overlays", () => {
    const s = useViewportStore.getState();
    expect(s.showGrid).toBe(true);
    s.toggleOverlay("showGrid");
    expect(useViewportStore.getState().showGrid).toBe(false);
  });

  it("sets the active tool", () => {
    const s = useViewportStore.getState();
    expect(s.activeTool).toBe("live");
    s.setTool("move");
    expect(useViewportStore.getState().activeTool).toBe("move");
  });
});

describe("history (undo/redo)", () => {
  beforeEach(() => {
    resetGeometryModel();
    useViewportStore.getState().clearEntities();
  });

  const mark = (id: string): MarkEntity => ({
    id,
    type: "mark",
    name: id,
    visible: true,
    className: "person",
    color: "#ff0000",
    position: [1, 2, 0],
    heading: 0,
    speed: 0,
  });

  it("undoes and redoes a region creation", () => {
    const s = useViewportStore.getState();
    s.commitHistory();
    s.upsertEntities([region("a")]);
    expect(useViewportStore.getState().entities["a"]).toBeDefined();

    s.undo();
    expect(useViewportStore.getState().entities["a"]).toBeUndefined();

    s.redo();
    expect(useViewportStore.getState().entities["a"]).toBeDefined();
  });

  it("keeps live marks out of undo", () => {
    const s = useViewportStore.getState();
    s.upsertEntities([mark("m1")], { fromMarks: true });
    s.commitHistory();
    s.upsertEntities([region("a")]);

    s.undo();
    const after = useViewportStore.getState();
    expect(after.entities["a"]).toBeUndefined();
    expect(after.entities["m1"]).toBeDefined();

    s.redo();
    const redone = useViewportStore.getState();
    expect(redone.entities["a"]).toBeDefined();
    expect(redone.entities["m1"]).toBeDefined();
  });

  it("is a no-op with empty stacks", () => {
    const s = useViewportStore.getState();
    expect(() => s.undo()).not.toThrow();
    expect(() => s.redo()).not.toThrow();
    expect(Object.keys(useViewportStore.getState().entities)).toHaveLength(0);
  });
});
