// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it } from "vitest";
import { useViewportStore } from "./store";
import type { RegionEntity } from "./types";

const region = (id: string): RegionEntity => ({
  id,
  type: "region",
  name: `Region ${id}`,
  visible: true,
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
});
