// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from "vitest";
import { Group, Mesh, MeshBasicMaterial, PerspectiveCamera } from "three";
import { attachMarkRenderer, createMarkLabelElement, createTrailRibbon, updateMarkDwellLabel } from "./renderMarks";
import { useViewportStore } from "./store";
import type { ViewportWorld } from "./world";
import {
  buildMarkEntities,
  colorFromTrackId,
  deriveHeading,
  markKeysLeftScene,
  markLabel,
  parseMarkPosition,
  parseRegionDwells,
  regionDwellRows,
  trackIdFromMarkKey,
} from "./marks";
import type { MarkEntity, RegionEntity, SensorEntity } from "./types";

describe("region dwell labels", () => {
  class Element {
    className = "";
    textContent = "";
    hidden = false;
    dataset: Record<string, string> = {};
    style: Record<string, string> = {};
    children: Element[] = [];
    append(...children: Element[]) { this.children.push(...children); }
    appendChild(child: Element) { this.children.push(child); return child; }
    getContext() { return null; }
    remove() { return; }
    replaceChildren(...children: Element[]) { this.children = children; }
    querySelector(selector: string): Element | null {
      for (const child of this.children) {
        if (child.className.split(" ").includes(selector.slice(1))) return child;
        const nested = child.querySelector(selector);
        if (nested) return nested;
      }
      return null;
    }
  }

  afterEach(() => vi.unstubAllGlobals());

  it("smooths rendered positions and trails without changing raw track state", () => {
    vi.stubGlobal("document", { createElement: () => new Element() });
    vi.stubGlobal("window", { addEventListener: vi.fn(), removeEventListener: vi.fn() });
    const initial = useViewportStore.getState();
    useViewportStore.setState({ entities: {}, smoothTracks: true, showTrails: true });
    const markRoot = new Group();
    const world = {
      markRoot, labelLayer: new Element(), host: { clientWidth: 100, clientHeight: 100 },
      onFrame: new Set<() => void>(), getActiveCamera: () => new PerspectiveCamera(),
    } as unknown as ViewportWorld;
    const cleanup = attachMarkRenderer(world);
    const sample = (x: number): MarkEntity => ({
      id: "mark:abcdef123", type: "mark", name: "person #abcdef", visible: true,
      className: "person", color: "#abcdef", position: [x, 0, 0], heading: 0, speed: 0,
    });
    try {
      useViewportStore.getState().upsertEntities([sample(0)], { fromMarks: true });
      useViewportStore.getState().upsertEntities([sample(9)], { fromMarks: true });
      expect(markRoot.children[0].position.x).toBe(2.25);
      const raw = useViewportStore.getState().entities["mark:abcdef123"] as MarkEntity;
      expect(raw.position).toEqual([9, 0, 0]);
      useViewportStore.setState({ showGrid: false });
      expect(markRoot.children[0].position.x).toBe(2.25);
      const trail = markRoot.children[0].getObjectByName("trail") as Mesh;
      expect(trail.geometry.attributes.position.getX(0)).toBeCloseTo(0);
      useViewportStore.getState().upsertEntities([sample(9)], { fromMarks: true });
      expect(markRoot.children[0].position.x).toBe(3.9375);
      useViewportStore.setState({ smoothTracks: false });
      expect(markRoot.children[0].position.x).toBe(9);
      useViewportStore.setState({ smoothTracks: true });
      expect(markRoot.children[0].position.x).toBe(9);
      useViewportStore.getState().upsertEntities([], { fromMarks: true, removeIds: [raw.id] });
      expect(markRoot.children).toHaveLength(0);
      useViewportStore.getState().upsertEntities([sample(30)], { fromMarks: true });
      expect(markRoot.children[0].position.x).toBe(30);
    } finally {
      cleanup();
      useViewportStore.setState(initial);
    }
  });

  it("lists named regions independently, updates seconds, and removes stale entries", () => {
    vi.stubGlobal("document", { createElement: () => new Element() });
    const label = createMarkLabelElement("person #one");
    const dwell = label.querySelector<HTMLDivElement>(".ss-vlabel--mark-dwell")!;
    const list = label.querySelector(".ss-vlabel--mark-dwell-list")!;
    expect(dwell.hidden).toBe(true);
    expect(dwell.children[0].textContent).toBe("Dwell Time:");
    updateMarkDwellLabel(label, [
      { regionId: "one", name: "Checkout", seconds: 12.34 },
      { regionId: "two", name: "Queue", seconds: 3.5 },
    ]);
    expect(dwell.hidden).toBe(false);
    expect(list.children).toHaveLength(2);
    expect(list.children[0].children[0].textContent).toBe("Checkout");
    expect(list.children[0].children[1].textContent).toBe("12.3 s");
    expect(list.children[1].children[0].textContent).toBe("Queue");
    expect(list.children[1].children[1].textContent).toBe("3.5 s");
    updateMarkDwellLabel(label, [{ regionId: "two", name: "Queue", seconds: 4.7 }]);
    expect(list.children).toHaveLength(1);
    expect(list.children[0].children[1].textContent).toBe("4.7 s");
    updateMarkDwellLabel(label, [{ regionId: "two", name: "Queue", seconds: 4.7 }], false);
    expect(dwell.hidden).toBe(true);
    updateMarkDwellLabel(label, [{ regionId: "two", name: "Queue", seconds: 4.7 }], true);
    expect(dwell.hidden).toBe(false);
    updateMarkDwellLabel(label, []);
    expect(dwell.hidden).toBe(true);
    expect(list.children).toHaveLength(0);
  });
});

describe("trail rendering", () => {
  it("renders beneath object bodies and respects scene depth", () => {
    const trail = createTrailRibbon("#ff0000");
    const material = trail.material as MeshBasicMaterial;
    try {
      expect(material.depthTest).toBe(true);
      expect(material.depthWrite).toBe(false);
      expect(material.transparent).toBe(true);
      expect(trail.renderOrder).toBe(-1);
    } finally {
      trail.geometry.dispose();
      material.dispose();
    }
  });
});

describe("parseMarkPosition", () => {
  it("parses array translations", () => {
    expect(parseMarkPosition([1.5, -2.5])).toEqual([1.5, -2.5]);
    expect(parseMarkPosition([1.5, -2.5, 9])).toEqual([1.5, -2.5]);
  });

  it("parses object translations", () => {
    expect(parseMarkPosition({ x: 3, y: 4 })).toEqual([3, 4]);
  });

  it("rejects missing and non-finite translations", () => {
    expect(parseMarkPosition(undefined)).toBeNull();
    expect(parseMarkPosition(null as never)).toBeNull();
    expect(parseMarkPosition([1])).toBeNull();
    expect(parseMarkPosition([Number.NaN, 2])).toBeNull();
    expect(parseMarkPosition({ x: 1 })).toBeNull();
    expect(parseMarkPosition("1,2" as never)).toBeNull();
  });
});

describe("deriveHeading", () => {
  it("computes heading from movement (0 = +X, CCW)", () => {
    expect(deriveHeading(0, 0, 1, 0, 0)).toBeCloseTo(0);
    expect(deriveHeading(0, 0, 0, 1, 0)).toBeCloseTo(Math.PI / 2);
    expect(deriveHeading(0, 0, -1, 0, 0)).toBeCloseTo(Math.PI);
  });

  it("keeps the previous heading when barely moved", () => {
    expect(deriveHeading(0, 0, 0.01, 0.01, 1.23)).toBeCloseTo(1.23);
  });
});

describe("markLabel", () => {
  it("shows only the id characters used for the track color", () => {
    expect(markLabel("person", "a1b2c3d4-e5f6")).toBe(
      "person #a1b2c3",
    );
    expect(markLabel("person", "a1b2c3d4-e5f6")).toBe(`person ${colorFromTrackId("a1b2c3d4-e5f6")}`);
    expect(markLabel("vehicle", 42)).toBe("vehicle #42");
  });
});

describe("colorFromTrackId", () => {
  it("matches 2D UI: # + first 6 characters of the id", () => {
    expect(colorFromTrackId("a1b2c3d4-e5f6-7890")).toBe("#a1b2c3");
    expect(colorFromTrackId("ff00aa")).toBe("#ff00aa");
  });
});

describe("trackIdFromMarkKey", () => {
  it("strips the mark: prefix", () => {
    expect(trackIdFromMarkKey("mark:a1b2c3d4")).toBe("a1b2c3d4");
    expect(trackIdFromMarkKey("a1b2c3d4")).toBe("a1b2c3d4");
  });
});

describe("markKeysLeftScene", () => {
  it("returns store keys omitted from the current frame", () => {
    expect(
      markKeysLeftScene(
        ["mark:a", "mark:b", "mark:c"],
        [{ id: "b", translation: [1, 2] }],
      ),
    ).toEqual(["mark:a", "mark:c"]);
  });

  it("clears all when the frame is empty", () => {
    expect(markKeysLeftScene(["mark:a", "mark:b"], [])).toEqual([
      "mark:a",
      "mark:b",
    ]);
  });

  it("keeps an id present even when translation is missing", () => {
    expect(
      markKeysLeftScene(["mark:a"], [{ id: "a" }]),
    ).toEqual([]);
  });
});

describe("buildMarkEntities", () => {
  it("keeps independent dwell times for overlapping regions and clears them on exit", () => {
    const seen = new Map();
    const mark = { id: "t1", type: "person", translation: [1, 2, 0] };
    const [inside] = buildMarkEntities([{
      ...mark,
      regions: {
        first: { entered: "2026-10-20T10:00:00Z", dwell: 12.5 },
        second: { entered: "2026-10-20T10:00:10Z", dwell: 2.5 },
      },
    }], seen, {}, 1000);
    expect(inside.regionDwells).toEqual([
      { regionId: "first", seconds: 12.5 },
      { regionId: "second", seconds: 2.5 },
    ]);
    const [outside] = buildMarkEntities([mark], seen, {}, 2000);
    expect(outside.regionDwells).toEqual([]);
  });

  it("ignores exited regions and invalid dwell values but keeps zero seconds", () => {
    expect(parseRegionDwells({
      zero: { entered: "now", dwell: 0 },
      exited: { dwell: 10 },
      negative: { entered: "now", dwell: -1 },
      invalid: { entered: "now", dwell: NaN },
      string: { entered: "now", dwell: "12" },
      empty: null,
    })).toEqual([{ regionId: "zero", seconds: 0 }]);
    expect(parseRegionDwells(undefined)).toEqual([]);
  });

  it("resolves region and sensor-region names without combining their dwell times", () => {
    const [mark] = buildMarkEntities([{
      id: "t1", translation: [1, 2], regions: {
        "region-id": { entered: "now", dwell: 12.5 },
        Sensor_1: { entered: "now", dwell: 8 },
        unknown: { entered: "now", dwell: 1 },
      },
    }], new Map(), {}, 1000);
    expect(regionDwellRows(mark, {
      "region-id": { type: "region", id: "region-id", name: "Checkout" } as RegionEntity,
      "sensor-id": { type: "sensor", id: "sensor-id", name: "Sensor 1" } as SensorEntity,
    })).toEqual([
      { regionId: "region-id", name: "Checkout", seconds: 12.5 },
      { regionId: "Sensor_1", name: "Sensor 1", seconds: 8 },
      { regionId: "unknown", name: "unknown", seconds: 1 },
    ]);
  });

  it("upserts positions and records seen state", () => {
    const seen = new Map();
    const entities = buildMarkEntities(
      [{ id: "t1", type: "person", translation: [1, 2, 0] }],
      seen,
      { person: "#ff0000" },
      1000,
    );
    expect(entities).toHaveLength(1);
    expect(entities[0]).toMatchObject({
      id: "mark:t1",
      className: "person",
      color: "#ff0000",
      position: [1, 2, 0],
    });
    expect(seen.get("mark:t1")?.lastSeen).toBe(1000);
  });
});
