// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from "vitest";
import { createTrackSmoother, readSmoothTracksPreference, TRACK_SMOOTHING_ALPHA, writeSmoothTracksPreference } from "./trackSmoothing";

afterEach(() => vi.unstubAllGlobals());

describe("Smooth preference persistence", () => {
  it("restores the selected value after a fresh viewport store load", async () => {
    const storage = new Map<string, string>([["ss-viewer-smooth-tracks", "1"]]);
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => storage.set(key, value),
    });
    vi.resetModules();
    const first = await import("./store");
    expect(first.useViewportStore.getState().smoothTracks).toBe(true);
    first.useViewportStore.getState().toggleOverlay("smoothTracks");
    expect(storage.get("ss-viewer-smooth-tracks")).toBe("0");
    vi.resetModules();
    const reloaded = await import("./store");
    expect(reloaded.useViewportStore.getState().smoothTracks).toBe(false);
  });

  it("restores both enabled and disabled choices from persistent browser storage", () => {
    const storage = new Map<string, string>();
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => storage.set(key, value),
    });
    expect(readSmoothTracksPreference()).toBe(false);
    writeSmoothTracksPreference(true);
    expect(readSmoothTracksPreference()).toBe(true);
    writeSmoothTracksPreference(false);
    expect(readSmoothTracksPreference()).toBe(false);
  });

  it("keeps controls usable when browser storage is blocked", () => {
    vi.stubGlobal("localStorage", {
      getItem: () => { throw new Error("Blocked"); },
      setItem: () => { throw new Error("Blocked"); },
    });
    expect(readSmoothTracksPreference()).toBe(false);
    expect(() => writeSmoothTracksPreference(true)).not.toThrow();
  });
});

describe("viewer-only track EMA", () => {
  it("initializes at the raw position, then uses a seven-update EMA without changing samples", () => {
    const smoother = createTrackSmoother();
    expect(TRACK_SMOOTHING_ALPHA).toBeCloseTo(1 / 4);
    expect(smoother.update("a", [0, 3, 6])).toEqual([0, 3, 6]);
    const sample = [9, 12, 15] as const;
    expect(smoother.update("a", sample)).toEqual([2.25, 5.25, 8.25]);
    expect(sample).toEqual([9, 12, 15]);
    expect(smoother.update("b", [30, 40, 50])).toEqual([30, 40, 50]);
  });

  it("advances on received samples, not repeated renders or unrelated store updates", () => {
    const smoother = createTrackSmoother();
    smoother.update("a", [0, 0, 0]);
    const sample = [9, 0, 0] as const;
    expect(smoother.update("a", sample)[0]).toBe(2.25);
    for (let frame = 0; frame < 100; frame++) expect(smoother.update("a", sample)[0]).toBe(2.25);
    expect(smoother.update("a", [9, 0, 0])[0]).toBe(3.9375);
  });

  it("forgets removed tracks and clears all state when disabled or changing scene", () => {
    const smoother = createTrackSmoother();
    smoother.update("a", [0, 0, 0]);
    smoother.update("b", [0, 0, 0]);
    smoother.retain(new Set(["b"]));
    expect(smoother.update("a", [9, 0, 0])[0]).toBe(9);
    expect(smoother.update("b", [9, 0, 0])[0]).toBe(2.25);
    smoother.clear();
    expect(smoother.update("b", [9, 0, 0])[0]).toBe(9);
  });
});