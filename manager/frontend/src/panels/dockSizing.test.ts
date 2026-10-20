// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from "vitest";
import {
  clampDockWidth,
  DEFAULT_DOCK_WIDTH,
  readDockWidth,
  writeDockWidth,
} from "./dockSizing";

afterEach(() => vi.unstubAllGlobals());

describe("side panel sizing", () => {
  it("supports wider previews while retaining viewport space", () => {
    expect(clampDockWidth(480, 1200)).toBe(480);
    expect(clampDockWidth(50, 1200)).toBe(240);
    expect(clampDockWidth(2000, 1200)).toBe(864);
  });
  it("fits narrow and empty containers", () => {
    expect(clampDockWidth(500, 390)).toBe(179);
    expect(clampDockWidth(500, 0)).toBe(0);
    expect(clampDockWidth(NaN, 1200)).toBe(DEFAULT_DOCK_WIDTH);
  });
  it("remembers the requested width and handles unavailable storage", () => {
    let saved: string | null = null;
    vi.stubGlobal("sessionStorage", {
      getItem: () => saved,
      setItem: (_key: string, value: string) => {
        saved = value;
      },
    });
    expect(readDockWidth()).toBe(DEFAULT_DOCK_WIDTH);
    writeDockWidth(480.4);
    expect(readDockWidth()).toBe(480);
    vi.stubGlobal("sessionStorage", {
      getItem: () => {
        throw new Error("Unavailable");
      },
    });
    expect(readDockWidth()).toBe(DEFAULT_DOCK_WIDTH);
  });
});
