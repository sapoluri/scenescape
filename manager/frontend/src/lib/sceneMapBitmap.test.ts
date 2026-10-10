// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { sceneMapBitmapUrl } from "./sceneMapBitmap";

describe("sceneMapBitmapUrl", () => {
  it("prefers thumbnail over map", () => {
    expect(
      sceneMapBitmapUrl({
        thumbnailUrl: "/media/a_2d.png",
        mapUrl: "/media/a.png",
      }),
    ).toBe("/media/a_2d.png");
  });

  it("skips non-bitmap map files", () => {
    expect(
      sceneMapBitmapUrl({
        thumbnailUrl: null,
        mapUrl: "/media/warehouse.glb",
      }),
    ).toBeNull();
    expect(
      sceneMapBitmapUrl({
        thumbnail: null,
        map: "/media/site.ply?v=1",
      }),
    ).toBeNull();
  });

  it("falls back to image map files", () => {
    expect(
      sceneMapBitmapUrl({
        thumbnailUrl: null,
        mapUrl: "/media/floor.png",
      }),
    ).toBe("/media/floor.png");
  });
});
