// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from "vitest";
import { parseCameraOptics } from "../../lib/cameraOptics";
import { api } from "../../lib/rest";
import { refineIntrinsics } from "./refineIntrinsics";
import type { CalibratePair } from "./types";

vi.mock("../../lib/rest", () => ({ api: { calculateIntrinsics: vi.fn() } }));
const optics = { ...parseCameraOptics({}), fx: 900, fy: 900, k1: 0.12 };
const pairs = Array.from({ length: 8 }, (_, index): CalibratePair => ({
  cam: [index * 10, index * 20], map: [index, index, 0],
}));

beforeEach(() => {
  vi.mocked(api.calculateIntrinsics).mockReset().mockResolvedValue({
    mtx: [[800, 0, 640], [0, 800, 360], [0, 0, 1]],
    dist: [[-0.15, 99, 99, 99, 99]],
  });
});

describe("calibration estimation locks", () => {
  it("does not estimate when both values are locked", async () => {
    expect(await refineIntrinsics("token", pairs, optics)).toBeNull();
    expect(api.calculateIntrinsics).not.toHaveBeenCalled();
  });

  it("estimates one shared focal length at six pairs while preserving distortion", async () => {
    const result = await refineIntrinsics("token", pairs.slice(0, 6), optics, { focal: false, k1: false });
    expect(result).toMatchObject({ fx: 800, fy: 800, k1: 0.12, k2: 0, p1: 0 });
    expect(api.calculateIntrinsics).toHaveBeenCalledWith("token", expect.objectContaining({
      fixIntrinsics: expect.objectContaining({ fx: false, fy: false, k1: true }),
    }));
  });

  it("estimates only k1 at eight pairs when focal length is locked", async () => {
    const result = await refineIntrinsics("token", pairs, optics, { focal: true, k1: false });
    expect(result).toMatchObject({ fx: 900, fy: 900, k1: -0.15, k2: 0, k3: 0, p1: 0, p2: 0 });
    expect(api.calculateIntrinsics).toHaveBeenCalledWith("token", expect.objectContaining({
      fixIntrinsics: expect.objectContaining({ fx: true, fy: true, k1: false, k2: true, p1: true }),
    }));
  });

  it.each([4, 5, 7])("does not estimate below the unlocked parameter threshold at %i pairs", async count => {
    const locks = { focal: count < 6 ? false : true, k1: false };
    expect(await refineIntrinsics("token", pairs.slice(0, count), optics, locks)).toBeNull();
    expect(api.calculateIntrinsics).not.toHaveBeenCalled();
  });
});