// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it } from "vitest";
import { CALIBRATION_HISTORY_LIMIT, getCalibrateState } from "./calibrateStore";
import { parseCameraOptics } from "../../lib/cameraOptics";
import { calibrationOpticsForFrame } from "./refineIntrinsics";

beforeEach(() => getCalibrateState().enter({ cameraId: "1", sensorId: "camera1", cameraName: "Camera 1" }));

describe("calibration optics controls", () => {
  it("retains the manual diagonal angle across frame normalization", () => {
    const optics = calibrationOpticsForFrame({ ...parseCameraOptics({}), fy: 400 }, 640, 480);
    getCalibrateState().setOptics(optics);
    getCalibrateState().setManualIntrinsics("dfov");
    const angle = getCalibrateState().manualDiagonalFov!;
    getCalibrateState().setOptics(calibrationOpticsForFrame(optics, 1280, 960, angle));
    expect(getCalibrateState().manualDiagonalFov).toBe(angle);
    expect(getCalibrateState().optics!.fy).toBeCloseTo(800, 10);
    getCalibrateState().setOpticsLock("k1", false);
    expect(getCalibrateState().manualIntrinsics).toBe("dfov");
    expect(getCalibrateState().locks.focal).toBe(true);
  });

  it("switches manual sources without changing the projection or dirtying values", () => {
    const optics = parseCameraOptics({});
    getCalibrateState().setOptics(optics);
    getCalibrateState().setManualIntrinsics("dfov");
    expect(getCalibrateState().manualDiagonalFov).toBeGreaterThan(0);
    expect(getCalibrateState().locks.focal).toBe(true);
    expect(getCalibrateState().optics).toBe(optics);
    getCalibrateState().setManualIntrinsics("neither");
    expect(getCalibrateState().locks.focal).toBe(false);
    expect(getCalibrateState().manualDiagonalFov).toBeNull();
    getCalibrateState().setManualIntrinsics("focal");
    expect(getCalibrateState().locks.focal).toBe(true);
    expect(getCalibrateState().optics).toBe(optics);
    expect(getCalibrateState().dirty).toBe(false);
  });

  it("restores clean state when a manual optics value is restored and resets locks on exit", () => {
    const optics = parseCameraOptics({});
    getCalibrateState().setOptics(optics);
    getCalibrateState().editOptics({ ...optics, k1: 0.1 });
    getCalibrateState().replacePoints([], [], false);
    expect(getCalibrateState().dirty).toBe(true);
    getCalibrateState().editOptics(optics);
    expect(getCalibrateState().dirty).toBe(false);
    getCalibrateState().setOpticsLock("k1", false);
    getCalibrateState().exit();
    expect(getCalibrateState().locks).toEqual({ focal: false, k1: true });
    expect(getCalibrateState().manualIntrinsics).toBe("neither");
  });

  it("defaults to Neither and does not dirty saved camera values when choosing a manual source", () => {
    expect(getCalibrateState().manualIntrinsics).toBe("neither");
    expect(getCalibrateState().locks).toEqual({ focal: false, k1: true });
    getCalibrateState().setManualIntrinsics("focal");
    expect(getCalibrateState().locks.focal).toBe(true);
    expect(getCalibrateState().dirty).toBe(false);
    getCalibrateState().enter({ cameraId: "2", sensorId: "camera2", cameraName: "Camera 2" });
    expect(getCalibrateState().manualIntrinsics).toBe("neither");
  });

  it("keeps manual optics dirty through point undo and clears them on save", () => {
    const optics = parseCameraOptics({});
    getCalibrateState().setOptics(optics);
    getCalibrateState().editOptics({ ...optics, fx: 900, fy: 900 });
    expect(getCalibrateState().opticsRevision).toBe(1);
    getCalibrateState().addCamPoint([10, 20]);
    getCalibrateState().undoPoint();
    expect(getCalibrateState().dirty).toBe(true);
    getCalibrateState().markClean();
    expect(getCalibrateState().dirty).toBe(false);
    expect(getCalibrateState().opticsDirty).toBe(false);
  });
});

describe("calibration point history", () => {
  it("undoes and redoes additions on either side", () => {
    getCalibrateState().addCamPoint([10, 20]);
    getCalibrateState().addMapPoint([1, 2, 0]);
    getCalibrateState().undoPoint();
    expect(getCalibrateState().camPoints).toEqual([[10, 20]]);
    expect(getCalibrateState().mapPoints).toEqual([]);
    getCalibrateState().redoPoint();
    expect(getCalibrateState().mapPoints).toEqual([[1, 2, 0]]);
  });

  it("groups a drag into one edit and restores saved-state dirtiness", () => {
    getCalibrateState().replacePoints([[10, 20]], [[1, 2, 0]], false);
    getCalibrateState().beginPointEdit();
    for (let index = 0; index < 100; index++) getCalibrateState().moveCamPoint(0, [index, 20]);
    getCalibrateState().endPointEdit();
    expect(getCalibrateState().pastPoints).toHaveLength(1);
    getCalibrateState().undoPoint();
    expect(getCalibrateState().camPoints).toEqual([[10, 20]]);
    expect(getCalibrateState().dirty).toBe(false);
    getCalibrateState().redoPoint();
    expect(getCalibrateState().camPoints).toEqual([[99, 20]]);
    expect(getCalibrateState().dirty).toBe(true);
  });

  it("undoes reset and point removal", () => {
    getCalibrateState().replacePoints([[10, 20]], [[1, 2, 0]], false);
    getCalibrateState().removeMapPointAt(0);
    getCalibrateState().undoPoint();
    expect(getCalibrateState().mapPoints).toEqual([[1, 2, 0]]);
    getCalibrateState().resetPoints();
    getCalibrateState().undoPoint();
    expect(getCalibrateState().camPoints).toEqual([[10, 20]]);
    expect(getCalibrateState().mapPoints).toEqual([[1, 2, 0]]);
  });

  it("clears redo for new edits but retains it for an unchanged gesture", () => {
    getCalibrateState().addCamPoint([10, 20]);
    getCalibrateState().undoPoint();
    getCalibrateState().beginPointEdit();
    getCalibrateState().endPointEdit();
    expect(getCalibrateState().futurePoints).toHaveLength(1);
    getCalibrateState().addCamPoint([30, 40]);
    expect(getCalibrateState().futurePoints).toHaveLength(0);
    getCalibrateState().redoPoint();
    expect(getCalibrateState().camPoints).toEqual([[30, 40]]);
  });

  it("handles undo and redo across a save correctly", () => {
    getCalibrateState().addCamPoint([10, 20]);
    getCalibrateState().markClean();
    getCalibrateState().undoPoint();
    expect(getCalibrateState().dirty).toBe(true);
    getCalibrateState().redoPoint();
    expect(getCalibrateState().dirty).toBe(false);
  });

  it("bounds history and clears it on camera change or exit", () => {
    for (let index = 0; index < 100; index++) getCalibrateState().addCamPoint([index, 20]);
    expect(getCalibrateState().pastPoints).toHaveLength(CALIBRATION_HISTORY_LIMIT);
    getCalibrateState().enter({ cameraId: "2", sensorId: "camera2", cameraName: "Camera 2" });
    expect(getCalibrateState().pastPoints).toHaveLength(0);
    expect(getCalibrateState().futurePoints).toHaveLength(0);
    getCalibrateState().addMapPoint([1, 2, 0]);
    getCalibrateState().exit();
    expect(getCalibrateState().pastPoints).toHaveLength(0);
  });
});