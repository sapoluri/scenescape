// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it } from "vitest";
import { CALIBRATION_HISTORY_LIMIT, getCalibrateState } from "./calibrateStore";

beforeEach(() => getCalibrateState().enter({ cameraId: "1", sensorId: "camera1", cameraName: "Camera 1" }));

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