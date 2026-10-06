// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getCalibrateState, useCalibrateStore } from "./calibrateStore";

/** Enter Phase 2.3 in-viewport calibrate for a scene camera. */
export function enterCameraCalibrate(args: {
  cameraId: string;
  sensorId: string;
  cameraName: string;
}): void {
  const cur = getCalibrateState();
  if (
    cur.active &&
    cur.dirty &&
    cur.cameraId !== args.cameraId &&
    !window.confirm("Switch camera and discard unsaved calibration points?")
  ) {
    return;
  }
  useCalibrateStore.getState().enter(args);
}

export { useCalibrateStore, getCalibrateState } from "./calibrateStore";
export { pairsToTransforms, TRANSFORM_TYPE_POINT } from "./types";
export type { CalibratePair, CameraOptics } from "./types";
