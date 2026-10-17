// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getViewportState } from "../store";
import { useVisualizeStore } from "../visualize/visualizeStore";
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
  // Visualize and calibrate share the split chrome — never both.
  useVisualizeStore.getState().exit();
  // Live tool: double-click adds points; empty left-drag still orbits.
  getViewportState().setTool("live");
  useCalibrateStore.getState().enter(args);
}

export {
  getCalibratePairs,
  getCalibrateState,
  useCalibrateStore,
} from "./calibrateStore";
export {
  pairsToTransforms,
  TRANSFORM_TYPE_POINT,
  transformsToPairs,
  zipCalibratePairs,
} from "./types";
export type { CalibratePair, CameraOptics } from "./types";
