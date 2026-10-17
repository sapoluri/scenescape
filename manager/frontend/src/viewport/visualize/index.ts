// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getViewportState } from "../store";
import { useVisualizeStore } from "./visualizeStore";

/** Enter the annotated live-camera visualization split. */
export function enterCameraVisualize(args: {
  cameraId: string;
  sensorId: string;
  cameraName: string;
}): void {
  getViewportState().setTool("live");
  useVisualizeStore.getState().enter(args);
  getViewportState().select(args.cameraId);
}

export { getVisualizeState, useVisualizeStore } from "./visualizeStore";
