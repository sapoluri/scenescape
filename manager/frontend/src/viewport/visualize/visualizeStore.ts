// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { create } from "zustand";
import { useCalibrateStore } from "../calibrate/calibrateStore";

export type VisualizeState = {
  active: boolean;
  /** Viewport / bootstrap camera entity id. */
  cameraId: string | null;
  sensorId: string | null;
  cameraName: string;
  /** Latest annotated live JPEG object URL (optional map projection). */
  feedImageUrl: string | null;
  enter: (args: {
    cameraId: string;
    sensorId: string;
    cameraName: string;
  }) => void;
  exit: () => void;
  setFeedImageUrl: (url: string | null) => void;
};

const idle = {
  active: false,
  cameraId: null as string | null,
  sensorId: null as string | null,
  cameraName: "",
  feedImageUrl: null as string | null,
};

/**
 * In-viewport live visualization split: annotated getimage feed beside the
 * 3D map. Mutually exclusive with calibrate mode.
 */
export const useVisualizeStore = create<VisualizeState>()((set) => ({
  ...idle,
  enter: ({ cameraId, sensorId, cameraName }) => {
    const cal = useCalibrateStore.getState();
    if (cal.active) {
      cal.exit();
    }
    set({
      ...idle,
      active: true,
      cameraId,
      sensorId,
      cameraName,
    });
  },
  exit: () => set({ ...idle }),
  setFeedImageUrl: (feedImageUrl) => set({ feedImageUrl }),
}));

export function getVisualizeState(): VisualizeState {
  return useVisualizeStore.getState();
}
