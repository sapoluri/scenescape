// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { create } from "zustand";
import type {
  CalibratePair,
  CalibratePending,
  CalibrateStep,
  CameraOptics,
} from "./types";

export type CalibrateState = {
  active: boolean;
  /** Viewport / bootstrap camera entity id (DB pk string). */
  cameraId: string | null;
  sensorId: string | null;
  cameraName: string;
  step: CalibrateStep;
  pairs: CalibratePair[];
  pending: CalibratePending;
  draftCam: [number, number] | null;
  dirty: boolean;
  optics: CameraOptics | null;
  /** Pose snapshot before calibrate (restore on cancel). */
  priorPose: {
    position: [number, number, number];
    rotation: [number, number, number];
    fov: number;
  } | null;
  enter: (args: {
    cameraId: string;
    sensorId: string;
    cameraName: string;
  }) => void;
  exit: () => void;
  setStep: (step: CalibrateStep) => void;
  setOptics: (optics: CameraOptics) => void;
  setPriorPose: (
    pose: {
      position: [number, number, number];
      rotation: [number, number, number];
      fov: number;
    } | null,
  ) => void;
  setDraftCam: (pt: [number, number] | null) => void;
  setPending: (p: CalibratePending) => void;
  addPair: (pair: CalibratePair) => void;
  undoPair: () => void;
  resetPairs: () => void;
  markClean: () => void;
};

const idle = {
  active: false,
  cameraId: null as string | null,
  sensorId: null as string | null,
  cameraName: "",
  step: "select" as CalibrateStep,
  pairs: [] as CalibratePair[],
  pending: "cam" as CalibratePending,
  draftCam: null as [number, number] | null,
  dirty: false,
  optics: null as CameraOptics | null,
  priorPose: null as CalibrateState["priorPose"],
};

/**
 * In-viewport calibration mode (Phase 2.3). Separate from the geometry
 * viewport store so Live editing state stays untouched.
 */
export const useCalibrateStore = create<CalibrateState>()((set) => ({
  ...idle,
  enter: ({ cameraId, sensorId, cameraName }) =>
    set({
      ...idle,
      active: true,
      cameraId,
      sensorId,
      cameraName,
      step: "pick",
      pending: "cam",
    }),
  exit: () => set({ ...idle }),
  setStep: (step) => set({ step }),
  setOptics: (optics) => set({ optics }),
  setPriorPose: (priorPose) => set({ priorPose }),
  setDraftCam: (draftCam) => set({ draftCam }),
  setPending: (pending) => set({ pending }),
  addPair: (pair) =>
    set((s) => ({
      pairs: [...s.pairs, pair],
      draftCam: null,
      pending: "cam",
      dirty: true,
      step: s.pairs.length + 1 >= 4 ? "save" : "pick",
    })),
  undoPair: () =>
    set((s) => {
      const pairs = s.pairs.slice(0, -1);
      return {
        pairs,
        draftCam: null,
        pending: "cam",
        dirty: pairs.length > 0 || s.dirty,
        step: pairs.length >= 4 ? "save" : "pick",
      };
    }),
  resetPairs: () =>
    set({
      pairs: [],
      draftCam: null,
      pending: "cam",
      dirty: true,
      step: "pick",
    }),
  markClean: () => set({ dirty: false }),
}));

export function getCalibrateState(): CalibrateState {
  return useCalibrateStore.getState();
}
