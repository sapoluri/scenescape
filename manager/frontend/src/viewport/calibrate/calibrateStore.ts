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
  /**
   * Which side still needs a click to finish the in-progress pair.
   * `null` = either side may start the next correspondence.
   */
  pending: CalibratePending;
  draftCam: [number, number] | null;
  draftMap: [number, number, number] | null;
  dirty: boolean;
  optics: CameraOptics | null;
  /** Latest calibration JPEG object URL (for map coverage projection). */
  feedImageUrl: string | null;
  /** Last live-solved REST-style pose (for projection camera). */
  previewPose: {
    position: [number, number, number];
    rotation: [number, number, number];
    fov: number;
  } | null;
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
  setFeedImageUrl: (url: string | null) => void;
  setPreviewPose: (
    pose: {
      position: [number, number, number];
      rotation: [number, number, number];
      fov: number;
    } | null,
  ) => void;
  setPriorPose: (
    pose: {
      position: [number, number, number];
      rotation: [number, number, number];
      fov: number;
    } | null,
  ) => void;
  /** Place / complete from the camera feed (either start or finish a pair). */
  pickCam: (pt: [number, number]) => void;
  /** Place / complete from the 3D map (either start or finish a pair). */
  pickMap: (pt: [number, number, number]) => void;
  clearDraft: () => void;
  /** Remove a correspondence by index (right-click). */
  removePairAt: (index: number) => void;
  undoPair: () => void;
  resetPairs: () => void;
  markClean: () => void;
};

function afterPairs(pairs: CalibratePair[], dirty: boolean) {
  return {
    pairs,
    draftCam: null as [number, number] | null,
    draftMap: null as [number, number, number] | null,
    pending: null as CalibratePending,
    dirty,
    step: (pairs.length >= 4 ? "save" : "pick") as CalibrateStep,
  };
}

const idle = {
  active: false,
  cameraId: null as string | null,
  sensorId: null as string | null,
  cameraName: "",
  step: "select" as CalibrateStep,
  pairs: [] as CalibratePair[],
  pending: null as CalibratePending,
  draftCam: null as [number, number] | null,
  draftMap: null as [number, number, number] | null,
  dirty: false,
  optics: null as CameraOptics | null,
  feedImageUrl: null as string | null,
  previewPose: null as CalibrateState["previewPose"],
  priorPose: null as CalibrateState["priorPose"],
};

/**
 * In-viewport calibration mode (Phase 2.3). Separate from the geometry
 * viewport store so Live editing state stays untouched.
 *
 * Correspondences are bidirectional: click camera then map, or map then
 * camera. Colors are assigned by pair index (see `colors.ts`).
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
      pending: null,
    }),
  exit: () => set({ ...idle }),
  setStep: (step) => set({ step }),
  setOptics: (optics) => set({ optics }),
  setFeedImageUrl: (feedImageUrl) => set({ feedImageUrl }),
  setPreviewPose: (previewPose) => set({ previewPose }),
  setPriorPose: (priorPose) => set({ priorPose }),
  pickCam: (pt) =>
    set((s) => {
      if (s.draftMap) {
        return afterPairs(
          [...s.pairs, { cam: pt, map: s.draftMap }],
          true,
        );
      }
      // Start (or replace) a camera-side draft; wait for the map.
      return {
        draftCam: pt,
        draftMap: null,
        pending: "scene" as const,
        dirty: true,
      };
    }),
  pickMap: (pt) =>
    set((s) => {
      if (s.draftCam) {
        return afterPairs(
          [...s.pairs, { cam: s.draftCam, map: pt }],
          true,
        );
      }
      // Start (or replace) a map-side draft; wait for the camera feed.
      return {
        draftMap: pt,
        draftCam: null,
        pending: "cam" as const,
        dirty: true,
      };
    }),
  clearDraft: () =>
    set({
      draftCam: null,
      draftMap: null,
      pending: null,
    }),
  removePairAt: (index) =>
    set((s) => {
      if (index < 0 || index >= s.pairs.length) {
        return s;
      }
      return afterPairs(
        s.pairs.filter((_, i) => i !== index),
        true,
      );
    }),
  undoPair: () =>
    set((s) => {
      if (s.draftCam || s.draftMap) {
        return {
          draftCam: null,
          draftMap: null,
          pending: null,
        };
      }
      const pairs = s.pairs.slice(0, -1);
      return afterPairs(pairs, pairs.length > 0 || s.dirty);
    }),
  resetPairs: () =>
    set({
      pairs: [],
      draftCam: null,
      draftMap: null,
      pending: null,
      dirty: true,
      step: "pick",
    }),
  markClean: () => set({ dirty: false }),
}));

export function getCalibrateState(): CalibrateState {
  return useCalibrateStore.getState();
}
