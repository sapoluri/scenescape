// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { create } from "zustand";
import type { CalibrateStep, CameraOptics } from "./types";
import { zipCalibratePairs } from "./types";

export type CalibrateState = {
  active: boolean;
  /** Viewport / bootstrap camera entity id (DB pk string). */
  cameraId: string | null;
  sensorId: string | null;
  cameraName: string;
  step: CalibrateStep;
  /** Camera-image points (px). Paired with mapPoints by index. */
  camPoints: [number, number][];
  /** Floor / scene points (meters, Z-up). Paired with camPoints by index. */
  mapPoints: [number, number, number][];
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
  /** True while refine/solve is in flight (non-blocking wizard cue). */
  updatingPose: boolean;
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
  setUpdatingPose: (updating: boolean) => void;
  addCamPoint: (pt: [number, number]) => void;
  addMapPoint: (pt: [number, number, number]) => void;
  moveCamPoint: (index: number, pt: [number, number]) => void;
  moveMapPoint: (index: number, pt: [number, number, number]) => void;
  removeCamPointAt: (index: number) => void;
  removeMapPointAt: (index: number) => void;
  undoPoint: () => void;
  resetPoints: () => void;
  /**
   * Replace both sides (e.g. AprilTag result or loaded stored transforms).
   * Pass dirty=false when hydrating saved correspondences on enter.
   */
  replacePoints: (
    camPoints: [number, number][],
    mapPoints: [number, number, number][],
    dirty?: boolean,
  ) => void;
  markClean: () => void;
};

function stepForCounts(camN: number, mapN: number): CalibrateStep {
  return camN >= 4 && mapN >= 4 && camN === mapN ? "save" : "pick";
}

function afterPoints(
  camPoints: [number, number][],
  mapPoints: [number, number, number][],
  dirty: boolean,
) {
  return {
    camPoints,
    mapPoints,
    dirty,
    step: stepForCounts(camPoints.length, mapPoints.length),
  };
}

const idle = {
  active: false,
  cameraId: null as string | null,
  sensorId: null as string | null,
  cameraName: "",
  step: "select" as CalibrateStep,
  camPoints: [] as [number, number][],
  mapPoints: [] as [number, number, number][],
  dirty: false,
  optics: null as CameraOptics | null,
  feedImageUrl: null as string | null,
  previewPose: null as CalibrateState["previewPose"],
  priorPose: null as CalibrateState["priorPose"],
  updatingPose: false,
};

/**
 * In-viewport calibration mode (Phase 2.3). Separate from the geometry
 * viewport store so Live editing state stays untouched.
 *
 * Interaction matches the legacy 2D/3D calibrator: double-click to add a
 * point on either view, right-click to delete, click-drag to move. Points
 * correspond by index (same color/number on each side).
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
    }),
  exit: () => set({ ...idle }),
  setStep: (step) => set({ step }),
  setOptics: (optics) => set({ optics }),
  setFeedImageUrl: (feedImageUrl) => set({ feedImageUrl }),
  setPreviewPose: (previewPose) => set({ previewPose }),
  setPriorPose: (priorPose) => set({ priorPose }),
  setUpdatingPose: (updatingPose) => set({ updatingPose }),
  addCamPoint: (pt) =>
    set((s) =>
      afterPoints([...s.camPoints, pt], s.mapPoints, true),
    ),
  addMapPoint: (pt) =>
    set((s) =>
      afterPoints(s.camPoints, [...s.mapPoints, pt], true),
    ),
  moveCamPoint: (index, pt) =>
    set((s) => {
      if (index < 0 || index >= s.camPoints.length) {
        return s;
      }
      const camPoints = s.camPoints.slice();
      camPoints[index] = pt;
      return afterPoints(camPoints, s.mapPoints, true);
    }),
  moveMapPoint: (index, pt) =>
    set((s) => {
      if (index < 0 || index >= s.mapPoints.length) {
        return s;
      }
      const mapPoints = s.mapPoints.slice();
      mapPoints[index] = pt;
      return afterPoints(s.camPoints, mapPoints, true);
    }),
  removeCamPointAt: (index) =>
    set((s) => {
      if (index < 0 || index >= s.camPoints.length) {
        return s;
      }
      return afterPoints(
        s.camPoints.filter((_, i) => i !== index),
        s.mapPoints,
        true,
      );
    }),
  removeMapPointAt: (index) =>
    set((s) => {
      if (index < 0 || index >= s.mapPoints.length) {
        return s;
      }
      return afterPoints(
        s.camPoints,
        s.mapPoints.filter((_, i) => i !== index),
        true,
      );
    }),
  undoPoint: () =>
    set((s) => {
      if (s.camPoints.length === 0 && s.mapPoints.length === 0) {
        return s;
      }
      if (s.camPoints.length > s.mapPoints.length) {
        return afterPoints(s.camPoints.slice(0, -1), s.mapPoints, true);
      }
      if (s.mapPoints.length > s.camPoints.length) {
        return afterPoints(s.camPoints, s.mapPoints.slice(0, -1), true);
      }
      // Equal counts: drop the last index from both sides.
      return afterPoints(
        s.camPoints.slice(0, -1),
        s.mapPoints.slice(0, -1),
        s.camPoints.length > 1 || s.dirty,
      );
    }),
  resetPoints: () =>
    set({
      camPoints: [],
      mapPoints: [],
      dirty: true,
      step: "pick",
    }),
  replacePoints: (camPoints, mapPoints, dirty = true) =>
    set(afterPoints(camPoints, mapPoints, dirty)),
  markClean: () => set({ dirty: false }),
}));

export function getCalibrateState(): CalibrateState {
  return useCalibrateStore.getState();
}

/** Completed equal-index correspondences (for solve / save). */
export function getCalibratePairs(): ReturnType<typeof zipCalibratePairs> {
  const s = useCalibrateStore.getState();
  if (s.camPoints.length !== s.mapPoints.length) {
    return [];
  }
  return zipCalibratePairs(s.camPoints, s.mapPoints);
}

declare global {
  interface Window {
    /** Debug/test handle for in-browser calibrate automation. */
    __ssCalibrate?: {
      getState: typeof getCalibrateState;
      getPairs: typeof getCalibratePairs;
      store: typeof useCalibrateStore;
    };
  }
}

if (typeof window !== "undefined") {
  window.__ssCalibrate = {
    getState: getCalibrateState,
    getPairs: getCalibratePairs,
    store: useCalibrateStore,
  };
}
