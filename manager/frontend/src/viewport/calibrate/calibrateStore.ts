// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { create } from "zustand";
import type { CalibrateStep, CalibrationLocks, CameraOptics, ManualIntrinsics } from "./types";
import { zipCalibratePairs } from "./types";
import { diagonalFovFromOptics, opticsPersistKey } from "../../lib/cameraOptics";

type PointSnapshot = {
  camPoints: [number, number][];
  mapPoints: [number, number, number][];
};

export const CALIBRATION_HISTORY_LIMIT = 50;

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
  pastPoints: PointSnapshot[];
  futurePoints: PointSnapshot[];
  pointEditStart: PointSnapshot | null;
  savedPointsKey: string;
  optics: CameraOptics | null;
  locks: CalibrationLocks;
  manualIntrinsics: ManualIntrinsics;
  manualDiagonalFov: number | null;
  opticsRevision: number;
  opticsDirty: boolean;
  savedOpticsKey: string | null;
  /** Latest calibration JPEG object URL (for map coverage projection). */
  feedImageUrl: string | null;
  feedImageSize: { width: number; height: number } | null;
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
  setOptics: (optics: CameraOptics, changed?: boolean) => void;
  editOptics: (optics: CameraOptics) => void;
  setOpticsLock: (key: keyof CalibrationLocks, locked: boolean) => void;
  setManualIntrinsics: (mode: ManualIntrinsics) => void;
  setFeedImageUrl: (url: string | null) => void;
  setFeedImageSize: (size: { width: number; height: number } | null) => void;
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
  redoPoint: () => void;
  beginPointEdit: () => void;
  endPointEdit: () => void;
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

function pointsKey(points: PointSnapshot): string {
  return JSON.stringify([points.camPoints, points.mapPoints]);
}

function snapshot(points: PointSnapshot): PointSnapshot {
  return {
    camPoints: points.camPoints.map(([x, y]) => [x, y]),
    mapPoints: points.mapPoints.map(([x, y, z]) => [x, y, z]),
  };
}

function editPoints(state: CalibrateState, points: PointSnapshot) {
  if (pointsKey(state) === pointsKey(points)) return state;
  return {
    ...afterPoints(points.camPoints, points.mapPoints, state.opticsDirty || pointsKey(points) !== state.savedPointsKey),
    pastPoints: state.pointEditStart ? state.pastPoints :
      [...state.pastPoints, snapshot(state)].slice(-CALIBRATION_HISTORY_LIMIT),
    futurePoints: state.pointEditStart ? state.futurePoints : [],
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
  pastPoints: [] as PointSnapshot[],
  futurePoints: [] as PointSnapshot[],
  pointEditStart: null as PointSnapshot | null,
  savedPointsKey: JSON.stringify([[], []]),
  optics: null as CameraOptics | null,
  locks: { focal: false, k1: true },
  manualIntrinsics: "neither" as ManualIntrinsics,
  manualDiagonalFov: null as number | null,
  opticsRevision: 0,
  opticsDirty: false,
  savedOpticsKey: null as string | null,
  feedImageUrl: null as string | null,
  feedImageSize: null as CalibrateState["feedImageSize"],
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
  setOptics: (optics, changed = false) => set((s) => {
    const savedOpticsKey = !s.opticsDirty && !changed ? opticsPersistKey(optics) : s.savedOpticsKey;
    const opticsDirty = opticsPersistKey(optics) !== savedOpticsKey;
    return { optics, savedOpticsKey, opticsDirty,
      manualDiagonalFov: s.manualIntrinsics === "dfov" ? s.manualDiagonalFov ?? diagonalFovFromOptics(optics) : null,
      dirty: opticsDirty || pointsKey(s) !== s.savedPointsKey };
  }),
  editOptics: (optics) => set((s) => {
    const opticsDirty = opticsPersistKey(optics) !== s.savedOpticsKey;
    return { optics, opticsDirty, opticsRevision: s.opticsRevision + 1,
      manualDiagonalFov: s.manualIntrinsics === "dfov" ? diagonalFovFromOptics(optics) : null,
      dirty: opticsDirty || pointsKey(s) !== s.savedPointsKey };
  }),
  setOpticsLock: (key, locked) => set((s) => ({
    locks: { ...s.locks, [key]: locked },
    ...(key === "focal" ? { manualIntrinsics: locked ? "focal" as const : "neither" as const,
      manualDiagonalFov: null } : {}),
  })),
  setManualIntrinsics: (manualIntrinsics) => set((s) => ({
    manualIntrinsics,
    manualDiagonalFov: manualIntrinsics === "dfov" && s.optics ? diagonalFovFromOptics(s.optics) : null,
    locks: { ...s.locks, focal: manualIntrinsics !== "neither" },
  })),
  setFeedImageUrl: (feedImageUrl) => set({ feedImageUrl }),
  setFeedImageSize: (feedImageSize) => set({ feedImageSize }),
  setPreviewPose: (previewPose) => set({ previewPose }),
  setPriorPose: (priorPose) => set({ priorPose }),
  setUpdatingPose: (updatingPose) => set({ updatingPose }),
  addCamPoint: (pt) =>
    set((s) =>
      editPoints(s, { camPoints: [...s.camPoints, pt], mapPoints: s.mapPoints }),
    ),
  addMapPoint: (pt) =>
    set((s) =>
      editPoints(s, { camPoints: s.camPoints, mapPoints: [...s.mapPoints, pt] }),
    ),
  moveCamPoint: (index, pt) =>
    set((s) => {
      if (index < 0 || index >= s.camPoints.length) {
        return s;
      }
      const camPoints = s.camPoints.slice();
      camPoints[index] = pt;
      return editPoints(s, { camPoints, mapPoints: s.mapPoints });
    }),
  moveMapPoint: (index, pt) =>
    set((s) => {
      if (index < 0 || index >= s.mapPoints.length) {
        return s;
      }
      const mapPoints = s.mapPoints.slice();
      mapPoints[index] = pt;
      return editPoints(s, { camPoints: s.camPoints, mapPoints });
    }),
  removeCamPointAt: (index) =>
    set((s) => {
      if (index < 0 || index >= s.camPoints.length) {
        return s;
      }
      return editPoints(s, {
        camPoints: s.camPoints.filter((_, i) => i !== index),
        mapPoints: s.mapPoints,
      });
    }),
  removeMapPointAt: (index) =>
    set((s) => {
      if (index < 0 || index >= s.mapPoints.length) {
        return s;
      }
      return editPoints(s, {
        camPoints: s.camPoints,
        mapPoints: s.mapPoints.filter((_, i) => i !== index),
      });
    }),
  undoPoint: () =>
    set((s) => {
      const previous = s.pastPoints[s.pastPoints.length - 1];
      if (!previous || s.pointEditStart) return s;
      return {
        ...afterPoints(previous.camPoints, previous.mapPoints, s.opticsDirty || pointsKey(previous) !== s.savedPointsKey),
        pastPoints: s.pastPoints.slice(0, -1),
        futurePoints: [...s.futurePoints, snapshot(s)].slice(-CALIBRATION_HISTORY_LIMIT),
      };
    }),
  redoPoint: () => set((s) => {
    const next = s.futurePoints[s.futurePoints.length - 1];
    if (!next || s.pointEditStart) return s;
    return {
      ...afterPoints(next.camPoints, next.mapPoints, s.opticsDirty || pointsKey(next) !== s.savedPointsKey),
      pastPoints: [...s.pastPoints, snapshot(s)].slice(-CALIBRATION_HISTORY_LIMIT),
      futurePoints: s.futurePoints.slice(0, -1),
    };
  }),
  beginPointEdit: () => set((s) => s.pointEditStart ? s : { pointEditStart: snapshot(s) }),
  endPointEdit: () => set((s) => {
    if (!s.pointEditStart) return s;
    const changed = pointsKey(s.pointEditStart) !== pointsKey(s);
    return {
      pointEditStart: null,
      pastPoints: changed ? [...s.pastPoints, s.pointEditStart].slice(-CALIBRATION_HISTORY_LIMIT) : s.pastPoints,
      futurePoints: changed ? [] : s.futurePoints,
    };
  }),
  resetPoints: () =>
    set((s) => editPoints(s, { camPoints: [], mapPoints: [] })),
  replacePoints: (camPoints, mapPoints, dirty = true) =>
    set((s) => dirty ? editPoints(s, { camPoints, mapPoints }) : {
      ...afterPoints(camPoints, mapPoints, s.opticsDirty),
      savedPointsKey: pointsKey({ camPoints, mapPoints }),
      pastPoints: [], futurePoints: [], pointEditStart: null,
    }),
  markClean: () => set((s) => ({ dirty: false, opticsDirty: false,
    savedOpticsKey: s.optics ? opticsPersistKey(s.optics) : null, savedPointsKey: pointsKey(s) })),
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
