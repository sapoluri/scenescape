// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { create } from "zustand";
import {
  restoreGeometry,
  snapshotGeometry,
  type GeometrySnapshot,
} from "./geometrySync";
import { getPersistState } from "./persistStore";
import type { ToolId, ViewportEntity, ViewPreset } from "./types";

export interface ViewportUiState {
  selectedId: string | null;
  viewPreset: ViewPreset;
  ortho: boolean;
  showGrid: boolean;
  showLabels: boolean;
  showTrails: boolean;
  showFrustums: boolean;
  activeTool: ToolId;
  /**
   * Camera entity ids whose live JPEGs are projected onto the scene map
   * (legacy "project frame" / "project all frames"). Empty = off.
   */
  projectCameraIds: string[];
  /** Map-projection overlay opacity 0–1 (legacy all-cameras opacity). */
  projectOpacity: number;
}

interface HistorySnapshot {
  /** Non-mark entities only — live marks are never part of undo. */
  entities: Record<string, ViewportEntity>;
  geometry: GeometrySnapshot;
}

/** Camera fields that only move/aim the existing three.js rig (no rebuild). */
const CAMERA_POSE_KEYS = new Set([
  "position",
  "rotation",
  "restRotation",
  "posed",
  "fov",
  "target",
]);

/** Sensor fields that only translate the existing node (no rebuild). */
const SENSOR_POSE_KEYS = new Set(["position"]);

function isPoseOnlyPatch(
  prev: ViewportEntity,
  patch: Partial<ViewportEntity>,
): boolean {
  const keys = Object.keys(patch);
  if (keys.length === 0) {
    return false;
  }
  if (prev.type === "camera") {
    return keys.every((k) => CAMERA_POSE_KEYS.has(k));
  }
  if (prev.type === "sensor") {
    return keys.every((k) => SENSOR_POSE_KEYS.has(k));
  }
  return false;
}

interface ViewportStore extends ViewportUiState {
  entities: Record<string, ViewportEntity>;
  /**
   * Bumped on structural (non-mark) entity changes so the three.js layer
   * can reconcile without re-hashing on every live-mark update.
   */
  structVersion: number;
  /**
   * Bumped when only camera/sensor pose fields change so the three.js layer
   * can update transforms without disposing and rebuilding meshes.
   */
  poseVersion: number;
  past: HistorySnapshot[];
  future: HistorySnapshot[];
  /** Insert or replace entities by id. */
  upsertEntities: (
    entities: ViewportEntity[],
    opts?: { fromMarks?: boolean },
  ) => void;
  /** Partial update of one entity. */
  updateEntity: (id: string, patch: Partial<ViewportEntity>) => void;
  removeEntity: (id: string) => void;
  removeEntitiesByType: (type: ViewportEntity["type"]) => void;
  clearEntities: () => void;
  select: (id: string | null) => void;
  setTool: (tool: ToolId) => void;
  setViewPreset: (preset: ViewPreset) => void;
  setOrtho: (ortho: boolean) => void;
  /**
   * Enable projection for one camera (add) or clear every projector (`null`).
   * Does not turn other cameras off when enabling — use setProjectCameraEnabled
   * to toggle a single id.
   */
  setProjectCamera: (id: string | null) => void;
  setProjectCameraEnabled: (id: string, on: boolean) => void;
  /** Toggle every posed camera's map projection (legacy "project all frames"). */
  setProjectAllCameras: (on: boolean) => void;
  setProjectOpacity: (opacity: number) => void;
  toggleOverlay: (key: "showGrid" | "showLabels" | "showTrails" | "showFrustums") => void;
  /**
   * Snapshot the current structural state onto the undo stack. Call BEFORE
   * a tool mutation (creation finish, gizmo drag start, delete).
   */
  commitHistory: () => void;
  undo: () => void;
  redo: () => void;
}

const initialUi: ViewportUiState = {
  selectedId: null,
  viewPreset: "persp",
  ortho: false,
  showGrid: true,
  showLabels: true,
  showTrails: true,
  showFrustums: true,
  activeTool: "live",
  projectCameraIds: [],
  projectOpacity: 0.8,
};

const HISTORY_LIMIT = 50;

function structuralSnapshot(
  entities: Record<string, ViewportEntity>,
): HistorySnapshot {
  const structural: Record<string, ViewportEntity> = {};
  for (const [id, e] of Object.entries(entities)) {
    if (e.type !== "mark") {
      structural[id] = e;
    }
  }
  return { entities: structural, geometry: snapshotGeometry() };
}

/**
 * Scene-graph store for the single-pane 3D viewport. Entities are keyed by
 * id; the three.js layer subscribes and reconciles meshes imperatively
 * (no React re-render per frame).
 */
export const useViewportStore = create<ViewportStore>()((set, get) => ({
  ...initialUi,
  entities: {},
  structVersion: 0,
  poseVersion: 0,
  past: [],
  future: [],
  upsertEntities: (entities, opts) =>
    set((s) => {
      const next = { ...s.entities };
      for (const e of entities) {
        next[e.id] = e;
      }
      return {
        entities: next,
        structVersion: opts?.fromMarks ? s.structVersion : s.structVersion + 1,
      };
    }),
  updateEntity: (id, patch) =>
    set((s) => {
      const prev = s.entities[id];
      if (!prev) {
        return s;
      }
      const poseOnly = isPoseOnlyPatch(prev, patch);
      return {
        entities: {
          ...s.entities,
          [id]: { ...prev, ...patch } as ViewportEntity,
        },
        ...(poseOnly
          ? { poseVersion: s.poseVersion + 1 }
          : { structVersion: s.structVersion + 1 }),
      };
    }),
  removeEntity: (id) =>
    set((s) => {
      if (!(id in s.entities)) {
        return s;
      }
      const next = { ...s.entities };
      delete next[id];
      getPersistState().dropBaseline(id);
      return {
        entities: next,
        selectedId: s.selectedId === id ? null : s.selectedId,
        projectCameraIds: s.projectCameraIds.filter((x) => x !== id),
        structVersion: s.structVersion + 1,
      };
    }),
  removeEntitiesByType: (type) =>
    set((s) => {
      const next: Record<string, ViewportEntity> = {};
      let removedSelected = false;
      const removedIds = new Set<string>();
      for (const [id, e] of Object.entries(s.entities)) {
        if (e.type === type) {
          if (s.selectedId === id) {
            removedSelected = true;
          }
          removedIds.add(id);
          getPersistState().dropBaseline(id);
          continue;
        }
        next[id] = e;
      }
      return {
        entities: next,
        selectedId: removedSelected ? null : s.selectedId,
        projectCameraIds: s.projectCameraIds.filter((x) => !removedIds.has(x)),
        structVersion: s.structVersion + 1,
      };
    }),
  clearEntities: () => {
    getPersistState().clearBaselines();
    return set((s) => ({
      entities: {},
      selectedId: null,
      projectCameraIds: [],
      structVersion: s.structVersion + 1,
      past: [],
      future: [],
    }));
  },
  select: (id) => set({ selectedId: id }),
  setTool: (activeTool) => set({ activeTool }),
  setViewPreset: (viewPreset) => set({ viewPreset }),
  /**
   * Ortho is the map-style / 2D editing view: always pairs with the top
   * preset so ROIs and tripwires are drawn on a flat plan. Persp restores
   * the angled perspective preset.
   */
  setOrtho: (ortho) =>
    set({
      ortho,
      viewPreset: ortho ? "top" : "persp",
    }),
  setProjectCamera: (id) =>
    set((s) => {
      if (id == null) {
        return { projectCameraIds: [] };
      }
      if (s.projectCameraIds.includes(id)) {
        return s;
      }
      return { projectCameraIds: [...s.projectCameraIds, id] };
    }),
  setProjectCameraEnabled: (id, on) =>
    set((s) => {
      const has = s.projectCameraIds.includes(id);
      if (on && !has) {
        return { projectCameraIds: [...s.projectCameraIds, id] };
      }
      if (!on && has) {
        return { projectCameraIds: s.projectCameraIds.filter((x) => x !== id) };
      }
      return s;
    }),
  setProjectAllCameras: (on) =>
    set((s) => {
      if (!on) {
        return { projectCameraIds: [] };
      }
      const ids = Object.values(s.entities)
        .filter(
          (e): e is Extract<ViewportEntity, { type: "camera" }> =>
            e.type === "camera" && e.posed,
        )
        .map((e) => e.id);
      return { projectCameraIds: ids };
    }),
  setProjectOpacity: (opacity) =>
    set({
      projectOpacity: Math.min(1, Math.max(0, opacity)),
    }),
  toggleOverlay: (key) => set((s) => ({ [key]: !s[key] }) as Partial<ViewportStore>),
  commitHistory: () =>
    set((s) => ({
      past: [...s.past, structuralSnapshot(s.entities)].slice(-HISTORY_LIMIT),
      future: [],
    })),
  undo: () => {
    const s = get();
    const prev = s.past[s.past.length - 1];
    if (!prev) {
      return;
    }
    restoreGeometry(prev.geometry);
    set((state) => {
      const marks: Record<string, ViewportEntity> = {};
      for (const [id, e] of Object.entries(state.entities)) {
        if (e.type === "mark") {
          marks[id] = e;
        }
      }
      const selectedGone =
        state.selectedId != null && !(state.selectedId in prev.entities);
      return {
        entities: { ...prev.entities, ...marks },
        selectedId: selectedGone ? null : state.selectedId,
        past: state.past.slice(0, -1),
        future: [...state.future, structuralSnapshot(state.entities)],
        structVersion: state.structVersion + 1,
      };
    });
  },
  redo: () => {
    const s = get();
    const nextSnap = s.future[s.future.length - 1];
    if (!nextSnap) {
      return;
    }
    restoreGeometry(nextSnap.geometry);
    set((state) => {
      const marks: Record<string, ViewportEntity> = {};
      for (const [id, e] of Object.entries(state.entities)) {
        if (e.type === "mark") {
          marks[id] = e;
        }
      }
      const selectedGone =
        state.selectedId != null && !(state.selectedId in nextSnap.entities);
      return {
        entities: { ...nextSnap.entities, ...marks },
        selectedId: selectedGone ? null : state.selectedId,
        past: [...state.past, structuralSnapshot(state.entities)],
        future: state.future.slice(0, -1),
        structVersion: state.structVersion + 1,
      };
    });
  },
}));

/** Non-reactive snapshot access for the imperative three.js layer. */
export const getViewportState = useViewportStore.getState;

declare global {
  interface Window {
    /** Debug/test handle for in-browser viewport automation. */
    __ssViewport?: {
      getState: typeof getViewportState;
      store: typeof useViewportStore;
    };
  }
}

if (typeof window !== "undefined") {
  window.__ssViewport = {
    getState: getViewportState,
    store: useViewportStore,
  };
}
