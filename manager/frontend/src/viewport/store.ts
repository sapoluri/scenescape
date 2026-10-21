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
import { readSmoothTracksPreference, writeSmoothTracksPreference } from "./trackSmoothing";

export interface ViewportUiState {
  selectedId: string | null;
  viewPreset: ViewPreset;
  ortho: boolean;
  showGrid: boolean;
  showLabels: boolean;
  showTrails: boolean;
  showFrustums: boolean;
  showTelemetry: boolean;
  smoothTracks: boolean;
  activeTool: ToolId;
  /**
   * Camera entity ids whose live JPEGs are projected onto the scene map
   * (legacy "project frame" / "project all frames"). Empty = off.
   */
  projectCameraIds: string[];
  /**
   * Per-camera map-projection opacity 0–1 (Properties). Missing ids use
   * {@link DEFAULT_PROJECT_OPACITY}. Used whenever Project-all is off.
   */
  projectOpacityById: Record<string, number>;
  /**
   * True while the Cameras “Project all” control is engaged. Live opacity then
   * uses {@link projectAllOpacity} as an override; per-camera values are kept
   * for when Project-all is turned off.
   */
  projectAllActive: boolean;
  /** Shared overlay opacity while {@link projectAllActive} is true. */
  projectAllOpacity: number;
}

export const DEFAULT_PROJECT_OPACITY = 0.8;

export function projectOpacityOf(
  byId: Record<string, number>,
  cameraId: string,
): number {
  const v = byId[cameraId];
  if (typeof v !== "number" || !Number.isFinite(v)) {
    return DEFAULT_PROJECT_OPACITY;
  }
  return Math.min(1, Math.max(0, v));
}

function clampOpacity(opacity: number): number {
  return Math.min(1, Math.max(0, opacity));
}

/** Opacity applied to a projector right now (Project-all override or per-cam). */
export function effectiveProjectOpacity(
  s: Pick<
    ViewportUiState,
    "projectAllActive" | "projectAllOpacity" | "projectOpacityById"
  >,
  cameraId: string,
): number {
  if (s.projectAllActive) {
    return clampOpacity(s.projectAllOpacity);
  }
  return projectOpacityOf(s.projectOpacityById, cameraId);
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

/** Scene map GLB TRS (no mesh rebuild). */
const MAP_POSE_KEYS = new Set(["position", "rotation", "scale"]);

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
  if (prev.type === "map") {
    return keys.every((k) => MAP_POSE_KEYS.has(k));
  }
  return false;
}

interface ViewportStore extends ViewportUiState {
  entities: Record<string, ViewportEntity>;
  typeVisibility: Partial<Record<ViewportEntity["type"], boolean>>;
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
  /**
   * Insert or replace entities by id. Optional `removeIds` apply in the same
   * write (one subscribe) — used by live marks for frame-diff clear + upsert.
   */
  upsertEntities: (
    entities: ViewportEntity[],
    opts?: { fromMarks?: boolean; removeIds?: string[] },
  ) => void;
  /** Partial update of one entity. */
  updateEntity: (id: string, patch: Partial<ViewportEntity>) => void;
  setTypeVisible: (type: ViewportEntity["type"], visible: boolean) => void;
  removeEntity: (id: string) => void;
  /** Batch remove by id. `fromMarks` skips structVersion (live track churn). */
  removeEntities: (
    ids: string[],
    opts?: { fromMarks?: boolean },
  ) => void;
  removeEntitiesByType: (type: ViewportEntity["type"]) => void;
  clearEntities: () => void;
  select: (id: string | null) => void;
  setTool: (tool: ToolId) => void;
  /**
   * Bumped when a finished camera, region, or tripwire should open in the
   * outliner with the properties split ready to edit.
   */
  editFocus: number;
  /** Select `id`, return to the Live tool, and request the properties split. */
  beginEditing: (id: string) => void;
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
  /** Set stored per-camera opacity (Properties; applies when Project-all is off). */
  setProjectOpacity: (cameraId: string, opacity: number) => void;
  /** Set the Project-all overlay override (does not change per-camera values). */
  setProjectAllOpacity: (opacity: number) => void;
  toggleOverlay: (
    key: "showGrid" | "showLabels" | "showTrails" | "showFrustums" | "showTelemetry" | "smoothTracks",
  ) => void;
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
  showTelemetry: false,
  smoothTracks: readSmoothTracksPreference(),
  activeTool: "live",
  projectCameraIds: [],
  projectOpacityById: {},
  projectAllActive: false,
  projectAllOpacity: DEFAULT_PROJECT_OPACITY,
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
  typeVisibility: {},
  structVersion: 0,
  poseVersion: 0,
  editFocus: 0,
  past: [],
  future: [],
  upsertEntities: (entities, opts) =>
    set((s) => {
      const removeIds = opts?.removeIds;
      if (
        entities.length === 0 &&
        (!removeIds || removeIds.length === 0)
      ) {
        return s;
      }
      const next = { ...s.entities };
      let selectedId = s.selectedId;
      let projectCameraIds = s.projectCameraIds;
      if (removeIds?.length) {
        const removed = new Set<string>();
        for (const id of removeIds) {
          if (!(id in next)) {
            continue;
          }
          delete next[id];
          removed.add(id);
          getPersistState().dropBaseline(id);
        }
        if (removed.size > 0) {
          if (selectedId && removed.has(selectedId)) {
            selectedId = null;
          }
          projectCameraIds = projectCameraIds.filter((x) => !removed.has(x));
        }
      }
      for (const e of entities) {
        next[e.id] = opts?.fromMarks
          ? { ...e, visible: next[e.id]?.visible ?? s.typeVisibility[e.type] ?? e.visible }
          : e;
      }
      return {
        entities: next,
        selectedId,
        projectCameraIds,
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
  removeEntity: (id) => get().removeEntities([id]),
  setTypeVisible: (type, visible) =>
    set((state) => ({
      typeVisibility: { ...state.typeVisibility, [type]: visible },
      entities: Object.fromEntries(
        Object.entries(state.entities).map(([id, entity]) => [
          id,
          entity.type === type ? { ...entity, visible } : entity,
        ]),
      ),
      structVersion: state.structVersion + 1,
    })),
  removeEntities: (ids, opts) =>
    set((s) => {
      if (ids.length === 0) {
        return s;
      }
      const next = { ...s.entities };
      const removed = new Set<string>();
      for (const id of ids) {
        if (!(id in next)) {
          continue;
        }
        delete next[id];
        removed.add(id);
        getPersistState().dropBaseline(id);
      }
      if (removed.size === 0) {
        return s;
      }
      return {
        entities: next,
        selectedId:
          s.selectedId && removed.has(s.selectedId) ? null : s.selectedId,
        projectCameraIds: s.projectCameraIds.filter((x) => !removed.has(x)),
        structVersion: opts?.fromMarks
          ? s.structVersion
          : s.structVersion + 1,
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
      if (removedIds.size === 0) {
        return s;
      }
      return {
        entities: next,
        selectedId: removedSelected ? null : s.selectedId,
        projectCameraIds: s.projectCameraIds.filter((x) => !removedIds.has(x)),
        // Live marks are not structural scene graph — avoid mesh rebuilds.
        structVersion:
          type === "mark" ? s.structVersion : s.structVersion + 1,
      };
    }),
  clearEntities: () => {
    getPersistState().clearBaselines();
    return set((s) => ({
      entities: {},
      typeVisibility: {},
      selectedId: null,
      projectCameraIds: [],
      structVersion: s.structVersion + 1,
      past: [],
      future: [],
    }));
  },
  select: (id) =>
    set((s) => {
      // Changing selection (including deselect) returns to Live. Same-id
      // reselect keeps the current transform tool so gizmo mode persists.
      if (id === s.selectedId) {
        return { selectedId: id };
      }
      return { selectedId: id, activeTool: "live" };
    }),
  setTool: (activeTool) => set({ activeTool }),
  beginEditing: (id) =>
    set((s) => ({
      selectedId: id,
      activeTool: "live",
      editFocus: s.editFocus + 1,
    })),
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
        return { projectCameraIds: [], projectAllActive: false };
      }
      if (s.projectCameraIds.includes(id)) {
        return s.projectAllActive ? { projectAllActive: false } : s;
      }
      return {
        projectCameraIds: [...s.projectCameraIds, id],
        projectAllActive: false,
      };
    }),
  setProjectCameraEnabled: (id, on) =>
    set((s) => {
      const has = s.projectCameraIds.includes(id);
      let projectCameraIds = s.projectCameraIds;
      if (on && !has) {
        projectCameraIds = [...s.projectCameraIds, id];
      } else if (!on && has) {
        projectCameraIds = s.projectCameraIds.filter((x) => x !== id);
      } else {
        return s;
      }
      // Leaving Project-all restores per-camera opacities for remaining ids.
      return {
        projectCameraIds,
        projectAllActive: false,
      };
    }),
  setProjectAllCameras: (on) =>
    set((s) => {
      if (!on) {
        return { projectCameraIds: [], projectAllActive: false };
      }
      const ids = Object.values(s.entities)
        .filter(
          (e): e is Extract<ViewportEntity, { type: "camera" }> =>
            e.type === "camera" && e.posed,
        )
        .map((e) => e.id);
      return {
        projectCameraIds: ids,
        projectAllActive: true,
        // Keep projectOpacityById intact — override is projectAllOpacity only.
      };
    }),
  setProjectOpacity: (cameraId, opacity) =>
    set((s) => ({
      projectOpacityById: {
        ...s.projectOpacityById,
        [cameraId]: clampOpacity(opacity),
      },
    })),
  setProjectAllOpacity: (opacity) =>
    set({ projectAllOpacity: clampOpacity(opacity) }),
  toggleOverlay: (key) =>
    set((s) => {
      const enabled = !s[key];
      if (key === "smoothTracks") writeSmoothTracksPreference(enabled);
      return { [key]: enabled } as Partial<ViewportStore>;
    }),
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
