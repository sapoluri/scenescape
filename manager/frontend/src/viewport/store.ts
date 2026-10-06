// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { create } from "zustand";
import {
  restoreGeometry,
  snapshotGeometry,
  type GeometrySnapshot,
} from "./geometrySync";
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
}

interface HistorySnapshot {
  /** Non-mark entities only — live marks are never part of undo. */
  entities: Record<string, ViewportEntity>;
  geometry: GeometrySnapshot;
}

interface ViewportStore extends ViewportUiState {
  entities: Record<string, ViewportEntity>;
  /**
   * Bumped on structural (non-mark) entity changes so the three.js layer
   * can reconcile without re-hashing on every live-mark update.
   */
  structVersion: number;
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
      return {
        entities: { ...s.entities, [id]: { ...prev, ...patch } as ViewportEntity },
        structVersion: s.structVersion + 1,
      };
    }),
  removeEntity: (id) =>
    set((s) => {
      if (!(id in s.entities)) {
        return s;
      }
      const next = { ...s.entities };
      delete next[id];
      return {
        entities: next,
        selectedId: s.selectedId === id ? null : s.selectedId,
        structVersion: s.structVersion + 1,
      };
    }),
  removeEntitiesByType: (type) =>
    set((s) => {
      const next: Record<string, ViewportEntity> = {};
      let removedSelected = false;
      for (const [id, e] of Object.entries(s.entities)) {
        if (e.type === type) {
          if (s.selectedId === id) {
            removedSelected = true;
          }
          continue;
        }
        next[id] = e;
      }
      return {
        entities: next,
        selectedId: removedSelected ? null : s.selectedId,
        structVersion: s.structVersion + 1,
      };
    }),
  clearEntities: () =>
    set((s) => ({
      entities: {},
      selectedId: null,
      structVersion: s.structVersion + 1,
      past: [],
      future: [],
    })),
  select: (id) => set({ selectedId: id }),
  setTool: (activeTool) => set({ activeTool }),
  setViewPreset: (viewPreset) => set({ viewPreset }),
  setOrtho: (ortho) => set({ ortho }),
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
