// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { create } from "zustand";
import type { ViewportEntity, ViewPreset } from "./types";

export interface ViewportUiState {
  selectedId: string | null;
  viewPreset: ViewPreset;
  ortho: boolean;
  showGrid: boolean;
  showLabels: boolean;
  showTrails: boolean;
  showFrustums: boolean;
}

interface ViewportStore extends ViewportUiState {
  entities: Record<string, ViewportEntity>;
  /** Insert or replace entities by id. */
  upsertEntities: (entities: ViewportEntity[]) => void;
  removeEntity: (id: string) => void;
  removeEntitiesByType: (type: ViewportEntity["type"]) => void;
  clearEntities: () => void;
  select: (id: string | null) => void;
  setViewPreset: (preset: ViewPreset) => void;
  setOrtho: (ortho: boolean) => void;
  toggleOverlay: (key: "showGrid" | "showLabels" | "showTrails" | "showFrustums") => void;
}

const initialUi: ViewportUiState = {
  selectedId: null,
  viewPreset: "persp",
  ortho: false,
  showGrid: true,
  showLabels: true,
  showTrails: true,
  showFrustums: true,
};

/**
 * Scene-graph store for the single-pane 3D viewport. Entities are keyed by
 * id; the three.js layer subscribes and reconciles meshes imperatively
 * (no React re-render per frame).
 */
export const useViewportStore = create<ViewportStore>()((set) => ({
  ...initialUi,
  entities: {},
  upsertEntities: (entities) =>
    set((s) => {
      const next = { ...s.entities };
      for (const e of entities) {
        next[e.id] = e;
      }
      return { entities: next };
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
      };
    }),
  clearEntities: () => set({ entities: {}, selectedId: null }),
  select: (id) => set({ selectedId: id }),
  setViewPreset: (viewPreset) => set({ viewPreset }),
  setOrtho: (ortho) => set({ ortho }),
  toggleOverlay: (key) => set((s) => ({ [key]: !s[key] }) as Partial<ViewportStore>),
}));

/** Non-reactive snapshot access for the imperative three.js layer. */
export const getViewportState = useViewportStore.getState;
