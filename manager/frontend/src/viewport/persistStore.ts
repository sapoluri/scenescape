// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { create } from "zustand";
import { entityPersistKey } from "../lib/entityPersist";
import { getViewportState } from "./store";
import type { ViewportEntity } from "./types";

const PERSISTABLE = new Set(["camera", "sensor", "region", "tripwire"]);

function isPersistable(e: ViewportEntity): boolean {
  return PERSISTABLE.has(e.type);
}

interface PersistStore {
  /** Last-saved fingerprint per entity id. */
  baselines: Record<string, string>;
  /**
   * Capture current store fingerprints as clean (seed / after save).
   * `onlyMissing` skips ids that already have a baseline so geometry
   * reseeds do not wipe in-progress dirty state.
   */
  captureBaselines: (
    ids?: string[],
    opts?: { onlyMissing?: boolean },
  ) => void;
  /** Drop baseline when an entity is removed. */
  dropBaseline: (id: string) => void;
  clearBaselines: () => void;
}

export const usePersistStore = create<PersistStore>()((set, get) => ({
  baselines: {},
  captureBaselines: (ids, opts) => {
    const entities = getViewportState().entities;
    const next = { ...get().baselines };
    const targets =
      ids ??
      Object.keys(entities).filter((id) => {
        const e = entities[id];
        return e && isPersistable(e);
      });
    let changed = false;
    for (const id of targets) {
      if (opts?.onlyMissing && id in next) {
        continue;
      }
      const e = entities[id];
      if (e && isPersistable(e)) {
        next[id] = entityPersistKey(e);
        changed = true;
      }
    }
    if (changed) {
      set({ baselines: next });
    }
  },
  dropBaseline: (id) =>
    set((s) => {
      if (!(id in s.baselines)) {
        return s;
      }
      const next = { ...s.baselines };
      delete next[id];
      return { baselines: next };
    }),
  clearBaselines: () => set({ baselines: {} }),
}));

export const getPersistState = usePersistStore.getState;

/** Persistable entities whose fingerprint differs from the last-saved baseline. */
export function listDirtyPersistableEntities(): ViewportEntity[] {
  const { entities } = getViewportState();
  const { baselines } = getPersistState();
  const dirty: ViewportEntity[] = [];
  for (const e of Object.values(entities)) {
    if (!isPersistable(e)) {
      continue;
    }
    const key = entityPersistKey(e);
    const base = baselines[e.id];
    // No baseline yet (just created) counts as dirty once seeded fields exist.
    if (base === undefined || base !== key) {
      dirty.push(e);
    }
  }
  return dirty;
}

export function countDirtyPersistable(): number {
  return listDirtyPersistableEntities().length;
}

export function isEntityPersistDirty(id: string): boolean {
  const e = getViewportState().entities[id];
  if (!e || !isPersistable(e)) {
    return false;
  }
  const base = getPersistState().baselines[id];
  return base === undefined || base !== entityPersistKey(e);
}

export function markEntityPersistClean(id: string): void {
  getPersistState().captureBaselines([id]);
}
