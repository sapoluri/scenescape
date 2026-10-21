// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { create } from "zustand";

type EditModeState = {
  workspaceActive: boolean;
  isSuperuser: boolean;
  editEnabled: boolean;
  canEdit: boolean;
  editSurfaceIds: string[];
  enterWorkspace: (isSuperuser: boolean) => void;
  leaveWorkspace: () => void;
  setEditing: (enabled: boolean) => void;
  setEditSurface: (id: string, open: boolean) => void;
};

export const useEditModeStore = create<EditModeState>()((set, get) => ({
  workspaceActive: false,
  isSuperuser: false,
  editEnabled: true,
  canEdit: false,
  editSurfaceIds: [],
  enterWorkspace: isSuperuser => set({
    workspaceActive: true, isSuperuser, canEdit: isSuperuser && get().editEnabled,
  }),
  leaveWorkspace: () => set({ workspaceActive: false, isSuperuser: false, canEdit: false, editSurfaceIds: [] }),
  setEditing: editEnabled => set(s => s.isSuperuser ? ({ editEnabled,
    canEdit: s.workspaceActive && editEnabled }) : s),
  setEditSurface: (id, open) => set(s => ({
    editSurfaceIds: open
      ? s.editSurfaceIds.includes(id) ? s.editSurfaceIds : [...s.editSurfaceIds, id]
      : s.editSurfaceIds.filter(surface => surface !== id),
  })),
}));

export function workspaceEditingAllowed(): boolean {
  const state = useEditModeStore.getState();
  return !state.workspaceActive || state.canEdit;
}

export function useWorkspaceEditing(): boolean {
  return useEditModeStore(state => !state.workspaceActive || state.canEdit);
}

export function assertWorkspaceWriteAllowed(method: string): void {
  if (!["GET", "HEAD", "OPTIONS"].includes(method.toUpperCase()) && !workspaceEditingAllowed()) {
    throw { status: 403, message: "Enable Edit to change scene configuration." };
  }
}