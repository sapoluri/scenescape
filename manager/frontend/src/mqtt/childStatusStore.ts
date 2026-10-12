// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { create } from "zustand";

export type ChildLinkStatus = "unknown" | "connected" | "disconnected" | "error";

export type ChildStatusEntry = {
  status: ChildLinkStatus;
  message?: string;
};

interface ChildStatusStore {
  byRemoteId: Record<string, ChildStatusEntry>;
  setStatus: (remoteId: string, entry: ChildStatusEntry) => void;
  clear: () => void;
}

/**
 * Reduce an MQTT child-status payload (legacy contract: plain "connected" /
 * "disconnected", or a short error string) into a store entry.
 */
export function reduceChildStatusPayload(payload: string): ChildStatusEntry {
  const text = (payload || "").trim();
  if (text === "connected") {
    return { status: "connected" };
  }
  if (text === "disconnected" || text === "isConnected") {
    // Probe echo / down — treat as not connected until a real "connected".
    return { status: "disconnected" };
  }
  if (!text) {
    return { status: "unknown" };
  }
  return { status: "error", message: text };
}

export const useChildStatusStore = create<ChildStatusStore>()((set) => ({
  byRemoteId: {},
  setStatus: (remoteId, entry) =>
    set((s) => ({
      byRemoteId: { ...s.byRemoteId, [remoteId]: entry },
    })),
  clear: () => set({ byRemoteId: {} }),
}));

export function getChildStatusState(): ChildStatusStore {
  return useChildStatusStore.getState();
}
