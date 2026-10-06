// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ComponentType } from "react";

/**
 * Phase 2.6 frontend seam: the Replay UI talks only to this interface —
 * it never imports Rerun (or any provider) types directly.
 */

export interface RecordingMeta {
  id: string;
  sceneId: string;
  /** Epoch ms of the first logged sample. */
  start: number;
  /** Epoch ms of the last logged sample. */
  end: number;
  /** Bytes on disk, for display. */
  size: number;
  /** Provider id that owns this recording ('rerun' | 'mcap' | 'jsonl' | …). */
  provider: string;
  /** URL the provider's viewer reads (e.g. the .rrd file). */
  url: string;
}

export interface ReplayProvider {
  /** 'rerun' | 'mcap' | 'jsonl' | … — selected per deployment via config. */
  id: string;
  listRecordings(sceneId: string): Promise<RecordingMeta[]>;
  /**
   * Returns the React component rendered in the viewport area during
   * Replay mode for this recording. Replay is view-only: the component
   * must never mutate Scenescape entities.
   */
  createReplayView(recording: RecordingMeta): ComponentType;
}
