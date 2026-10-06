// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ComponentType } from "react";
import type { RecordingMeta, ReplayProvider } from "./types";
import { readAuthToken } from "../lib/authToken";
import "./StubProvider.css";

/**
 * Phase 2 stub provider: stands in for the Rerun-backed implementation
 * until the recorder service (2.0) is writing .rrd files. listRecordings
 * hits the real endpoint when it exists and falls back to empty.
 */

function StubReplayView(): React.JSX.Element {
  return (
    <div className="ss-replay-stub" role="status">
      <p className="ss-replay-stub-title">Replay view</p>
      <p className="ss-replay-stub-hint">
        The Rerun viewer lands here once the recorder service is writing
        recordings. Replay is view-only — it never edits the scene.
      </p>
    </div>
  );
}

export class StubProvider implements ReplayProvider {
  readonly id = "stub";

  async listRecordings(sceneId: string): Promise<RecordingMeta[]> {
    try {
      const token = readAuthToken();
      const res = await fetch(
        `/api/v1/recordings/?scene=${encodeURIComponent(sceneId)}`,
        {
          headers: {
            Accept: "application/json",
            ...(token ? { Authorization: `Token ${token}` } : {}),
          },
        },
      );
      if (!res.ok) {
        return [];
      }
      const data = (await res.json()) as { recordings?: RecordingMeta[] };
      return Array.isArray(data.recordings) ? data.recordings : [];
    } catch {
      return [];
    }
  }

  createReplayView(_recording: RecordingMeta): ComponentType {
    return StubReplayView;
  }
}
