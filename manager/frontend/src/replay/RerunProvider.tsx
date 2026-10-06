// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ComponentType } from "react";
import WebViewer from "@rerun-io/web-viewer-react";
import type { RecordingMeta, ReplayProvider } from "./types";
import "./RerunProvider.css";

/**
 * Phase 2.2 Rerun-backed ReplayProvider.
 *
 * Renders the self-hosted Rerun web viewer (npm bundle — no hotlinking
 * app.rerun.io per the 2.5 caveats) for the selected .rrd recording.
 * The viewer is view-only: it never mutates Scenescape entities.
 *
 * Version lock (2.5): @rerun-io/web-viewer-react must match the rerun-sdk
 * version that wrote the .rrd files. Both are pinned to 0.38.1; the
 * recorder stamps the SDK version in each recording's metadata.
 */

function RerunReplayView({ url }: { url: string }): React.JSX.Element {
  return (
    <div className="ss-rerun-viewer">
      <WebViewer rrd={url} width="100%" height="100%" />
    </div>
  );
}

export class RerunProvider implements ReplayProvider {
  readonly id = "rerun";

  async listRecordings(sceneId: string): Promise<RecordingMeta[]> {
    // Same endpoint as the stub; the manager serves .rrd metadata.
    const { readAuthToken } = await import("../lib/authToken");
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
      const all = Array.isArray(data.recordings) ? data.recordings : [];
      // Only recordings this provider can render.
      return all.filter((r) => r.provider === "rerun");
    } catch {
      return [];
    }
  }

  createReplayView(recording: RecordingMeta): ComponentType {
    const url = recording.url;
    return function RerunView(): React.JSX.Element {
      return <RerunReplayView url={url} />;
    };
  }
}
