// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState, type ComponentType } from "react";
import WebViewer from "@rerun-io/web-viewer-react";
import { readAuthToken } from "../lib/authToken";
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
 *
 * Auth: the Rerun viewer fetches `rrd` with a plain GET (no Authorization
 * header). We download the recording with the session Token first and hand
 * the viewer a blob: URL so the protected download endpoint works.
 */

async function fetchRecordingBlobUrl(url: string): Promise<string> {
  const token = readAuthToken();
  const res = await fetch(url, {
    credentials: "same-origin",
    headers: {
      ...(token ? { Authorization: `Token ${token}` } : {}),
    },
  });
  if (!res.ok) {
    throw new Error(`Failed to download recording (${res.status})`);
  }
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

function RerunReplayView({ url }: { url: string }): React.JSX.Element {
  const [rrd, setRrd] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | null = null;
    setRrd(null);
    setError(null);
    fetchRecordingBlobUrl(url)
      .then((blobUrl) => {
        if (cancelled) {
          URL.revokeObjectURL(blobUrl);
          return;
        }
        objectUrl = blobUrl;
        setRrd(blobUrl);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(
            err instanceof Error ? err.message : "Failed to load recording",
          );
        }
      });
    return () => {
      cancelled = true;
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [url]);

  if (error) {
    return (
      <div className="ss-rerun-viewer ss-rerun-status" role="alert">
        <p>Failed to load Rerun recording.</p>
        <p className="ss-rerun-hint">{error}</p>
      </div>
    );
  }
  if (!rrd) {
    return (
      <div className="ss-rerun-viewer ss-rerun-status">
        <p>Loading recording…</p>
      </div>
    );
  }
  return (
    <div className="ss-rerun-viewer">
      <WebViewer rrd={rrd} width="100%" height="100%" />
    </div>
  );
}

export class RerunProvider implements ReplayProvider {
  readonly id = "rerun";

  async listRecordings(sceneId: string): Promise<RecordingMeta[]> {
    // Same endpoint as the stub; the manager serves .rrd metadata.
    try {
      const token = readAuthToken();
      const res = await fetch(
        `/api/v1/recordings/?scene=${encodeURIComponent(sceneId)}`,
        {
          credentials: "same-origin",
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
