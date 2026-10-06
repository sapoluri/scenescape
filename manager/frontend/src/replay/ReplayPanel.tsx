// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";
import {
  resolveReplayProvider,
  type RecordingMeta,
} from "./index";
import "./ReplayPanel.css";

/**
 * Replay shell (Phase 2.2): recording picker + provider view.
 * View-only by contract — the provider component never edits the scene.
 */

function formatTime(ms: number): string {
  return new Date(ms).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function ReplayPanel({
  sceneId,
  onClose,
}: {
  sceneId: string;
  onClose: () => void;
}): React.JSX.Element {
  const [recordings, setRecordings] = useState<RecordingMeta[] | null>(null);
  const [selected, setSelected] = useState<RecordingMeta | null>(null);
  const provider = resolveReplayProvider();

  useEffect(() => {
    let cancelled = false;
    provider.listRecordings(sceneId).then((list) => {
      if (!cancelled) {
        setRecordings(list);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [sceneId, provider]);

  if (selected) {
    const View = provider.createReplayView(selected);
    return (
      <div className="ss-replay">
        <div className="ss-replay-bar">
          <button
            type="button"
            className="ss-viewport-bar-btn"
            onClick={() => setSelected(null)}
            title="Back to recordings"
          >
            <i className="bi bi-arrow-left" aria-hidden="true" />
            <span>Recordings</span>
          </button>
          <span className="ss-replay-title">
            {formatTime(selected.start)} – {formatTime(selected.end)}
          </span>
          <button
            type="button"
            className="ss-viewport-bar-btn"
            onClick={onClose}
            title="Back to live"
          >
            <i className="bi bi-broadcast" aria-hidden="true" />
            <span>Live</span>
          </button>
        </div>
        <div className="ss-replay-view">
          <View />
        </div>
      </div>
    );
  }

  return (
    <div className="ss-replay">
      <div className="ss-replay-bar">
        <span className="ss-replay-title">Recordings</span>
        <button
          type="button"
          className="ss-viewport-bar-btn"
          onClick={onClose}
          title="Back to live"
        >
          <i className="bi bi-broadcast" aria-hidden="true" />
          <span>Live</span>
        </button>
      </div>
      <div className="ss-replay-list">
        {recordings === null ? (
          <p className="ss-replay-status">Loading recordings…</p>
        ) : recordings.length === 0 ? (
          <div className="ss-replay-empty">
            <p>No recordings yet.</p>
            <p className="ss-replay-hint">
              The recorder service writes time-partitioned recordings per
              scene. They will appear here once recording is enabled.
            </p>
          </div>
        ) : (
          <ul>
            {recordings.map((r) => (
              <li key={r.id}>
                <button type="button" onClick={() => setSelected(r)}>
                  <span className="ss-replay-r-name">
                    {formatTime(r.start)} – {formatTime(r.end)}
                  </span>
                  <span className="ss-replay-r-meta">
                    {formatSize(r.size)} · {r.provider}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
