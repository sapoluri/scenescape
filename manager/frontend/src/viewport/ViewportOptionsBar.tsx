// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useViewportStore } from "./store";

export type SceneChromeActions = {
  sceneName: string;
  isSuperuser?: boolean;
  /** When set, show Delete (hard-contract #scene-delete). */
  canDelete?: boolean;
  sceneRate?: string;
  onDeleteScene?: () => void;
};

type Props = {
  /** Empty home shell: hide Mode, scene chrome, and scene-only overlays. */
  emptyScene?: boolean;
  calibrating?: boolean;
  mode?: "live" | "replay";
  onModeChange?: (mode: "live" | "replay") => void;
  sceneChrome?: SceneChromeActions | null;
};

function OverlayCheck({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: () => void;
  label: string;
}) {
  return (
    <label className={`vchk${checked ? " on" : ""}`}>
      <input type="checkbox" checked={checked} onChange={onChange} />
      <span className="box" aria-hidden="true" />
      {label}
    </label>
  );
}

/**
 * Viewport options bar: scene identity/actions (when open) + how the 3D view
 * is shown (Mode, projection, overlays). App navigation / MQTT live on
 * ViewportMenuBar; edit tools on ToolHeader.
 */
export function ViewportOptionsBar({
  emptyScene = false,
  calibrating = false,
  mode = "live",
  onModeChange,
  sceneChrome = null,
}: Props) {
  const ortho = useViewportStore((s) => s.ortho);
  const showGrid = useViewportStore((s) => s.showGrid);
  const showLabels = useViewportStore((s) => s.showLabels);
  const showTrails = useViewportStore((s) => s.showTrails);
  const setOrtho = useViewportStore((s) => s.setOrtho);
  const toggleOverlay = useViewportStore((s) => s.toggleOverlay);
  const showScene = !emptyScene && sceneChrome;

  return (
    <div
      className="ss-viewport-bar ss-viewport-options"
      id={showScene ? "ss-scene-chrome" : undefined}
      role="toolbar"
      aria-label="Viewport options"
    >
      {showScene ? (
        <>
          <div
            className="ss-scene-header-actions ss-viewport-scene-actions"
            role="group"
            aria-label="Scene"
          >
            <h2 className="ss-viewport-scene-name" id="scene_name">
              {sceneChrome.sceneName}
            </h2>
            <a
              className="ss-viewport-bar-btn"
              id="export-scene"
              href="#"
              title={`Export ${sceneChrome.sceneName}`}
            >
              <i className="bi bi-box-arrow-up" aria-hidden="true" />
              <span className="sr-only">Export</span>
            </a>
            {sceneChrome.isSuperuser ? (
              <a
                className="ss-viewport-bar-btn"
                id="scene-edit"
                href="?ss=scene-manage"
                title={`Edit ${sceneChrome.sceneName}`}
                aria-label={`Edit ${sceneChrome.sceneName}`}
              >
                <i className="bi bi-pencil" aria-hidden="true" />
              </a>
            ) : null}
            {sceneChrome.isSuperuser && sceneChrome.canDelete ? (
              <button
                type="button"
                className="ss-viewport-bar-btn ss-viewport-bar-btn--danger"
                id="scene-delete"
                title={`Delete ${sceneChrome.sceneName}`}
                aria-label={`Delete ${sceneChrome.sceneName}`}
                onClick={() => sceneChrome.onDeleteScene?.()}
              >
                <i className="bi bi-trash" aria-hidden="true" />
              </button>
            ) : null}
          </div>
          <span className="ss-viewport-bar-sep" aria-hidden="true" />
          <span className="ss-viewport-bar-label">Mode</span>
          <div className="seg" role="group" aria-label="Live or replay">
            <button
              type="button"
              className={mode === "live" ? "on" : ""}
              onClick={() => onModeChange?.("live")}
              title="Live view (MQTT)"
              disabled={calibrating}
            >
              Live
            </button>
            <button
              type="button"
              className={mode === "replay" ? "on" : ""}
              onClick={() => onModeChange?.("replay")}
              title="Replay a recording"
              disabled={calibrating}
            >
              Replay
            </button>
          </div>
          <span className="ss-viewport-bar-sep" aria-hidden="true" />
        </>
      ) : null}
      <span className="ss-viewport-bar-label">View</span>
      <div className="seg" role="group" aria-label="Projection">
        <button
          type="button"
          className={!ortho ? "on" : ""}
          onClick={() => setOrtho(false)}
          title="Perspective view (5)"
        >
          Persp
        </button>
        <button
          type="button"
          className={ortho ? "on" : ""}
          onClick={() => setOrtho(true)}
          title="Top-down orthographic map view for ROIs and tripwires (5)"
        >
          Ortho
        </button>
      </div>
      <span className="ss-viewport-bar-sep" aria-hidden="true" />
      <span className="ss-viewport-bar-label">Overlays</span>
      <OverlayCheck
        checked={showGrid}
        onChange={() => toggleOverlay("showGrid")}
        label="Grid"
      />
      {!emptyScene ? (
        <>
          <OverlayCheck
            checked={showTrails}
            onChange={() => toggleOverlay("showTrails")}
            label="Trails"
          />
          <OverlayCheck
            checked={showLabels}
            onChange={() => toggleOverlay("showLabels")}
            label="Labels"
          />
        </>
      ) : null}
      {showScene ? (
        <>
          <div className="ss-viewport-bar-spacer" aria-hidden="true" />
          <div className="scene-rate ss-scene-rate telemetry-hide">
            Rate: &nbsp;
            <span id="scene-rate">{sceneChrome.sceneRate ?? "--"}</span>
            &nbsp; Hz
          </div>
        </>
      ) : null}
    </div>
  );
}
