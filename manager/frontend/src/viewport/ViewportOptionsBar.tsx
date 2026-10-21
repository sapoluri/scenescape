// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect } from "react";
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
  onExportScene?: () => void;
  sceneChrome?: SceneChromeActions | null;
};

function OverlayCheck({
  checked,
  onChange,
  label,
  title,
}: {
  checked: boolean;
  onChange: () => void;
  label: string;
  title?: string;
}) {
  return (
    <label className={`vchk${checked ? " on" : ""}`} title={title}>
      <input type="checkbox" checked={checked} onChange={onChange} />
      <span className="box" aria-hidden="true" />
      {label}
    </label>
  );
}

/**
 * Drive the parked hard-contract `#show-telemetry` (sscape.js) without a
 * second id — camera FPS + scene rate only update while it is checked.
 */
function useShowTelemetry(): [boolean, () => void] {
  const on = useViewportStore((state) => state.showTelemetry);

  useEffect(() => {
    const el = document.getElementById(
      "show-telemetry",
    ) as HTMLInputElement | null;
    if (!el) {
      return;
    }
    const sync = () => useViewportStore.setState({ showTelemetry: el.checked });
    sync();
    el.addEventListener("change", sync);
    el.dispatchEvent(new Event("change", { bubbles: true }));
    return () => el.removeEventListener("change", sync);
  }, []);

  const toggle = useCallback(() => {
    const el = document.getElementById(
      "show-telemetry",
    ) as HTMLInputElement | null;
    if (!el) {
      return;
    }
    el.checked = !el.checked;
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }, []);

  return [on, toggle];
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
  onExportScene,
  sceneChrome = null,
}: Props) {
  const ortho = useViewportStore((s) => s.ortho);
  const showGrid = useViewportStore((s) => s.showGrid);
  const showLabels = useViewportStore((s) => s.showLabels);
  const showTrails = useViewportStore((s) => s.showTrails);
  const setOrtho = useViewportStore((s) => s.setOrtho);
  const toggleOverlay = useViewportStore((s) => s.toggleOverlay);
  const [showTelemetry, toggleTelemetry] = useShowTelemetry();
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
            <button
              type="button"
              className="ss-viewport-bar-btn"
              id="export-scene"
              title={`Export ${sceneChrome.sceneName}`}
              aria-label={`Export ${sceneChrome.sceneName}`}
              onClick={onExportScene}
            >
              <i className="bi bi-box-arrow-up" aria-hidden="true" />
              <span className="sr-only">Export</span>
            </button>
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
      <div
        className="ss-viewport-overlays"
        role="group"
        aria-label="Overlays"
      >
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
            <OverlayCheck
              checked={showTelemetry}
              onChange={toggleTelemetry}
              label="Telemetry"
              title="Show scene controller rate and camera preview FPS"
            />
          </>
        ) : null}
      </div>
      {showScene ? (
        <>
          <div className="ss-viewport-bar-spacer" aria-hidden="true" />
          <div className="scene-rate ss-scene-rate telemetry-hide" hidden={!showTelemetry}>
            Rate: &nbsp;
            <span id="scene-rate">{sceneChrome.sceneRate ?? "--"}</span>
            &nbsp; Hz
          </div>
        </>
      ) : null}
    </div>
  );
}
