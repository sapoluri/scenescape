// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from "react";
import { persistGeometry } from "../lib/legacyBridge";
import { CameraStripOverlay } from "../panels/CameraStripOverlay";
import type {
  SceneCameraBootstrap,
  SceneSensorBootstrap,
} from "../scene/types";
import { CalibrateFeedPane } from "./calibrate/CalibrateFeedPane";
import { CalibrateWizard } from "./calibrate/CalibrateWizard";
import { useCalibrateStore } from "./calibrate/calibrateStore";
import { useCalibrateSession } from "./calibrate/useCalibrateSession";
import { useViewportEntities } from "./entities";
import { useViewportMarks } from "./marks";
import { attachMarkRenderer } from "./renderMarks";
import { getViewportState, useViewportStore } from "./store";
import { TOOLS, toolDef } from "./tools";
import type { ToolId } from "./types";
import { useViewportTools } from "./useViewportTools";
import "./viewport.css";
import { createViewportWorld, type ViewportTheme, type ViewportWorld } from "./world";

/**
 * Single-pane 3D viewport (Phase 1.1 foundation, Phase 1.2 tools).
 *
 * Z-up world matching the Scenescape domain. Composition:
 * - `useViewportMarks` feeds live tracked-object marks from the frozen
 *   `ss-scene-objects` event into the store;
 * - `attachMarkRenderer` draws them imperatively;
 * - `useViewportEntities` seeds regions/tripwires (geometry model) and
 *   cameras/sensors (bootstrap + REST poses) and reconciles their meshes;
 * - `useViewportTools` wires selection, the TransformControls gizmo,
 *   creation gestures, and keyboard shortcuts.
 * React never re-renders per frame.
 */

interface SceneViewportProps {
  sceneId: string;
  assetMarkColors?: Record<string, string>;
  cameras: SceneCameraBootstrap[];
  sensors: SceneSensorBootstrap[];
  cameraRates?: Record<string, string>;
  authToken: string;
  isSuperuser?: boolean;
  onOpenLibrary?: () => void;
  onOpenScene?: () => void;
  mode?: "live" | "replay";
  onModeChange?: (mode: "live" | "replay") => void;
}

function readTheme(): ViewportTheme {
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

function useTheme(): ViewportTheme {
  const [theme, setTheme] = useState<ViewportTheme>(readTheme);
  useEffect(() => {
    const mo = new MutationObserver(() => setTheme(readTheme()));
    mo.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    return () => mo.disconnect();
  }, []);
  return theme;
}

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

/** Rekey temp `draft-*` ids to server uids after a geometry save. */
function remapStoreIds(idMap: Record<string, string>): void {
  const s = getViewportState();
  let changed = false;
  const next = { ...s.entities };
  for (const [oldId, newId] of Object.entries(idMap)) {
    if (!oldId || !newId || oldId === newId) {
      continue;
    }
    const entity = next[oldId];
    if (!entity) {
      continue;
    }
    delete next[oldId];
    next[newId] = { ...entity, id: newId };
    changed = true;
  }
  if (!changed) {
    return;
  }
  useViewportStore.setState((state) => ({
    entities: next,
    selectedId:
      state.selectedId && idMap[state.selectedId]
        ? idMap[state.selectedId]
        : state.selectedId,
    structVersion: state.structVersion + 1,
  }));
}

function ToolHeader() {
  const activeTool = useViewportStore((s) => s.activeTool);
  const setTool = useViewportStore((s) => s.setTool);
  const canUndo = useViewportStore((s) => s.past.length > 0);
  const canRedo = useViewportStore((s) => s.future.length > 0);
  const undo = useViewportStore((s) => s.undo);
  const redo = useViewportStore((s) => s.redo);
  const [roiDirty, setRoiDirty] = useState(false);
  const [tripDirty, setTripDirty] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const onRoi = (e: Event) =>
      setRoiDirty(Boolean((e as CustomEvent<boolean>).detail));
    const onTrip = (e: Event) =>
      setTripDirty(Boolean((e as CustomEvent<boolean>).detail));
    window.addEventListener("ss-roi-dirty", onRoi);
    window.addEventListener("ss-trip-dirty", onTrip);
    return () => {
      window.removeEventListener("ss-roi-dirty", onRoi);
      window.removeEventListener("ss-trip-dirty", onTrip);
    };
  }, []);

  const dirty = roiDirty || tripDirty;

  const onSave = async () => {
    if (saving || !dirty) {
      return;
    }
    setSaving(true);
    try {
      const result = await persistGeometry();
      if (result && typeof result === "object") {
        remapStoreIds(result.roiIds);
        remapStoreIds(result.tripIds);
      }
    } finally {
      setSaving(false);
    }
  };

  const pickTool = (id: ToolId) => {
    setTool(id);
  };

  return (
    <div className="ss-viewport-tools" role="toolbar" aria-label="Edit tools">
      <span className="ss-viewport-bar-label">Tools</span>
      <div className="ss-tools-group" role="group" aria-label="Tools">
        {TOOLS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={`ss-tool-btn${activeTool === t.id ? " on" : ""}`}
            onClick={() => pickTool(t.id)}
            title={`${t.label} (${t.shortcut}) — ${t.hint}`}
            aria-pressed={activeTool === t.id}
          >
            <i className={`bi ${t.icon}`} aria-hidden="true" />
            <span className="ss-tool-label">{t.label}</span>
          </button>
        ))}
      </div>
      <span className="ss-viewport-bar-sep" aria-hidden="true" />
      <div className="ss-tools-group" role="group" aria-label="History">
        <button
          type="button"
          className="ss-tool-btn"
          disabled={!canUndo}
          onClick={undo}
          title="Undo (Ctrl+Z)"
          aria-label="Undo"
        >
          <i className="bi bi-arrow-counterclockwise" aria-hidden="true" />
        </button>
        <button
          type="button"
          className="ss-tool-btn"
          disabled={!canRedo}
          onClick={redo}
          title="Redo (Ctrl+Shift+Z)"
          aria-label="Redo"
        >
          <i className="bi bi-arrow-clockwise" aria-hidden="true" />
        </button>
      </div>
      <span className="ss-viewport-bar-sep" aria-hidden="true" />
      <button
        type="button"
        className={`ss-tool-btn ss-tool-save${dirty ? " is-dirty" : ""}`}
        disabled={!dirty || saving}
        onClick={onSave}
        title={
          dirty
            ? "Save regions/tripwires to the server (same as the 2D editors)"
            : "No unsaved geometry changes"
        }
      >
        <i className="bi bi-check-lg" aria-hidden="true" />
        <span className="ss-tool-label">{saving ? "Saving…" : "Save"}</span>
      </button>
      <span
        className="ss-tool-hint"
        title={toolDef(activeTool).hint}
        aria-hidden="true"
      >
        {toolDef(activeTool).hint}
      </span>
    </div>
  );
}

export function SceneViewport({
  sceneId,
  assetMarkColors,
  cameras,
  sensors,
  cameraRates = {},
  authToken,
  isSuperuser = false,
  onOpenLibrary,
  onOpenScene,
  mode = "live",
  onModeChange,
}: SceneViewportProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const labelLayerRef = useRef<HTMLDivElement>(null);
  const [world, setWorld] = useState<ViewportWorld | null>(null);
  const [webglFailed, setWebglFailed] = useState(false);
  const theme = useTheme();

  useViewportMarks({ assetMarkColors });

  useEffect(() => {
    const host = hostRef.current;
    const labelLayer = labelLayerRef.current;
    if (!host || !labelLayer) {
      return;
    }
    const w = createViewportWorld(host, labelLayer);
    if (!w) {
      setWebglFailed(true);
      return;
    }
    setWorld(w);
    const detachMarks = attachMarkRenderer(w);
    return () => {
      detachMarks();
      setWorld(null);
      w.dispose();
    };
  }, [sceneId]);

  useEffect(() => {
    world?.setTheme(theme);
  }, [world, theme]);

  useViewportEntities(world, { cameras, sensors, authToken });
  useViewportTools(world);
  useCalibrateSession(world, authToken);

  const calibrating = useCalibrateStore((s) => s.active);

  // View state -> world (presets, projection, grid).
  useEffect(() => {
    if (!world) {
      return;
    }
    return useViewportStore.subscribe((s, prev) => {
      if (s.viewPreset !== prev.viewPreset || s.ortho !== prev.ortho) {
        world.setActiveCamera(s.ortho);
      }
      if (s.showGrid !== prev.showGrid) {
        world.setGridVisible(s.showGrid);
      }
    });
  }, [world]);

  // Keyboard: 5 toggles persp/ortho (tool keys live in useViewportTools).
  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      const t = ev.target as HTMLElement | null;
      if (
        t &&
        (t.tagName === "INPUT" ||
          t.tagName === "TEXTAREA" ||
          t.isContentEditable)
      ) {
        return;
      }
      const s = getViewportState();
      if (ev.key === "5") {
        s.setOrtho(!s.ortho);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const ortho = useViewportStore((s) => s.ortho);
  const showGrid = useViewportStore((s) => s.showGrid);
  const showLabels = useViewportStore((s) => s.showLabels);
  const showTrails = useViewportStore((s) => s.showTrails);
  const setOrtho = useViewportStore((s) => s.setOrtho);
  const toggleOverlay = useViewportStore((s) => s.toggleOverlay);

  return (
    <div
      className={`ss-viewport${calibrating ? " is-calibrating" : ""}`}
      data-scene-id={sceneId}
    >
      <div className="ss-viewport-bar" role="toolbar" aria-label="Viewport">
        {onOpenScene && (
          <button
            type="button"
            className="ss-viewport-bar-btn"
            onClick={onOpenScene}
            title="Open scene gallery (⌘O)"
          >
            <i className="bi bi-folder2-open" aria-hidden="true" />
            <span>Open</span>
          </button>
        )}
        {onOpenLibrary && (
          <button
            type="button"
            className="ss-viewport-bar-btn"
            onClick={onOpenLibrary}
            title="Object library (B)"
          >
            <i className="bi bi-collection" aria-hidden="true" />
            <span>Library</span>
          </button>
        )}
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
        <span className="ss-viewport-bar-label">View</span>
        <div className="seg" role="group" aria-label="Projection">
          <button
            type="button"
            className={!ortho ? "on" : ""}
            onClick={() => setOrtho(false)}
            title="Perspective (5)"
          >
            Persp
          </button>
          <button
            type="button"
            className={ortho ? "on" : ""}
            onClick={() => setOrtho(true)}
            title="Orthographic (5)"
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
        <span className="ss-viewport-beta">3D beta</span>
      </div>
      {calibrating ? (
        <CalibrateWizard
          sceneId={sceneId}
          authToken={authToken}
          cameras={cameras}
        />
      ) : (
        <ToolHeader />
      )}
      <div className={calibrating ? "ss-viewport-split" : "ss-viewport-main"}>
        <div className={calibrating ? "ss-viewport-split-3d" : "ss-viewport-main-3d"}>
          <div className="ss-viewport-host" ref={hostRef}>
            {webglFailed && (
              <p className="ss-viewport-error">
                WebGL is not available in this browser.
              </p>
            )}
          </div>
          <div
            className="ss-viewport-labels"
            ref={labelLayerRef}
            aria-hidden="true"
          />
        </div>
        {calibrating ? <CalibrateFeedPane /> : null}
      </div>
      {!calibrating && cameras.length > 0 && (
        <CameraStripOverlay
          cameras={cameras}
          cameraRates={cameraRates}
          authToken={authToken}
          isSuperuser={isSuperuser}
        />
      )}
    </div>
  );
}
