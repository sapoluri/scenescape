// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useRef, useState } from "react";
import Stats from "three/addons/libs/stats.module.js";
import { useAppToast } from "../components/ToastProvider";
import { saveAllDirtyEntities } from "../lib/entityPersist";
import type {
  SceneCameraBootstrap,
  SceneChildBootstrap,
  SceneSensorBootstrap,
} from "../scene/types";
import { CalibrateFeedPane } from "./calibrate/CalibrateFeedPane";
import { CalibrateWizard } from "./calibrate/CalibrateWizard";
import { useCalibrateStore } from "./calibrate/calibrateStore";
import { useCalibrateSession } from "./calibrate/useCalibrateSession";
import { useViewportEntities } from "./entities";
import { useViewportMarks } from "./marks";
import { countDirtyPersistable, usePersistStore } from "./persistStore";
import { attachMarkRenderer } from "./renderMarks";
import { useLiveCameraProjection } from "./useLiveCameraProjection";
import { useSceneMap } from "./useSceneMap";
import { getViewportState, useViewportStore } from "./store";
import { TOOLS, toolDef } from "./tools";
import type { ToolId } from "./types";
import { useViewportTools } from "./useViewportTools";
import {
  ViewportMenuBar,
  ViewportPoweredBy,
} from "./ViewportMenuBar";
import {
  ViewportOptionsBar,
  type SceneChromeActions,
} from "./ViewportOptionsBar";
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
  /** Omit or empty for the home empty-scene shell (orbit only). */
  sceneId?: string | null;
  sceneName?: string;
  mapUrl?: string | null;
  mapScale?: number | null;
  assetMarkColors?: Record<string, string>;
  cameras?: SceneCameraBootstrap[];
  sensors?: SceneSensorBootstrap[];
  children?: SceneChildBootstrap[];
  authToken: string;
  onOpenLibrary?: () => void;
  onOpenScene?: () => void;
  onOpenCameras?: () => void;
  onOpenSensors?: () => void;
  mode?: "live" | "replay";
  onModeChange?: (mode: "live" | "replay") => void;
  /** Scene-agnostic MQTT pill on the app menu bar. */
  mqttConnected?: boolean | null;
  /** Scene name / export / edit / delete / rate on the options bar. */
  sceneChrome?: SceneChromeActions | null;
  /**
   * When false, parent owns ViewportMenuBar + ViewportOptionsBar (full-bleed
   * chrome above the Outliner dock). Default true for home / standalone.
   */
  showTopChrome?: boolean;
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

function ToolHeader({
  sceneId,
  authToken,
  cameras,
  sensors,
}: {
  sceneId: string;
  authToken: string;
  cameras: SceneCameraBootstrap[];
  sensors: SceneSensorBootstrap[];
}) {
  const toast = useAppToast();
  const activeTool = useViewportStore((s) => s.activeTool);
  const setTool = useViewportStore((s) => s.setTool);
  const canUndo = useViewportStore((s) => s.past.length > 0);
  const canRedo = useViewportStore((s) => s.future.length > 0);
  const undo = useViewportStore((s) => s.undo);
  const redo = useViewportStore((s) => s.redo);
  const entities = useViewportStore((s) => s.entities);
  const baselines = usePersistStore((s) => s.baselines);
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

  const entityDirtyCount = useMemo(
    () => countDirtyPersistable(),
    [entities, baselines],
  );
  const geometryDirty = roiDirty || tripDirty;
  const dirtyCount = entityDirtyCount + (geometryDirty ? 1 : 0);
  const dirty = dirtyCount > 0;

  const onSaveAll = async () => {
    if (saving || !dirty || !authToken || !sceneId) {
      return;
    }
    setSaving(true);
    try {
      const result = await saveAllDirtyEntities({
        authToken,
        sceneId,
        cameras,
        sensors,
        geometryDirty,
      });
      if (result.saved > 0 && !result.errors.length) {
        toast.show(
          result.saved === 1
            ? "Saved 1 change"
            : `Saved ${result.saved} changes`,
          "ok",
        );
      } else if (result.saved > 0 && result.errors.length) {
        toast.show(
          `Saved ${result.saved}; ${result.errors.length} failed`,
          "bad",
        );
      } else if (result.errors.length) {
        toast.show(result.errors[0] || "Save failed", "bad");
      }
      setRoiDirty(false);
      setTripDirty(false);
    } finally {
      setSaving(false);
    }
  };

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
      if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "s") {
        ev.preventDefault();
        void onSaveAll();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dirty, saving, authToken, sceneId, cameras, sensors, geometryDirty]);

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
            title={`${t.label} (${t.shortcut}) ? ${t.hint}`}
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
        disabled={!dirty || saving || !authToken}
        onClick={() => void onSaveAll()}
        title={
          dirty
            ? `Save all unsaved cameras, sensors, regions, and tripwires (${entityDirtyCount} item${entityDirtyCount === 1 ? "" : "s"}${geometryDirty ? " + geometry" : ""}) ? Ctrl+S`
            : "No unsaved changes"
        }
      >
        <i className="bi bi-check-lg" aria-hidden="true" />
        <span className="ss-tool-label">
          {saving
            ? "Saving?"
            : dirty
              ? `Save all${entityDirtyCount > 0 ? ` (${entityDirtyCount})` : ""}`
              : "Save all"}
        </span>
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
  sceneId = null,
  sceneName = "",
  mapUrl = null,
  mapScale = null,
  assetMarkColors,
  cameras = [],
  sensors = [],
  children = [],
  authToken,
  onOpenLibrary,
  onOpenScene,
  onOpenCameras,
  onOpenSensors,
  mode = "live",
  onModeChange,
  mqttConnected = null,
  sceneChrome = null,
  showTopChrome = true,
}: SceneViewportProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const labelLayerRef = useRef<HTMLDivElement>(null);
  const statsHostRef = useRef<HTMLDivElement>(null);
  const [world, setWorld] = useState<ViewportWorld | null>(null);
  const [webglFailed, setWebglFailed] = useState(false);
  const theme = useTheme();
  const emptyScene = !sceneId;

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
    const detachMarks = attachMarkRenderer(w, {
      // Token is stable for a scene session; world rebuilds on sceneId change.
      authToken: sceneId ? authToken : "",
    });
    return () => {
      detachMarks();
      setWorld(null);
      w.dispose();
    };
    // authToken is read at scene open; remounting the WebGL world on token
    // churn would flash the viewport.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sceneId]);

  // three.js Stats (FPS) ? bottom-left, same placement as the legacy 3D UI.
  useEffect(() => {
    if (!world || !statsHostRef.current) {
      return;
    }
    const stats = new Stats();
    stats.showPanel(0);
    statsHostRef.current.replaceChildren(stats.dom);
    const onFrame = () => stats.update();
    world.onFrame.add(onFrame);
    return () => {
      world.onFrame.delete(onFrame);
      statsHostRef.current?.replaceChildren();
    };
  }, [world]);

  useEffect(() => {
    world?.setTheme(theme);
  }, [world, theme]);

  useSceneMap(world, {
    sceneId: sceneId || "",
    sceneName,
    mapUrl,
    mapScale,
    authToken,
  });
  useViewportEntities(world, {
    cameras: emptyScene ? [] : cameras,
    sensors: emptyScene ? [] : sensors,
    children: emptyScene ? [] : children,
    authToken: emptyScene ? "" : authToken,
  });
  useViewportTools(world);
  useCalibrateSession(world, emptyScene ? "" : authToken);
  useLiveCameraProjection(world, emptyScene ? [] : cameras);

  const calibrating = useCalibrateStore((s) => s.active) && !emptyScene;

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

  return (
    <div
      className={`ss-viewport${calibrating ? " is-calibrating" : ""}${emptyScene ? " is-empty" : ""}`}
      data-scene-id={sceneId || ""}
    >
      {showTopChrome ? (
        <>
          <ViewportMenuBar
            onOpenScene={onOpenScene}
            onOpenCameras={onOpenCameras}
            onOpenSensors={onOpenSensors}
            onOpenLibrary={onOpenLibrary}
            mqttConnected={mqttConnected}
          />
          <ViewportOptionsBar
            emptyScene={emptyScene}
            calibrating={calibrating}
            mode={mode}
            onModeChange={onModeChange}
            sceneChrome={emptyScene ? null : sceneChrome}
          />
        </>
      ) : null}
      {!emptyScene &&
        (calibrating ? (
          <CalibrateWizard
            sceneId={sceneId || ""}
            authToken={authToken}
            cameras={cameras}
          />
        ) : (
          <ToolHeader
            sceneId={sceneId || ""}
            authToken={authToken}
            cameras={cameras}
            sensors={sensors}
          />
        ))}
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
          <div
            className="ss-viewport-stats"
            ref={statsHostRef}
            aria-hidden="true"
          />
          <ViewportPoweredBy />
        </div>
        {calibrating ? <CalibrateFeedPane /> : null}
      </div>
    </div>
  );
}
