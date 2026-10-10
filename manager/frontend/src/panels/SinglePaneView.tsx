// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useState } from "react";
import { LibraryDrawer } from "../library/LibraryDrawer";
import { MarkEditor } from "../library/MarkEditor";
import type {
  SceneCameraBootstrap,
  SceneSensorBootstrap,
} from "../scene/types";
import { SceneViewport } from "../viewport/SceneViewport";
import { OpenSceneDialog, useOpenSceneHotkey } from "./OpenSceneDialog";
import { Outliner } from "./Outliner";
import { Properties } from "./Properties";
import { Telemetry } from "./Telemetry";
import { ReplayPanel } from "../replay/ReplayPanel";
import "./Outliner.css";
import "./Properties.css";
import "./SinglePaneView.css";

type DockTab = "outliner" | "properties" | "telemetry";

const DOCK_TABS: { id: DockTab; label: string }[] = [
  { id: "outliner", label: "Outliner" },
  { id: "properties", label: "Properties" },
  { id: "telemetry", label: "Telemetry" },
];

const DOCK_COLLAPSED_KEY = "ss-sp-dock-collapsed";

function readDockCollapsed(): boolean {
  try {
    return sessionStorage.getItem(DOCK_COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

function writeDockCollapsed(collapsed: boolean): void {
  try {
    sessionStorage.setItem(DOCK_COLLAPSED_KEY, collapsed ? "1" : "0");
  } catch {
    /* private mode / quota */
  }
}

export interface SinglePaneViewProps {
  sceneId: string;
  sceneName: string;
  mapUrl?: string | null;
  mapScale?: number | null;
  cameras: SceneCameraBootstrap[];
  sensors: SceneSensorBootstrap[];
  cameraRates?: Record<string, string>;
  authToken: string;
  assetMarkColors?: Record<string, string>;
  isSuperuser?: boolean;
}

/**
 * Single-pane 3D workspace (Phase 1.3): the 3D viewport fills the window with
 * a docked right panel (Outliner / Properties / Telemetry tabs), a left
 * Object Library drawer (B), and the ⌘O Open Scene gallery dialog.
 *
 * Phase 1.4 makes this the scene-detail view; the 2D/3D toggle above it is
 * temporary.
 */
export function SinglePaneView({
  sceneId,
  sceneName,
  mapUrl = null,
  mapScale = null,
  cameras,
  sensors,
  cameraRates = {},
  authToken,
  assetMarkColors,
  isSuperuser = false,
}: SinglePaneViewProps) {
  const [dockTab, setDockTab] = useState<DockTab>("outliner");
  const [dockCollapsed, setDockCollapsed] = useState(readDockCollapsed);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [libraryRefresh, setLibraryRefresh] = useState(0);
  // undefined = editor closed; null = create-new form; string = edit asset.
  const [editorAsset, setEditorAsset] = useState<string | null | undefined>(
    undefined,
  );
  const [sceneDialogOpen, setSceneDialogOpen] = useState(false);
  const [replayMode, setReplayMode] = useState(false);

  useOpenSceneHotkey(() => setSceneDialogOpen(true));

  const setCollapsed = useCallback((collapsed: boolean) => {
    setDockCollapsed(collapsed);
    writeDockCollapsed(collapsed);
  }, []);

  const pickTab = useCallback(
    (id: DockTab) => {
      setDockTab(id);
      if (dockCollapsed) {
        setCollapsed(false);
      }
    },
    [dockCollapsed, setCollapsed],
  );

  return (
    <div className="ss-single-pane">
      <div className="ss-sp-viewport">
        {replayMode ? (
          <ReplayPanel sceneId={sceneId} onClose={() => setReplayMode(false)} />
        ) : (
          <SceneViewport
            sceneId={sceneId}
            sceneName={sceneName}
            mapUrl={mapUrl}
            mapScale={mapScale}
            assetMarkColors={assetMarkColors}
            cameras={cameras}
            sensors={sensors}
            cameraRates={cameraRates}
            authToken={authToken}
            isSuperuser={isSuperuser}
            onOpenLibrary={() => setLibraryOpen(true)}
            onOpenScene={() => setSceneDialogOpen(true)}
            mode="live"
            onModeChange={(m) => setReplayMode(m === "replay")}
          />
        )}
      </div>

      <aside
        className={`ss-sp-dock${dockCollapsed ? " is-collapsed" : ""}`}
        aria-label="Scene panels"
      >
        <div className="ss-sp-dock-rail">
          <div className="ss-sp-dock-rail-inner">
            <button
              type="button"
              className="ss-sp-dock-toggle"
              aria-expanded={!dockCollapsed}
              aria-controls="ss-sp-dock-main"
              title={dockCollapsed ? "Expand panel" : "Collapse panel"}
              onClick={() => setCollapsed(!dockCollapsed)}
            >
              <i
                className={`bi ${dockCollapsed ? "bi-chevron-left" : "bi-chevron-right"}`}
                aria-hidden="true"
              />
              <span className="sr-only">
                {dockCollapsed ? "Expand panel" : "Collapse panel"}
              </span>
            </button>
            {dockCollapsed ? (
              <div
                className="ss-sp-dock-rail-tabs"
                role="tablist"
                aria-label="Outliner, Properties, Telemetry"
                aria-orientation="vertical"
              >
                {DOCK_TABS.map((t) => (
                  <button
                    key={t.id}
                    type="button"
                    role="tab"
                    aria-selected={dockTab === t.id}
                    className={`ss-sp-dock-rail-tab${dockTab === t.id ? " is-active" : ""}`}
                    onClick={() => pickTab(t.id)}
                    title={t.label}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        </div>
        <div id="ss-sp-dock-main" className="ss-sp-dock-main">
          <div
            className="ss-sp-dock-tabs"
            role="tablist"
            aria-label="Outliner, Properties, Telemetry"
          >
            {DOCK_TABS.map((t) => (
              <button
                key={t.id}
                type="button"
                role="tab"
                aria-selected={dockTab === t.id}
                className={`ss-sp-dock-tab${dockTab === t.id ? " is-active" : ""}`}
                onClick={() => pickTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </div>
          <div className="ss-sp-dock-body">
            {dockTab === "outliner" && <Outliner isSuperuser={isSuperuser} />}
            {dockTab === "properties" && (
              <Properties
                sceneName={sceneName}
                sceneId={sceneId}
                cameras={cameras}
                sensors={sensors}
                authToken={authToken}
                isSuperuser={isSuperuser}
              />
            )}
            {dockTab === "telemetry" && <Telemetry sceneId={sceneId} />}
          </div>
        </div>
      </aside>

      <LibraryDrawer
        open={libraryOpen}
        onClose={() => setLibraryOpen(false)}
        onOpen={() => setLibraryOpen(true)}
        authToken={authToken}
        onSelectAsset={(uid) => {
          setEditorAsset(uid);
          setLibraryOpen(false);
        }}
        refreshToken={libraryRefresh}
      />
      {editorAsset !== undefined && (
        <MarkEditor
          assetId={editorAsset}
          authToken={authToken}
          onClose={() => setEditorAsset(undefined)}
          onSaved={() => {
            setLibraryRefresh((n) => n + 1);
            setEditorAsset(undefined);
          }}
        />
      )}

      <OpenSceneDialog
        open={sceneDialogOpen}
        onClose={() => setSceneDialogOpen(false)}
        onOpenScene={(uid) => {
          setSceneDialogOpen(false);
          window.location.assign(`/${uid}/`);
        }}
        authToken={authToken}
      />
    </div>
  );
}
