// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import { LibraryDrawer } from "../library/LibraryDrawer";
import { MarkEditor } from "../library/MarkEditor";
import { useChildStatusMqtt } from "../mqtt/useChildStatusMqtt";
import type {
  SceneCameraBootstrap,
  SceneChildBootstrap,
  SceneSensorBootstrap,
} from "../scene/types";
import { useCalibrateStore } from "../viewport/calibrate/calibrateStore";
import { useViewportStore } from "../viewport/store";
import { SceneViewport } from "../viewport/SceneViewport";
import { ViewportMenuBar } from "../viewport/ViewportMenuBar";
import {
  ViewportOptionsBar,
  type SceneChromeActions,
} from "../viewport/ViewportOptionsBar";
import {
  OpenAdminListDialog,
  type OpenAdminListKind,
} from "./OpenAdminListDialog";
import { OpenSceneDialog, useOpenSceneHotkey } from "./OpenSceneDialog";
import { LivePreviewPane } from "./LivePreviewPane";
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
  children?: SceneChildBootstrap[];
  onChildrenChange?: Dispatch<SetStateAction<SceneChildBootstrap[]>>;
  cameraRates?: Record<string, string>;
  authToken: string;
  assetMarkColors?: Record<string, string>;
  isSuperuser?: boolean;
  mqttConnected?: boolean | null;
  sceneChrome?: SceneChromeActions | null;
}

/**
 * Single-pane 3D workspace (Phase 1.3): the 3D viewport fills the window with
 * a docked right panel (Outliner / Properties / Telemetry tabs), a left
 * Object Library drawer (B), and the ⌘O Open Scene gallery dialog.
 *
 * Outliner Cameras → Preview splits the dock: active tab on top, Live glance
 * on the bottom half. Maximize fills the dock; close via Outliner Preview.
 */
export function SinglePaneView({
  sceneId,
  sceneName,
  mapUrl = null,
  mapScale = null,
  cameras,
  sensors,
  children = [],
  onChildrenChange,
  cameraRates = {},
  authToken,
  assetMarkColors,
  isSuperuser = false,
  mqttConnected = null,
  sceneChrome = null,
}: SinglePaneViewProps) {
  const [dockTab, setDockTab] = useState<DockTab>("outliner");
  const [dockCollapsed, setDockCollapsed] = useState(readDockCollapsed);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewMaximized, setPreviewMaximized] = useState(false);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [libraryRefresh, setLibraryRefresh] = useState(0);
  const remoteChildIds = useMemo(
    () =>
      children
        .filter((c) => c.childType === "remote" && c.remoteChildId)
        .map((c) => String(c.remoteChildId)),
    [children],
  );
  useChildStatusMqtt(remoteChildIds);
  // undefined = editor closed; null = create-new form; string = edit asset.
  const [editorAsset, setEditorAsset] = useState<string | null | undefined>(
    undefined,
  );
  const [sceneDialogOpen, setSceneDialogOpen] = useState(false);
  const [adminListKind, setAdminListKind] = useState<OpenAdminListKind | null>(
    null,
  );
  const [adminListRefresh, setAdminListRefresh] = useState(0);
  const [replayMode, setReplayMode] = useState(false);
  const calibrating = useCalibrateStore((s) => s.active);
  const projectCameraIds = useViewportStore((s) => s.projectCameraIds);

  useOpenSceneHotkey(() => {
    setAdminListKind(null);
    setSceneDialogOpen(true);
  });

  const setCollapsed = useCallback((collapsed: boolean) => {
    setDockCollapsed(collapsed);
    writeDockCollapsed(collapsed);
  }, []);

  const closePreview = useCallback(() => {
    setPreviewOpen(false);
    setPreviewMaximized(false);
  }, []);

  const setPreview = useCallback(
    (open: boolean) => {
      setPreviewOpen(open);
      if (open) {
        setCollapsed(false);
      } else {
        setPreviewMaximized(false);
      }
    },
    [setCollapsed],
  );

  // Projection needs JPEG frames — open Preview (and Live View) with it.
  useEffect(() => {
    if (projectCameraIds.length === 0 || cameras.length === 0) {
      return;
    }
    setPreviewOpen(true);
    setCollapsed(false);
  }, [projectCameraIds, cameras.length, setCollapsed]);

  // Drop Preview if the scene loses all cameras.
  useEffect(() => {
    if (cameras.length === 0 && previewOpen) {
      closePreview();
    }
  }, [cameras.length, previewOpen, closePreview]);

  const pickTab = useCallback(
    (id: DockTab) => {
      setDockTab(id);
      if (dockCollapsed) {
        setCollapsed(false);
      }
    },
    [dockCollapsed, setCollapsed],
  );

  const openSceneDialog = useCallback(() => {
    setAdminListKind(null);
    setSceneDialogOpen(true);
  }, []);
  const openCameras = useCallback(() => {
    setSceneDialogOpen(false);
    setAdminListKind("cameras");
  }, []);
  const openSensors = useCallback(() => {
    setSceneDialogOpen(false);
    setAdminListKind("sensors");
  }, []);

  return (
    <div className="ss-single-pane">
      <div className="ss-sp-chrome">
        <ViewportMenuBar
          onOpenScene={openSceneDialog}
          onOpenCameras={openCameras}
          onOpenSensors={openSensors}
          onOpenLibrary={() => setLibraryOpen(true)}
          mqttConnected={mqttConnected}
        />
        <ViewportOptionsBar
          emptyScene={false}
          calibrating={calibrating && !replayMode}
          mode={replayMode ? "replay" : "live"}
          onModeChange={(m) => setReplayMode(m === "replay")}
          sceneChrome={sceneChrome}
        />
      </div>
      <div className="ss-sp-body">
        <div className="ss-sp-viewport">
          {replayMode ? (
            <ReplayPanel
              sceneId={sceneId}
              onClose={() => setReplayMode(false)}
            />
          ) : (
            <SceneViewport
              sceneId={sceneId}
              sceneName={sceneName}
              mapUrl={mapUrl}
              mapScale={mapScale}
              assetMarkColors={assetMarkColors}
              cameras={cameras}
              sensors={sensors}
              children={children}
              authToken={authToken}
              showTopChrome={false}
              onOpenLibrary={() => setLibraryOpen(true)}
              onOpenScene={openSceneDialog}
              onOpenCameras={openCameras}
              onOpenSensors={openSensors}
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
          <div
            id="ss-sp-dock-main"
            className={[
              "ss-sp-dock-main",
              previewOpen ? "is-preview-split" : "",
              previewOpen && previewMaximized ? "is-preview-maximized" : "",
            ]
              .filter(Boolean)
              .join(" ")}
          >
            <div className="ss-sp-dock-tabs-pane">
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
                {dockTab === "outliner" && (
                  <Outliner
                    isSuperuser={isSuperuser}
                    showPreviewToggle={cameras.length > 0}
                    previewOpen={previewOpen}
                    onPreviewOpenChange={setPreview}
                  />
                )}
                {dockTab === "properties" && (
                  <Properties
                    sceneName={sceneName}
                    sceneId={sceneId}
                    cameras={cameras}
                    sensors={sensors}
                    children={children}
                    onChildrenChange={onChildrenChange}
                    authToken={authToken}
                    isSuperuser={isSuperuser}
                  />
                )}
                {dockTab === "telemetry" && <Telemetry sceneId={sceneId} />}
              </div>
            </div>
            {previewOpen && cameras.length > 0 ? (
              <div className="ss-sp-dock-glance">
                <LivePreviewPane
                  cameras={cameras}
                  cameraRates={cameraRates}
                  authToken={authToken}
                  isSuperuser={isSuperuser}
                  maximized={previewMaximized}
                  onMaximizedChange={setPreviewMaximized}
                />
              </div>
            ) : null}
          </div>
        </aside>
      </div>

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
        isSuperuser={isSuperuser}
      />
      <OpenAdminListDialog
        open={adminListKind !== null}
        kind={adminListKind || "cameras"}
        onClose={() => {
          setAdminListKind(null);
          setAdminListRefresh((n) => n + 1);
        }}
        refreshToken={adminListRefresh}
      />
    </div>
  );
}
