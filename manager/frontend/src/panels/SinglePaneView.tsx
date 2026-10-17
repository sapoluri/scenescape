// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
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

type DockTab = "outliner" | "telemetry";

const DOCK_TABS: { id: DockTab; label: string }[] = [
  { id: "outliner", label: "Outliner" },
  { id: "telemetry", label: "Telemetry" },
];

const DOCK_COLLAPSED_KEY = "ss-sp-dock-collapsed";
const PROPS_RATIO_KEY = "ss-sp-props-ratio";
const PROPS_RATIO_DEFAULT = 0.42;

function readPropsRatio(): number {
  try {
    const n = Number(sessionStorage.getItem(PROPS_RATIO_KEY));
    if (Number.isFinite(n) && n > 0.15 && n < 0.85) {
      return n;
    }
  } catch {
    /* private mode / quota */
  }
  return PROPS_RATIO_DEFAULT;
}

function writePropsRatio(ratio: number): void {
  try {
    sessionStorage.setItem(PROPS_RATIO_KEY, String(ratio));
  } catch {
    /* private mode / quota */
  }
}

function clampPropsRatio(ratio: number, height: number): number {
  if (height <= 0) {
    return Math.min(0.85, Math.max(0.15, ratio));
  }
  const min = Math.min(0.4, 96 / height);
  return Math.min(1 - min, Math.max(min, ratio));
}

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
 * a docked right panel (Outliner / Telemetry), a left Object Library drawer
 * (B), and the ⌘O Open Scene gallery dialog.
 *
 * Selecting an entity opens Properties as a bottom split of the dock, the
 * same way Preview opens the live glance. The split between Outliner and
 * Properties can be dragged. Outliner Cameras → Preview adds the glance
 * under that stack. Maximize fills the dock; close via Outliner Preview.
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
  const [propsRatio, setPropsRatio] = useState(readPropsRatio);
  const propsRatioRef = useRef(propsRatio);
  propsRatioRef.current = propsRatio;
  const editorRef = useRef<HTMLDivElement>(null);
  const splitDrag = useRef<{ pointerId: number } | null>(null);
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
  const selectedId = useViewportStore((s) => s.selectedId);
  const editFocus = useViewportStore((s) => s.editFocus);
  const propertiesOpen = selectedId != null;

  useOpenSceneHotkey(() => {
    setAdminListKind(null);
    setSceneDialogOpen(true);
  });

  const setCollapsed = useCallback((collapsed: boolean) => {
    setDockCollapsed(collapsed);
    writeDockCollapsed(collapsed);
  }, []);

  useEffect(() => {
    if (editFocus < 1) {
      return;
    }
    setDockTab("outliner");
    setCollapsed(false);
  }, [editFocus, setCollapsed]);

  const applySplitRatio = useCallback((clientY: number) => {
    const editor = editorRef.current;
    if (!editor) {
      return;
    }
    const rect = editor.getBoundingClientRect();
    const ratio = clampPropsRatio(
      (rect.bottom - clientY) / rect.height,
      rect.height,
    );
    propsRatioRef.current = ratio;
    setPropsRatio(ratio);
  }, []);

  const onSplitPointerDown = (ev: ReactPointerEvent<HTMLDivElement>) => {
    if (ev.button !== 0) {
      return;
    }
    ev.preventDefault();
    ev.currentTarget.setPointerCapture(ev.pointerId);
    splitDrag.current = { pointerId: ev.pointerId };
    applySplitRatio(ev.clientY);
  };

  const onSplitPointerMove = (ev: ReactPointerEvent<HTMLDivElement>) => {
    if (splitDrag.current?.pointerId !== ev.pointerId) {
      return;
    }
    applySplitRatio(ev.clientY);
  };

  const onSplitPointerUp = (ev: ReactPointerEvent<HTMLDivElement>) => {
    if (splitDrag.current?.pointerId !== ev.pointerId) {
      return;
    }
    splitDrag.current = null;
    writePropsRatio(propsRatioRef.current);
  };

  const onSplitKeyDown = (ev: ReactKeyboardEvent<HTMLDivElement>) => {
    if (ev.key !== "ArrowUp" && ev.key !== "ArrowDown") {
      return;
    }
    ev.preventDefault();
    const editor = editorRef.current;
    const height = editor?.getBoundingClientRect().height ?? 0;
    const delta = ev.key === "ArrowUp" ? 0.04 : -0.04;
    const ratio = clampPropsRatio(propsRatioRef.current + delta, height);
    propsRatioRef.current = ratio;
    setPropsRatio(ratio);
    writePropsRatio(ratio);
  };

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
                  aria-label="Outliner and Telemetry"
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
              propertiesOpen ? "is-props-split" : "",
            ]
              .filter(Boolean)
              .join(" ")}
          >
            <div className="ss-sp-dock-editor" ref={editorRef}>
              <div
                className="ss-sp-dock-tabs-pane"
                style={
                  propertiesOpen && !(previewOpen && previewMaximized)
                    ? { flex: `${1 - propsRatio} 1 0` }
                    : undefined
                }
              >
                <div
                  className="ss-sp-dock-tabs"
                  role="tablist"
                  aria-label="Outliner and Telemetry"
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
                  {dockTab === "telemetry" && <Telemetry sceneId={sceneId} />}
                </div>
              </div>
              {propertiesOpen ? (
                <>
                  <div
                    className="ss-sp-dock-split"
                    role="separator"
                    aria-orientation="horizontal"
                    aria-label="Resize outliner and properties"
                    aria-valuemin={15}
                    aria-valuemax={85}
                    aria-valuenow={Math.round(propsRatio * 100)}
                    tabIndex={0}
                    onPointerDown={onSplitPointerDown}
                    onPointerMove={onSplitPointerMove}
                    onPointerUp={onSplitPointerUp}
                    onPointerCancel={onSplitPointerUp}
                    onKeyDown={onSplitKeyDown}
                  />
                  <div
                    className="ss-sp-dock-props"
                    style={{ flex: `${propsRatio} 1 0` }}
                  >
                    <div className="ss-sp-dock-props-head">Properties</div>
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
                  </div>
                </>
              ) : null}
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
