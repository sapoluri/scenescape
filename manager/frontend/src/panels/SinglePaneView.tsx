// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
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
import { getViewportState, useViewportStore } from "../viewport/store";
import { useVisualizeStore } from "../viewport/visualize";
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
import { LivePreviewPane, ProjectionFeedHost } from "./LivePreviewPane";
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
/** Pixel height when the user has shrunk Properties; null = size to content. */
const PROPS_HEIGHT_KEY = "ss-sp-props-height-px";
/** Pixel height for the Preview overlay; null = CSS default fraction. */
const PREVIEW_HEIGHT_KEY = "ss-sp-preview-height-px";

function readHeightPx(key: string): number | null {
  try {
    const n = Number(sessionStorage.getItem(key));
    if (Number.isFinite(n) && n >= 96) {
      return n;
    }
  } catch {
    /* private mode / quota */
  }
  return null;
}

function writeHeightPx(key: string, px: number | null): void {
  try {
    if (px == null) {
      sessionStorage.removeItem(key);
    } else {
      sessionStorage.setItem(key, String(Math.round(px)));
    }
  } catch {
    /* private mode / quota */
  }
}

function readPropsHeightPx(): number | null {
  return readHeightPx(PROPS_HEIGHT_KEY);
}

function writePropsHeightPx(px: number | null): void {
  writeHeightPx(PROPS_HEIGHT_KEY, px);
}

function readPreviewHeightPx(): number | null {
  return readHeightPx(PREVIEW_HEIGHT_KEY);
}

function writePreviewHeightPx(px: number | null): void {
  writeHeightPx(PREVIEW_HEIGHT_KEY, px);
}

function clampOverlayHeightPx(px: number, hostHeight: number): number {
  const max = Math.max(96, hostHeight * 0.85);
  return Math.min(max, Math.max(96, px));
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
  onCamerasChange?: Dispatch<SetStateAction<SceneCameraBootstrap[]>>;
  onSensorsChange?: Dispatch<SetStateAction<SceneSensorBootstrap[]>>;
  onChildrenChange?: Dispatch<SetStateAction<SceneChildBootstrap[]>>;
  cameraRates?: Record<string, string>;
  authToken: string;
  assetMarkColors?: Record<string, string>;
  isSuperuser?: boolean;
  mqttConnected?: boolean | null;
  /** Bootstrap WSS broker URL for Telemetry connect details. */
  wssConnection?: string;
  sceneChrome?: SceneChromeActions | null;
  /** Bump when a camera/sensor is created via sheet so the Open gallery reloads. */
  directoryRefreshToken?: number;
}

/**
 * Single-pane 3D workspace (Phase 1.3): the 3D viewport fills the window with
 * a docked right panel (Outliner / Telemetry), a left Object Library drawer
 * (B), and the ⌘O Open Scene gallery dialog.
 *
 * Selecting an entity opens Properties as a content-sized overlay on top of
 * the Outliner (drag the top handle to shrink). Outliner Cameras → Preview
 * opens the live glance instead — Properties and Preview are mutually
 * exclusive. Maximize fills the dock; close Preview via the Outliner toggle.
 */
export function SinglePaneView({
  sceneId,
  sceneName,
  mapUrl = null,
  mapScale = null,
  cameras,
  sensors,
  children = [],
  onCamerasChange,
  onSensorsChange,
  onChildrenChange,
  cameraRates = {},
  authToken,
  assetMarkColors,
  isSuperuser = false,
  mqttConnected = null,
  wssConnection = "",
  sceneChrome = null,
  directoryRefreshToken = 0,
}: SinglePaneViewProps) {
  const [dockTab, setDockTab] = useState<DockTab>("outliner");
  const [dockCollapsed, setDockCollapsed] = useState(readDockCollapsed);
  const [propsHeightPx, setPropsHeightPx] = useState<number | null>(
    readPropsHeightPx,
  );
  const propsHeightRef = useRef(propsHeightPx);
  propsHeightRef.current = propsHeightPx;
  const [previewHeightPx, setPreviewHeightPx] = useState<number | null>(
    readPreviewHeightPx,
  );
  const previewHeightRef = useRef(previewHeightPx);
  previewHeightRef.current = previewHeightPx;
  const editorRef = useRef<HTMLDivElement>(null);
  const dockMainRef = useRef<HTMLDivElement>(null);
  const splitDrag = useRef<{ pointerId: number; kind: "props" | "preview" } | null>(
    null,
  );
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
  // Parent token (sheet create/edit) + local bumps (dialog close) both reload.
  const directoryRefresh = directoryRefreshToken + adminListRefresh;
  const [replayMode, setReplayMode] = useState(false);
  const calibrating = useCalibrateStore((s) => s.active);
  const visualizing = useVisualizeStore((s) => s.active);
  const projectCameraIds = useViewportStore((s) => s.projectCameraIds);
  const selectedId = useViewportStore((s) => s.selectedId);
  const editFocus = useViewportStore((s) => s.editFocus);
  // Properties and Preview share the dock bottom — never both at once.
  // Live visualize keeps Preview open and suppresses Properties.
  const propertiesOpen =
    selectedId != null && !previewOpen && !visualizing;

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
    // Creating / focusing an entity prefers Properties over Preview.
    setPreviewOpen(false);
    setPreviewMaximized(false);
  }, [editFocus, setCollapsed]);

  // Selecting an entity closes Preview so Properties can open — unless
  // Live visualize is active (thumbnail click selects + opens the split).
  useEffect(() => {
    if (selectedId == null || useVisualizeStore.getState().active) {
      return;
    }
    setPreviewOpen(false);
    setPreviewMaximized(false);
  }, [selectedId]);

  const applyPropsHeight = useCallback((clientY: number) => {
    const editor = editorRef.current;
    if (!editor) {
      return;
    }
    const rect = editor.getBoundingClientRect();
    const px = clampOverlayHeightPx(rect.bottom - clientY, rect.height);
    propsHeightRef.current = px;
    setPropsHeightPx(px);
  }, []);

  const applyPreviewHeight = useCallback((clientY: number) => {
    const main = dockMainRef.current;
    if (!main) {
      return;
    }
    const rect = main.getBoundingClientRect();
    const px = clampOverlayHeightPx(rect.bottom - clientY, rect.height);
    previewHeightRef.current = px;
    setPreviewHeightPx(px);
  }, []);

  const onPropsSplitPointerDown = (ev: ReactPointerEvent<HTMLDivElement>) => {
    if (ev.button !== 0) {
      return;
    }
    ev.preventDefault();
    ev.currentTarget.setPointerCapture(ev.pointerId);
    splitDrag.current = { pointerId: ev.pointerId, kind: "props" };
    applyPropsHeight(ev.clientY);
  };

  const onPreviewSplitPointerDown = (ev: ReactPointerEvent<HTMLDivElement>) => {
    if (ev.button !== 0) {
      return;
    }
    ev.preventDefault();
    ev.currentTarget.setPointerCapture(ev.pointerId);
    splitDrag.current = { pointerId: ev.pointerId, kind: "preview" };
    applyPreviewHeight(ev.clientY);
  };

  const onSplitPointerMove = (ev: ReactPointerEvent<HTMLDivElement>) => {
    if (splitDrag.current?.pointerId !== ev.pointerId) {
      return;
    }
    if (splitDrag.current.kind === "preview") {
      applyPreviewHeight(ev.clientY);
    } else {
      applyPropsHeight(ev.clientY);
    }
  };

  const onSplitPointerUp = (ev: ReactPointerEvent<HTMLDivElement>) => {
    if (splitDrag.current?.pointerId !== ev.pointerId) {
      return;
    }
    const kind = splitDrag.current.kind;
    splitDrag.current = null;
    if (kind === "preview") {
      writePreviewHeightPx(previewHeightRef.current);
    } else {
      writePropsHeightPx(propsHeightRef.current);
    }
  };

  const onPropsSplitKeyDown = (ev: ReactKeyboardEvent<HTMLDivElement>) => {
    if (ev.key !== "ArrowUp" && ev.key !== "ArrowDown") {
      return;
    }
    ev.preventDefault();
    const editor = editorRef.current;
    const editorHeight = editor?.getBoundingClientRect().height ?? 0;
    const current =
      propsHeightRef.current ??
      Math.min(editorHeight * 0.5, editorHeight * 0.85);
    const delta = ev.key === "ArrowUp" ? 24 : -24;
    const px = clampOverlayHeightPx(current + delta, editorHeight);
    propsHeightRef.current = px;
    setPropsHeightPx(px);
    writePropsHeightPx(px);
  };

  const onPreviewSplitKeyDown = (ev: ReactKeyboardEvent<HTMLDivElement>) => {
    if (ev.key !== "ArrowUp" && ev.key !== "ArrowDown") {
      return;
    }
    ev.preventDefault();
    const main = dockMainRef.current;
    const hostHeight = main?.getBoundingClientRect().height ?? 0;
    const current =
      previewHeightRef.current ??
      Math.min(hostHeight * 0.42, hostHeight * 0.85);
    const delta = ev.key === "ArrowUp" ? 24 : -24;
    const px = clampOverlayHeightPx(current + delta, hostHeight);
    previewHeightRef.current = px;
    setPreviewHeightPx(px);
    writePreviewHeightPx(px);
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
        // Clear selection so Properties does not share the dock with Preview.
        getViewportState().select(null);
      } else {
        setPreviewMaximized(false);
      }
    },
    [setCollapsed],
  );

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
          calibrating={(calibrating || visualizing) && !replayMode}
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
              isSuperuser={isSuperuser}
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
            ref={dockMainRef}
            className={[
              "ss-sp-dock-main",
              previewOpen ? "is-preview-split" : "",
              previewOpen && previewMaximized ? "is-preview-maximized" : "",
            ]
              .filter(Boolean)
              .join(" ")}
          >
            <div className="ss-sp-dock-editor" ref={editorRef}>
              <div className="ss-sp-dock-tabs-pane">
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
                  {/* Keep both mounted so MQTT event listeners stay live. */}
                  <div
                    className="ss-sp-dock-panel"
                    hidden={dockTab !== "outliner"}
                  >
                    <Outliner
                      isSuperuser={isSuperuser}
                      authToken={authToken}
                      cameras={cameras}
                      sensors={sensors}
                      children={children}
                      onCamerasChange={onCamerasChange}
                      onSensorsChange={onSensorsChange}
                      onChildrenChange={onChildrenChange}
                      showPreviewToggle={cameras.length > 0}
                      previewOpen={previewOpen}
                      onPreviewOpenChange={setPreview}
                    />
                  </div>
                  <div
                    className="ss-sp-dock-panel"
                    hidden={dockTab !== "telemetry"}
                  >
                    <Telemetry
                      sceneId={sceneId}
                      wssConnection={wssConnection}
                      sensors={sensors}
                    />
                  </div>
                </div>
              </div>
              {propertiesOpen ? (
                <div
                  className={`ss-sp-dock-props${propsHeightPx != null ? " is-resized" : ""}`}
                  style={
                    propsHeightPx != null
                      ? ({
                          "--ss-props-height": `${propsHeightPx}px`,
                        } as CSSProperties)
                      : undefined
                  }
                >
                  <div
                    className="ss-sp-dock-split"
                    role="separator"
                    aria-orientation="horizontal"
                    aria-label="Resize properties panel"
                    aria-valuemin={96}
                    aria-valuemax={85}
                    tabIndex={0}
                    onPointerDown={onPropsSplitPointerDown}
                    onPointerMove={onSplitPointerMove}
                    onPointerUp={onSplitPointerUp}
                    onPointerCancel={onSplitPointerUp}
                    onKeyDown={onPropsSplitKeyDown}
                  />
                  <div className="ss-sp-dock-props-head">
                    <span className="ss-sp-dock-props-title">Properties</span>
                    <button
                      type="button"
                      className="ss-sp-dock-props-close"
                      title="Close properties (Esc)"
                      aria-label="Close properties"
                      onClick={() => getViewportState().select(null)}
                    >
                      <i className="bi bi-x" aria-hidden="true" />
                    </button>
                  </div>
                  <Properties
                    sceneName={sceneName}
                    sceneId={sceneId}
                    cameras={cameras}
                    sensors={sensors}
                    children={children}
                    onCamerasChange={onCamerasChange}
                    onSensorsChange={onSensorsChange}
                    authToken={authToken}
                    isSuperuser={isSuperuser}
                  />
                </div>
              ) : null}
            </div>
            {previewOpen && cameras.length > 0 ? (
              <div
                className={[
                  "ss-sp-dock-glance",
                  !previewMaximized && previewHeightPx != null
                    ? "is-resized"
                    : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
                style={
                  !previewMaximized && previewHeightPx != null
                    ? ({
                        "--ss-preview-height": `${previewHeightPx}px`,
                      } as CSSProperties)
                    : undefined
                }
              >
                {!previewMaximized ? (
                  <div
                    className="ss-sp-dock-split"
                    role="separator"
                    aria-orientation="horizontal"
                    aria-label="Resize camera preview"
                    aria-valuemin={96}
                    aria-valuemax={85}
                    tabIndex={0}
                    onPointerDown={onPreviewSplitPointerDown}
                    onPointerMove={onSplitPointerMove}
                    onPointerUp={onSplitPointerUp}
                    onPointerCancel={onSplitPointerUp}
                    onKeyDown={onPreviewSplitKeyDown}
                  />
                ) : null}
                <LivePreviewPane
                  cameras={cameras}
                  cameraRates={cameraRates}
                  maximized={previewMaximized}
                  onMaximizedChange={setPreviewMaximized}
                />
              </div>
            ) : null}
            {/* JPEG hosts for Project frame when Preview pane is closed. */}
            <ProjectionFeedHost
              cameras={cameras}
              active={
                !previewOpen &&
                projectCameraIds.length > 0 &&
                cameras.length > 0
              }
            />
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
        refreshToken={directoryRefresh}
      />
    </div>
  );
}
