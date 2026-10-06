// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
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
import "./Outliner.css";
import "./Properties.css";
import "./SinglePaneView.css";

type DockTab = "outliner" | "properties" | "telemetry";

const DOCK_TABS: { id: DockTab; label: string }[] = [
  { id: "outliner", label: "Outliner" },
  { id: "properties", label: "Properties" },
  { id: "telemetry", label: "Telemetry" },
];

export interface SinglePaneViewProps {
  sceneId: string;
  sceneName: string;
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
  cameras,
  sensors,
  cameraRates = {},
  authToken,
  assetMarkColors,
  isSuperuser = false,
}: SinglePaneViewProps) {
  const [dockTab, setDockTab] = useState<DockTab>("outliner");
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [libraryRefresh, setLibraryRefresh] = useState(0);
  // undefined = editor closed; null = create-new form; string = edit asset.
  const [editorAsset, setEditorAsset] = useState<string | null | undefined>(
    undefined,
  );
  const [sceneDialogOpen, setSceneDialogOpen] = useState(false);

  useOpenSceneHotkey(() => setSceneDialogOpen(true));

  return (
    <div className="ss-single-pane">
      <div className="ss-sp-viewport">
        <SceneViewport
          sceneId={sceneId}
          assetMarkColors={assetMarkColors}
          cameras={cameras}
          sensors={sensors}
          cameraRates={cameraRates}
          authToken={authToken}
          isSuperuser={isSuperuser}
        />
      </div>

      <aside className="ss-sp-dock" aria-label="Scene panels">
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
              onClick={() => setDockTab(t.id)}
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
              cameras={isSuperuser ? cameras : []}
            />
          )}
          {dockTab === "telemetry" && <Telemetry sceneId={sceneId} />}
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
