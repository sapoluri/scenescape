// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ToastProvider, useAppToast } from "../components/ToastProvider";
import { LegacyConfirmHost } from "../components/LegacyConfirmHost";
import { useSheetFromQuery } from "../hooks/useSheetFromQuery";
import type { SheetAction } from "../lib/sheetQuery";
import { LibraryDrawer } from "../library/LibraryDrawer";
import { MarkEditor } from "../library/MarkEditor";
import {
  OpenAdminListDialog,
  type OpenAdminListKind,
} from "../panels/OpenAdminListDialog";
import {
  OpenSceneDialog,
  useOpenSceneHotkey,
  type GalleryScene,
} from "../panels/OpenSceneDialog";
import { SceneViewport } from "../viewport/SceneViewport";
import { SceneSheet } from "../sheets/SceneSheet";
import { SceneImportDialog } from "../sheets/SceneImportDialog";
import { SceneManagePanel } from "../sheets/SceneManagePanel";
import { ChildSheet } from "../sheets/ChildSheet";
import { CameraSheet } from "../sheets/CameraSheet";
import { SensorSheet } from "../sheets/SensorSheet";
import "../components/Button.css";
import "../panels/SinglePaneView.css";
import "./ScenesHomeApp.css";

export type SceneHomeCard = {
  id: string;
  name: string;
  georeferenced?: boolean;
  thumbnailUrl: string | null;
  mapUrl: string | null;
  detailUrl: string;
  manageUrl: string;
  deleteUrl: string | null;
  counts: {
    sensors: number;
    regions: number;
    tripwires: number;
  };
};

export type ScenesHomeBootstrap = {
  authToken: string;
  isSuperuser: boolean;
  scenes: SceneHomeCard[];
};

type Props = {
  bootstrap: ScenesHomeBootstrap;
};

const HOME_ACTIONS = new Set([
  "scene-create",
  "scene-import",
  "scene-manage",
  "child-create",
  "cam-create",
  "cam-edit",
  "sensor-create",
  "sensor-edit",
]);

function isHomeAction(v: string | null): v is Exclude<SheetAction, null> {
  return Boolean(v && HOME_ACTIONS.has(v));
}

function toGalleryScenes(cards: SceneHomeCard[]): GalleryScene[] {
  return cards.map((s) => ({
    uid: s.id,
    name: s.name,
    detailUrl: s.detailUrl,
    manageUrl: s.manageUrl,
    deleteUrl: s.deleteUrl,
    thumbnailUrl: s.thumbnailUrl,
    mapUrl: s.mapUrl,
    counts: s.counts,
  }));
}

function ScenesHomeInner({ bootstrap }: Props) {
  const { sheet, open, close } = useSheetFromQuery();
  useAppToast();
  const [sceneDialogOpen, setSceneDialogOpen] = useState(
    () => !(sheet.action && isHomeAction(sheet.action)),
  );
  const [adminListKind, setAdminListKind] = useState<OpenAdminListKind | null>(
    null,
  );
  const [adminListRefresh, setAdminListRefresh] = useState(0);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [libraryRefresh, setLibraryRefresh] = useState(0);
  // undefined = editor closed; null = create-new; string = edit asset.
  const [editorAsset, setEditorAsset] = useState<string | null | undefined>(
    undefined,
  );

  useOpenSceneHotkey(() => {
    setAdminListKind(null);
    setSceneDialogOpen(true);
  });

  useEffect(() => {
    document.documentElement.classList.add("ss-scene-workspace");
    document.body.classList.add("ss-scene-workspace");
    return () => {
      document.documentElement.classList.remove("ss-scene-workspace");
      document.body.classList.remove("ss-scene-workspace");
    };
  }, []);

  // Browser Back restores this page from bfcache with whatever React state
  // was frozen at navigate-away (often gallery closed). Re-open the gallery
  // so home matches a fresh visit unless a sheet is active.
  const sheetActionRef = useRef(sheet.action);
  sheetActionRef.current = sheet.action;
  useEffect(() => {
    const onPageShow = (ev: PageTransitionEvent) => {
      if (!ev.persisted) {
        return;
      }
      const action = sheetActionRef.current;
      if (!(action && isHomeAction(action))) {
        setSceneDialogOpen(true);
      }
    };
    window.addEventListener("pageshow", onPageShow);
    return () => window.removeEventListener("pageshow", onPageShow);
  }, []);

  useEffect(() => {
    const onClick = (ev: Event) => {
      const target = ev.target as HTMLElement | null;
      const link = target?.closest("a[href]") as HTMLAnchorElement | null;
      if (!link?.href) {
        return;
      }
      let url: URL;
      try {
        url = new URL(link.href, window.location.origin);
      } catch {
        return;
      }
      const ss = url.searchParams.get("ss");
      if (!ss || !isHomeAction(ss)) {
        return;
      }
      if (url.pathname === "/" || url.pathname === "") {
        ev.preventDefault();
        open(ss, url.searchParams.get("id"));
        setSceneDialogOpen(false);
      }
    };
    document.addEventListener("click", onClick, true);
    return () => document.removeEventListener("click", onClick, true);
  }, [open]);

  // Sheets hide the galleries while open; dismissing a sheet reopens scenes
  // unless a cameras/sensors browser was the prior surface.
  const prevSheetAction = useRef(sheet.action);
  useEffect(() => {
    const prev = prevSheetAction.current;
    prevSheetAction.current = sheet.action;
    if (sheet.action && isHomeAction(sheet.action)) {
      setSceneDialogOpen(false);
      setAdminListKind(null);
    } else if (prev && !sheet.action) {
      setSceneDialogOpen(true);
      setAdminListRefresh((n) => n + 1);
    }
  }, [sheet.action]);

  const reload = useCallback(() => {
    window.location.reload();
  }, []);

  const openCreate = useCallback(() => {
    setSceneDialogOpen(false);
    open("scene-create");
  }, [open]);

  const openImport = useCallback(() => {
    setSceneDialogOpen(false);
    open("scene-import");
  }, [open]);

  const galleryScenes = useMemo(
    () => toGalleryScenes(bootstrap.scenes || []),
    [bootstrap.scenes],
  );

  const sceneOptions = useMemo(
    () =>
      (bootstrap.scenes || []).map((s) => ({
        id: s.id,
        name: s.name,
        georeferenced: Boolean(s.georeferenced),
        mapUrl: s.mapUrl || null,
      })),
    [bootstrap.scenes],
  );
  const manageSceneId =
    sheet.action === "scene-manage" && sheet.id ? sheet.id : "";

  return (
    <div className="ss-scenes-home ss-scenes-home--workspace">
      <div className="ss-single-pane">
        <div className="ss-sp-viewport">
          <SceneViewport
            authToken={bootstrap.authToken}
            isSuperuser={bootstrap.isSuperuser}
            onOpenScene={() => {
              setAdminListKind(null);
              setSceneDialogOpen(true);
            }}
            onOpenCameras={() => {
              setSceneDialogOpen(false);
              setAdminListKind("cameras");
            }}
            onOpenSensors={() => {
              setSceneDialogOpen(false);
              setAdminListKind("sensors");
            }}
            onOpenLibrary={() => setLibraryOpen(true)}
          />
        </div>
      </div>

      <LibraryDrawer
        open={libraryOpen}
        onClose={() => setLibraryOpen(false)}
        onOpen={() => setLibraryOpen(true)}
        authToken={bootstrap.authToken}
        onSelectAsset={(uid) => {
          setEditorAsset(uid);
          setLibraryOpen(false);
        }}
        refreshToken={libraryRefresh}
      />
      {editorAsset !== undefined ? (
        <MarkEditor
          assetId={editorAsset}
          authToken={bootstrap.authToken}
          onClose={() => setEditorAsset(undefined)}
          onSaved={() => {
            setLibraryRefresh((n) => n + 1);
            setEditorAsset(undefined);
          }}
        />
      ) : null}

      <OpenSceneDialog
        open={sceneDialogOpen}
        onClose={() => setSceneDialogOpen(false)}
        onOpenScene={(uid) => {
          // Leave the dialog open in page state so bfcache / Back restores
          // home with the gallery visible.
          window.location.assign(`/${uid}/`);
        }}
        authToken={bootstrap.authToken}
        isSuperuser={bootstrap.isSuperuser}
        initialScenes={galleryScenes}
        onNewScene={openCreate}
        onImportScene={openImport}
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

      {bootstrap.isSuperuser ? (
        <>
          <SceneSheet
            open={sheet.action === "scene-create"}
            mode="create"
            sceneUid={null}
            authToken={bootstrap.authToken}
            onClose={close}
            onSaved={(uid, opts) => {
              if (uid) {
                const q =
                  opts?.setup === "reconstruct" ? "?setup=reconstruct" : "";
                window.location.href = `/${uid}/${q}`;
                return;
              }
              reload();
            }}
          />
          <SceneImportDialog
            open={sheet.action === "scene-import"}
            authToken={bootstrap.authToken}
            onClose={close}
            onImported={reload}
          />
          <SceneManagePanel
            open={Boolean(manageSceneId)}
            sceneId={manageSceneId}
            authToken={bootstrap.authToken}
            onClose={close}
            onSaved={reload}
          />
          <ChildSheet
            open={sheet.action === "child-create"}
            mode="create"
            parentSceneId=""
            scenes={sceneOptions}
            authToken={bootstrap.authToken}
            onClose={close}
            onSaved={reload}
          />
          <CameraSheet
            open={
              sheet.action === "cam-create" || sheet.action === "cam-edit"
            }
            mode={sheet.action === "cam-edit" ? "edit" : "create"}
            sceneId=""
            scenes={sceneOptions}
            sensorUid={
              sheet.action === "cam-edit" ? sheet.id || null : null
            }
            authToken={bootstrap.authToken}
            onClose={close}
            onSaved={() => {
              close();
              setAdminListKind("cameras");
              setAdminListRefresh((n) => n + 1);
            }}
          />
          <SensorSheet
            open={
              sheet.action === "sensor-create" ||
              sheet.action === "sensor-edit"
            }
            mode={sheet.action === "sensor-edit" ? "edit" : "create"}
            sceneId=""
            scenes={sceneOptions}
            sensorUid={
              sheet.action === "sensor-edit" ? sheet.id || null : null
            }
            authToken={bootstrap.authToken}
            onClose={close}
            onSaved={() => {
              close();
              setAdminListKind("sensors");
              setAdminListRefresh((n) => n + 1);
            }}
          />
        </>
      ) : null}
    </div>
  );
}

export function ScenesHomeApp({ bootstrap }: Props) {
  return (
    <ToastProvider>
      <LegacyConfirmHost>
        <ScenesHomeInner bootstrap={bootstrap} />
      </LegacyConfirmHost>
    </ToastProvider>
  );
}
