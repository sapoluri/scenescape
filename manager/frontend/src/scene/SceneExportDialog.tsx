// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { api } from "../lib/rest";

type ZipArchive = {
  file: (path: string, data: Blob | string) => void;
  generateAsync: (options: { type: "blob" }) => Promise<Blob>;
};

type ZipConstructor = new () => ZipArchive;

type SceneMapAsset = {
  name?: unknown;
  map?: unknown;
};

type SceneExportData = Record<string, unknown> & {
  name?: unknown;
  map?: unknown;
  children?: unknown;
};

type Props = {
  open: boolean;
  sceneId: string;
  sceneName: string;
  authToken: string;
  onClose: () => void;
};

function archiveBaseName(name: string): string {
  return name.trim().replace(/\s+/g, "_").replace(/[\\/:*?"<>|]/g, "_") || "scene";
}

function mapExtension(url: string): string {
  const path = new URL(url, window.location.href).pathname;
  const extension = path.split(".").pop() || "bin";
  return /^[a-z0-9]{1,8}$/i.test(extension) ? extension : "bin";
}

function getZipConstructor(): ZipConstructor {
  const ctor = (window as Window & { JSZip?: ZipConstructor }).JSZip;
  if (!ctor) {
    throw new Error("Scene archive export is unavailable in this page");
  }
  return ctor;
}

/** Export matches the legacy scene archive format: scene JSON plus map assets. */
export function SceneExportDialog({
  open,
  sceneId,
  sceneName,
  authToken,
  onClose,
}: Props) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const onExport = async () => {
    if (busy) {
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const scene = (await api.getScene(authToken, sceneId)) as SceneExportData;
      const baseName = archiveBaseName(
        typeof scene.name === "string" ? scene.name : sceneName,
      );
      const zip = new (getZipConstructor())();
      zip.file(`${baseName}.json`, JSON.stringify(scene, null, 2));

      const assets: Array<{ url: string; name: string }> = [];
      if (typeof scene.map === "string" && scene.map.trim()) {
        assets.push({ url: scene.map, name: baseName });
      }
      if (Array.isArray(scene.children)) {
        for (const child of scene.children as SceneMapAsset[]) {
          if (typeof child?.map === "string" && child.map.trim()) {
            assets.push({
              url: child.map,
              name:
                typeof child.name === "string"
                  ? archiveBaseName(child.name)
                  : "child_scene",
            });
          }
        }
      }

      const omitted: string[] = [];
      for (const asset of assets) {
        try {
          const response = await fetch(asset.url, {
            credentials: "same-origin",
          });
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }
          zip.file(`${asset.name}.${mapExtension(asset.url)}`, await response.blob());
        } catch {
          omitted.push(asset.name);
        }
      }

      const blob = await zip.generateAsync({ type: "blob" });
      const objectUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = `${baseName}.zip`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);

      if (omitted.length > 0) {
        setMessage(
          `Archive downloaded, but these map assets could not be included: ${omitted.join(", ")}.`,
        );
      } else {
        onClose();
      }
    } catch (err) {
      setMessage((err as Error).message || "Scene export failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <ConfirmDialog
      open={open}
      title="Export scene"
      confirmLabel={busy ? "Exporting…" : "Download ZIP"}
      cancelLabel="Cancel"
      danger={false}
      busy={busy}
      onConfirm={() => void onExport()}
      onCancel={() => {
        if (!busy) {
          setMessage(null);
          onClose();
        }
      }}
    >
      <p>
        Download {sceneName} as a ZIP containing its scene data and available
        map files.
      </p>
      {message ? (
        <p role="alert" style={{ color: "var(--ss-status-bad)" }}>
          {message}
        </p>
      ) : null}
    </ConfirmDialog>
  );
}