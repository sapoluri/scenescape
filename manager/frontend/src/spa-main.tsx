// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Static-shell entry: chrome + scenes home or scene detail from ui-bootstrap API.
 * Mounted by /static/ui/shell.html (no Django page template).
 */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { loadUiBootstrap } from "./lib/uiBootstrap";
import { ensureSceneDetailDom } from "./lib/ensureSceneDetailDom";
import { sceneMapBitmapUrl } from "./lib/sceneMapBitmap";
import { AppChrome } from "./chrome/AppChrome";
import type { ChromeBootstrap } from "./chrome/types";
import {
  ScenesHomeApp,
  type ScenesHomeBootstrap,
} from "./scenes/ScenesHomeApp";
import { SceneDetailApp } from "./scene/SceneDetailApp";
import type { SceneDetailBootstrap } from "./scene/types";
import "./tokens/tokens.css";
import "./scene-detail.css";

function sceneIdFromPath(): string | null {
  const m = window.location.pathname.match(
    /^\/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\/?/i,
  );
  return m?.[1] ?? null;
}

async function mountChrome(): Promise<void> {
  const bootstrap = await loadUiBootstrap<ChromeBootstrap>(
    "ss-chrome-bootstrap",
    "chrome",
  );
  const host = document.getElementById("ss-chrome-root");
  if (!bootstrap || !host) {
    return;
  }
  createRoot(host).render(
    <StrictMode>
      <AppChrome bootstrap={bootstrap} />
    </StrictMode>,
  );
}

async function mountPage(): Promise<void> {
  const spaRoot = document.getElementById("ss-spa-root");
  if (!spaRoot) {
    return;
  }
  const sceneId = sceneIdFromPath();
  if (sceneId) {
    const bootstrap = await loadUiBootstrap<SceneDetailBootstrap>(
      "ss-scene-detail-bootstrap",
      "scene",
      sceneId,
    );
    if (!bootstrap) {
      return;
    }
    const detailRoot = document.createElement("div");
    detailRoot.id = "ss-scene-detail-root";
    detailRoot.className = "ss-scene-detail-root";
    spaRoot.appendChild(detailRoot);
    window.ssUseReactMap = Boolean(sceneMapBitmapUrl(bootstrap.scene));
    window.ssReactOwnsMqtt = Boolean(window.ssUseReactMap);
    window.ssReactOwnsCameraStrip = Boolean(window.ssUseReactMap);
    ensureSceneDetailDom(bootstrap);
    document.documentElement.classList.add("ss-scene-workspace");
    document.body.classList.add("ss-scene-workspace");
    createRoot(detailRoot).render(
      <StrictMode>
        <SceneDetailApp bootstrap={bootstrap} />
      </StrictMode>,
    );
    return;
  }

  const bootstrap = await loadUiBootstrap<ScenesHomeBootstrap>(
    "ss-scenes-home-bootstrap",
    "scenes",
  );
  if (!bootstrap) {
    return;
  }
  const homeRoot = document.createElement("div");
  homeRoot.id = "ss-scenes-home-app";
  spaRoot.appendChild(homeRoot);
  createRoot(homeRoot).render(
    <StrictMode>
      <ScenesHomeApp bootstrap={bootstrap} />
    </StrictMode>,
  );
}

async function main(): Promise<void> {
  await mountChrome();
  await mountPage();
}

void main();
