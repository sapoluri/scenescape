// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { readBootstrapJson } from "./lib/bootstrap";
import { ensureSceneDetailDom } from "./lib/ensureSceneDetailDom";
import { sceneMapBitmapUrl } from "./lib/sceneMapBitmap";
import { SceneDetailApp } from "./scene/SceneDetailApp";
import type { SceneDetailBootstrap } from "./scene/types";
import "./tokens/tokens.css";
import "./scene-detail.css";

const bootstrap = readBootstrapJson<SceneDetailBootstrap>(
  "ss-scene-detail-bootstrap",
);
const rootEl = document.getElementById("ss-scene-detail-root");

if (bootstrap && rootEl) {
  /* Prefer React map before sscape.js document.ready reads the flag. */
  window.ssUseReactMap = Boolean(sceneMapBitmapUrl(bootstrap.scene));
  /* Scene-detail MQTT connect is owned by React when the React map is on. */
  window.ssReactOwnsMqtt = Boolean(window.ssUseReactMap);
  window.ssReactOwnsCameraStrip = Boolean(window.ssUseReactMap);
  /* Map host + geometry hiddens from bootstrap — not Django template siblings. */
  ensureSceneDetailDom(bootstrap);
  /* Own the viewport before paint settles — Django chrome becomes a slim shell. */
  document.documentElement.classList.add("ss-scene-workspace");
  document.body.classList.add("ss-scene-workspace");
  createRoot(rootEl).render(
    <StrictMode>
      <SceneDetailApp bootstrap={bootstrap} />
    </StrictMode>,
  );
}
