// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { readBootstrapJson } from "./lib/bootstrap";
import {
  ScenesHomeApp,
  type ScenesHomeBootstrap,
} from "./scenes/ScenesHomeApp";
import "./tokens/tokens.css";

const bootstrap = readBootstrapJson<ScenesHomeBootstrap>(
  "ss-scenes-home-bootstrap",
);
const host =
  document.getElementById("ss-scenes-home-app") ||
  (() => {
    const el = document.createElement("div");
    el.id = "ss-scenes-home-app";
    const main =
      document.querySelector("main") ||
      document.querySelector(".container") ||
      document.body;
    main.appendChild(el);
    return el;
  })();

if (bootstrap) {
  createRoot(host).render(
    <StrictMode>
      <ScenesHomeApp bootstrap={bootstrap} />
    </StrictMode>,
  );
}
