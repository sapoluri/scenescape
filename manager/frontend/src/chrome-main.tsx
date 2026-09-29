// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { readBootstrapJson } from "./lib/bootstrap";
import { AppChrome } from "./chrome/AppChrome";
import type { ChromeBootstrap } from "./chrome/types";
import "./tokens/tokens.css";

const bootstrap = readBootstrapJson<ChromeBootstrap>("ss-chrome-bootstrap");
const host = document.getElementById("ss-chrome-root");

if (bootstrap && host) {
  createRoot(host).render(
    <StrictMode>
      <AppChrome bootstrap={bootstrap} />
    </StrictMode>,
  );
}
