// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { ToastProvider } from "./components/ToastProvider";
import { readBootstrapJson } from "./lib/bootstrap";
import { ModelsDirectoryApp } from "./models/ModelsDirectoryApp";
import "./tokens/tokens.css";

type Bootstrap = { isSuperuser?: boolean };

const bootstrap =
  readBootstrapJson<Bootstrap>("ss-models-directory-bootstrap") || {};
const rootEl = document.getElementById("ss-models-directory-root");
if (rootEl) {
  createRoot(rootEl).render(
    <StrictMode>
      <ToastProvider>
        <ModelsDirectoryApp isSuperuser={Boolean(bootstrap.isSuperuser)} />
      </ToastProvider>
    </StrictMode>,
  );
}
