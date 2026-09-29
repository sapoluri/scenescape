// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { AdminListApp, type AdminListBootstrap } from "./admin/AdminListApp";
import { readBootstrapJson } from "./lib/bootstrap";
import "./tokens/tokens.css";

const bootstrap = readBootstrapJson<AdminListBootstrap>(
  "ss-admin-list-bootstrap",
);
const rootEl = document.getElementById("ss-admin-list-root");

if (bootstrap && rootEl) {
  createRoot(rootEl).render(
    <StrictMode>
      <AdminListApp bootstrap={bootstrap} />
    </StrictMode>,
  );
}
