// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AppChrome } from "./AppChrome";
import { ViewportMenuBar } from "../viewport/ViewportMenuBar";
import type { ChromeBootstrap } from "./types";

const bootstrap: ChromeBootstrap = {
  authenticated: true, username: "test-user", isStaff: true, isKubernetes: false,
  appName: "Scenescape", appVersion: "test", appGitCommit: "test", docsVersion: "test",
  urls: { home: "/", scenes: "/", cameras: "/cameras/", sensors: "/sensors/",
    assets: "/assets/", admin: "/admin/", signOut: "/sign_out/", docs: "/docs/",
    support: "/support/", intel: "/", intelLogo: "/logo.png" },
};

vi.mock("./useChromeBootstrap", () => ({ useChromeBootstrap: () => bootstrap }));
vi.mock("./AboutModal", () => ({ AboutModal: () => null }));
vi.mock("./NavTrailingControls", () => ({ NavTrailingControls: () => null }));
afterEach(() => vi.unstubAllGlobals());

describe("System Administration menu entry", () => {
  it.each([true, false])("uses the same label and external icon with staff=%s", staff => {
    bootstrap.isStaff = staff;
    vi.stubGlobal("window", { location: { pathname: "/" } });
    vi.stubGlobal("document", { querySelector: () => null });
    for (const component of [<AppChrome bootstrap={bootstrap} />, <ViewportMenuBar />]) {
      const markup = renderToStaticMarkup(component);
      if (!staff) {
        expect(markup).not.toContain('id="nav-admin"');
        continue;
      }
      const entry = markup.match(/<a[^>]*id="nav-admin"[^>]*>[\s\S]*?<\/a>/)![0];
      expect(entry).toContain("System Administration");
      expect(entry).toContain("ss-admin-link");
      expect(entry).toContain('class="bi bi-box-arrow-up-right"');
      expect(entry).toContain('href="/admin/"');
      expect(entry).toContain('target="admin"');
    }
  });
});