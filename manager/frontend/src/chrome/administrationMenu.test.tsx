// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AppChrome } from "./AppChrome";
import { ViewportMenuBar } from "../viewport/ViewportMenuBar";
import type { ChromeBootstrap } from "./types";
import { useEditModeStore } from "../lib/editMode";

vi.mock("../lib/editMode", async importOriginal => {
  const original = await importOriginal<typeof import("../lib/editMode")>();
  return { ...original, useWorkspaceEditing: () => original.workspaceEditingAllowed() };
});

const bootstrap: ChromeBootstrap = {
  authenticated: true, username: "test-user", isStaff: true, isSuperuser: true, isKubernetes: false,
  appName: "Scenescape", appVersion: "test", appGitCommit: "test", docsVersion: "test",
  urls: { home: "/", scenes: "/", cameras: "/cameras/", sensors: "/sensors/",
    assets: "/assets/", admin: "/admin/", signOut: "/sign_out/", docs: "/docs/",
    support: "/support/", intel: "/", intelLogo: "/logo.png" },
};

vi.mock("./useChromeBootstrap", () => ({ useChromeBootstrap: () => bootstrap }));
vi.mock("./AboutModal", () => ({ AboutModal: () => null }));
vi.mock("./NavTrailingControls", () => ({ NavTrailingControls: () => null }));
const originalEditMode = useEditModeStore.getState();
afterEach(() => {
  useEditModeStore.setState(originalEditMode);
  bootstrap.isSuperuser = true;
  vi.unstubAllGlobals();
});

describe("System Administration menu entry", () => {
  it.each([false, true])("hides administration for read-only role or mode with superuser=%s", superuser => {
    bootstrap.isStaff = true;
    bootstrap.isSuperuser = superuser;
    useEditModeStore.getState().enterWorkspace(superuser);
    if (superuser) useEditModeStore.getState().setEditing(false);
    vi.stubGlobal("window", { location: { pathname: "/" } });
    vi.stubGlobal("document", { querySelector: () => null });
    for (const component of [<AppChrome bootstrap={bootstrap} />, <ViewportMenuBar />]) {
      expect(renderToStaticMarkup(component)).not.toContain('id="nav-admin"');
    }
  });

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