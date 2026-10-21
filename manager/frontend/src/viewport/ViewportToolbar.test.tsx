// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ViewportToolbar } from "./ViewportToolbar";

afterEach(() => vi.unstubAllGlobals());

describe("floating toolbar layout", () => {
  const cases = (["vertical", "horizontal"] as const).flatMap(orientation =>
    [false, true].map(collapsed => ({ orientation, collapsed })),
  );

  it.each(cases)("renders $orientation with collapsed=$collapsed", layout => {
    vi.stubGlobal("sessionStorage", { getItem: () => JSON.stringify(layout) });
    const markup = renderToStaticMarkup(<ViewportToolbar><button>Tool action</button></ViewportToolbar>);
    expect(markup).toContain(`aria-orientation="${layout.orientation}"`);
    expect(markup).toContain(`aria-label="${layout.collapsed ? "Show" : "Hide"} toolbar"`);
    expect(markup).toContain(`aria-expanded="${!layout.collapsed}"`);
    expect(markup.match(/class="ss-tools-layout-button"/g)).toHaveLength(3);
    expect(markup).toContain('title="Vertical toolbar"');
    expect(markup).toContain('title="Horizontal toolbar"');
    expect(markup).toContain('class="bi bi-window"');
    const icon = layout.orientation === "vertical"
      ? layout.collapsed ? "down" : "up"
      : layout.collapsed ? "right" : "left";
    expect(markup).toContain(`bi-chevron-${icon}`);
    expect(markup.includes('class="ss-tools-content" hidden=""')).toBe(layout.collapsed);
    expect(markup).toContain("Tool action");
  });

  it.each([null, "invalid", '{"orientation":"unknown","collapsed":"yes"}'])(
    "defaults safely for unavailable or invalid layout=%s",
    saved => {
      vi.stubGlobal("sessionStorage", { getItem: () => {
        if (saved === null) throw new Error("Storage unavailable");
        return saved;
      } });
      const markup = renderToStaticMarkup(<ViewportToolbar>Tools</ViewportToolbar>);
      expect(markup).toContain('aria-orientation="vertical"');
      expect(markup).toContain('aria-expanded="true"');
    },
  );
});