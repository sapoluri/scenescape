// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useId, useState, type ReactNode } from "react";

type ToolbarLayout = {
  orientation: "vertical" | "horizontal";
  collapsed: boolean;
};

const STORAGE_KEY = "ss-viewport-toolbar-layout";

function readLayout(): ToolbarLayout {
  try {
    const saved = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "null");
    return {
      orientation: saved?.orientation === "horizontal" ? "horizontal" : "vertical",
      collapsed: saved?.collapsed === true,
    };
  } catch {
    return { orientation: "vertical", collapsed: false };
  }
}

export function ViewportToolbar({ children }: { children: ReactNode }) {
  const [layout, setLayout] = useState(readLayout);
  const contentId = useId();
  const vertical = layout.orientation === "vertical";
  const toggleLabel = layout.collapsed ? "Show toolbar" : "Hide toolbar";
  const toggleIcon = vertical
    ? layout.collapsed ? "bi-chevron-down" : "bi-chevron-up"
    : layout.collapsed ? "bi-chevron-right" : "bi-chevron-left";

  useEffect(() => {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(layout));
    } catch {
      return;
    }
  }, [layout]);

  return (
    <div
      className={`ss-viewport-tools is-${layout.orientation}${layout.collapsed ? " is-collapsed" : ""}`}
      role="toolbar"
      aria-label="Edit tools"
      aria-orientation={layout.orientation}
    >
      <div className="ss-tools-layout-header" role="group" aria-label="Toolbar layout">
        <button
          type="button"
          className="ss-tools-layout-button"
          title={toggleLabel}
          aria-label={toggleLabel}
          aria-expanded={!layout.collapsed}
          aria-controls={contentId}
          onClick={() => setLayout(current => ({ ...current, collapsed: !current.collapsed }))}
        >
          <i className={`bi ${toggleIcon}`} aria-hidden="true" />
        </button>
        {(["vertical", "horizontal"] as const).map(orientation => (
          <button
            key={orientation}
            type="button"
            className="ss-tools-layout-button"
            title={orientation === "vertical" ? "Vertical toolbar" : "Horizontal toolbar"}
            aria-label={orientation === "vertical" ? "Vertical toolbar" : "Horizontal toolbar"}
            aria-pressed={layout.orientation === orientation}
            onClick={() => setLayout(current => ({ ...current, orientation }))}
          >
            <i className={`bi ${orientation === "vertical" ? "bi-layout-sidebar" : "bi-window"}`} aria-hidden="true" />
          </button>
        ))}
      </div>
      <div id={contentId} className="ss-tools-content" hidden={layout.collapsed}>
        {children}
      </div>
    </div>
  );
}