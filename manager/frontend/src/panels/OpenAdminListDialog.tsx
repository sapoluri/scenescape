// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Floating Cameras / Sensors browser — same chrome as Open Scene, backed by
 * ui-bootstrap list payloads so Selenium hard contracts
 * (`#ss-admin-list-root`, `#new-camera`, `#new-sensor`, table actions) stay
 * intact without navigating away from the 3D workspace.
 */

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import { createPortal } from "react-dom";
import {
  AdminListApp,
  type AdminListBootstrap,
  type AdminListRow,
} from "../admin/AdminListApp";
import { loadUiBootstrap, type UiBootstrapPage } from "../lib/uiBootstrap";
import { SCENE_TAB_EVENT } from "../lib/sceneTab";
import "./OpenSceneDialog.css";
import "./OpenAdminListDialog.css";

export type OpenAdminListKind = "cameras" | "sensors";

type Props = {
  open: boolean;
  kind: OpenAdminListKind;
  onClose: () => void;
  /** Bump to force a reload after create/delete. */
  refreshToken?: number;
};

function sceneIdFromRow(row: AdminListRow): string | null {
  for (const cell of row.cells) {
    if (!cell.href) {
      continue;
    }
    const m = cell.href.match(/\/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\/?/i);
    if (m) {
      return m[1];
    }
  }
  return null;
}

/** Parse `?ss=` / `id=` without depending on `window.location`. */
export function sheetQueryFromHref(href: string): {
  ss: string | null;
  id: string | null;
} {
  const qIndex = href.indexOf("?");
  if (qIndex < 0) {
    return { ss: null, id: null };
  }
  const params = new URLSearchParams(href.slice(qIndex + 1));
  return { ss: params.get("ss"), id: params.get("id") };
}

/**
 * Rewrite list-page action hrefs so create/edit stay on the current workspace
 * (`?ss=…`) and calibrate jumps to the owning scene when known.
 */
export function adaptListBootstrapForWorkspace(
  boot: AdminListBootstrap,
  kind: OpenAdminListKind,
): AdminListBootstrap {
  const createSs = kind === "cameras" ? "cam-create" : "sensor-create";
  const editSs = kind === "cameras" ? "cam-edit" : "sensor-edit";
  const calibrateSs =
    kind === "cameras" ? "calibrate-cam" : "calibrate-sensor";

  return {
    ...boot,
    breadcrumbs: undefined,
    primaryAction: boot.primaryAction
      ? {
          ...boot.primaryAction,
          href: `?ss=${createSs}`,
        }
      : undefined,
    rows: boot.rows.map((row) => ({
      ...row,
      actions: (row.actions || []).map((action) => {
        if (action.label === "Delete" || action.tone === "danger") {
          return action;
        }
        const { ss, id } = sheetQueryFromHref(action.href);
        if (!ss) {
          return action;
        }
        if (ss === calibrateSs) {
          const sceneId = sceneIdFromRow(row);
          if (sceneId) {
            return {
              ...action,
              href: `/${sceneId}/?ss=${ss}&id=${encodeURIComponent(id || "")}`,
            };
          }
        }
        if (ss === editSs || ss === createSs || ss === calibrateSs) {
          return {
            ...action,
            href: `?ss=${ss}${id ? `&id=${encodeURIComponent(id)}` : ""}`,
          };
        }
        return action;
      }),
    })),
  };
}

export function OpenAdminListDialog({
  open,
  kind,
  onClose,
  refreshToken = 0,
}: Props) {
  const [boot, setBoot] = useState<AdminListBootstrap | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const page: UiBootstrapPage = kind;

  useEffect(() => {
    if (!open) return;
    const onSheetOpen = (event: Event) => {
      if ((event as CustomEvent<{ tabId?: string }>).detail?.tabId === kind) onClose();
    };
    window.addEventListener(SCENE_TAB_EVENT, onSheetOpen);
    return () => window.removeEventListener(SCENE_TAB_EVENT, onSheetOpen);
  }, [open, kind, onClose]);

  useEffect(() => {
    if (!open) {
      return;
    }
    setError(null);
    setLoading(true);
    let cancelled = false;
    // No embedded json_script on the 3D shell — fall through to the API.
    void loadUiBootstrap<AdminListBootstrap>(
      `ss-${kind}-list-bootstrap`,
      page,
    ).then((payload) => {
      if (cancelled) {
        return;
      }
      if (!payload) {
        setError(`Failed to load ${kind}.`);
        setBoot(null);
        setLoading(false);
        return;
      }
      setBoot(adaptListBootstrapForWorkspace(payload, kind));
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [open, kind, page, refreshToken]);

  const onDialogKeyDown = useCallback(
    (ev: ReactKeyboardEvent) => {
      if (ev.key === "Escape") {
        ev.stopPropagation();
        onClose();
      }
    },
    [onClose],
  );

  if (typeof document === "undefined" || !open) {
    return null;
  }

  const title = kind === "cameras" ? "Cameras" : "Sensors";

  const node = (
    <div
      className="ss-open-scene-backdrop"
      onMouseDown={(ev) => {
        if (ev.target === ev.currentTarget) {
          onClose();
        }
      }}
    >
      <div
        ref={dialogRef}
        className="ss-open-scene-dialog ss-open-admin-dialog"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onKeyDown={onDialogKeyDown}
      >
        <div className="ss-open-scene-header">
          <h2 className="ss-open-scene-title">{title}</h2>
          <div className="ss-open-scene-header-actions">
            <button
              type="button"
              className="ss-open-scene-close"
              onClick={onClose}
              aria-label="Close"
            >
              ×
            </button>
          </div>
        </div>
        <div className="ss-open-admin-body" id="ss-admin-list-root">
          {loading ? (
            <p className="ss-open-scene-status">Loading {kind}…</p>
          ) : error ? (
            <p className="ss-open-scene-status ss-open-scene-error" role="alert">
              {error}
            </p>
          ) : boot ? (
            <AdminListApp bootstrap={boot} />
          ) : null}
        </div>
      </div>
    </div>
  );

  return createPortal(node, document.body);
}
