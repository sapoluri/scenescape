// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import { ConfirmDialog } from "./ConfirmDialog";
import { Button } from "./Button";
import "./Drawer.css";

type Props = {
  open: boolean;
  title: string;
  children: ReactNode;
  /** Primary actions (Save) — shown next to Cancel in the header. */
  actions?: ReactNode;
  /**
   * Show Cancel beside header actions. Defaults to false — the × close control
   * already dismisses (with the same dirty-leave confirm).
   */
  showCancel?: boolean;
  /** Optional extra bottom bar (prefer header Cancel + Save instead). */
  footer?: ReactNode;
  wide?: boolean;
  dirty?: boolean;
  leaveTitle?: string;
  leaveBody?: string;
  onClose: () => void;
};

/**
 * Right-side command surface for simple create/edit sheets.
 * Cancel and Save sit together in the header (trailing edge).
 * Shares Escape / dirty-leave confirm with WorkspacePanel.
 */
export function Drawer({
  open,
  title,
  children,
  actions,
  showCancel,
  footer,
  wide = false,
  dirty = false,
  leaveTitle = "Leave without saving?",
  leaveBody = "You may have unsaved changes. Leave without saving?",
  onClose,
}: Props) {
  const titleId = useId();
  const closeRef = useRef<HTMLButtonElement>(null);
  const [confirmClose, setConfirmClose] = useState(false);
  const cancelVisible = showCancel ?? false;
  const dirtyRef = useRef(dirty);
  dirtyRef.current = dirty;
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  const requestClose = useCallback(() => {
    if (dirtyRef.current) {
      setConfirmClose(true);
      return;
    }
    onCloseRef.current();
  }, []);

  // Prefer the single-pane body (viewport + outliner dock) so sheets overlay
  // the outliner but not the menu / options bars. Then the 3D stage alone;
  // browse pages fall back to document.body.
  const stage =
    typeof document !== "undefined"
      ? document.querySelector<HTMLElement>(".ss-sp-body") ??
        document.querySelector<HTMLElement>(
          ".ss-viewport-main-3d, .ss-viewport-split-3d",
        )
      : null;
  const portalTarget =
    stage ?? (typeof document !== "undefined" ? document.body : null);

  useEffect(() => {
    if (!open) {
      setConfirmClose(false);
      return;
    }
    // Only lock page scroll for full-viewport drawers (no stage host).
    const prev = document.body.style.overflow;
    if (!stage) {
      document.body.style.overflow = "hidden";
    }
    // Autofocus close only when the drawer opens — not when dirty flips,
    // or the first keystroke steals focus from the field being edited.
    closeRef.current?.focus();
    const panel = closeRef.current?.closest(
      ".ss-drawer-panel",
    ) as HTMLElement | null;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        requestClose();
        return;
      }
      if (e.key !== "Tab" || !panel) {
        return;
      }
      const focusable = panel.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );
      if (!focusable.length) {
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      if (!stage) {
        document.body.style.overflow = prev;
      }
      window.removeEventListener("keydown", onKey);
    };
  }, [open, requestClose, stage]);

  if (!open || !portalTarget) {
    return null;
  }

  return createPortal(
    <div
      className={`ss-drawer-root is-open${stage ? " is-stage" : ""}`}
      role="presentation"
    >
      <div className="ss-drawer-backdrop" onClick={requestClose} />
      <aside
        className={`ss-drawer-panel${wide ? " ss-drawer-panel--wide" : ""}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
      >
        <div className="ss-drawer-header">
          <h2 className="ss-drawer-title" id={titleId}>
            {title}
          </h2>
          <div className="ss-drawer-header-end">
            {cancelVisible || actions ? (
              <div className="ss-drawer-header-actions">
                {cancelVisible ? (
                  <Button variant="secondary" onClick={requestClose}>
                    Cancel
                  </Button>
                ) : null}
                {actions}
              </div>
            ) : null}
            <button
              ref={closeRef}
              type="button"
              className="ss-drawer-close"
              aria-label="Close"
              onClick={requestClose}
            >
              ×
            </button>
          </div>
        </div>
        <div className="ss-drawer-body">{children}</div>
        {footer ? <div className="ss-drawer-footer">{footer}</div> : null}
      </aside>
      <ConfirmDialog
        open={confirmClose}
        title={leaveTitle}
        confirmLabel="Leave"
        danger
        onConfirm={() => {
          setConfirmClose(false);
          onClose();
        }}
        onCancel={() => setConfirmClose(false)}
      >
        <p>{leaveBody}</p>
      </ConfirmDialog>
    </div>,
    portalTarget,
  );
}
