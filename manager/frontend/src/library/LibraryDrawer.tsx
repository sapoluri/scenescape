// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useRef, useState } from "react";
import {
  glbBasename,
  listAssets,
  type Asset3DRecord,
} from "./AssetLibrary";
import type { RestError } from "../lib/rest";
import "./LibraryDrawer.css";

type Props = {
  /** Controlled visibility. The drawer stays mounted so `B` can reopen it. */
  open: boolean;
  onClose: () => void;
  /** Called when `B` is pressed while closed. Required for keyboard reopen. */
  onOpen?: () => void;
  /** Token for the Asset3D REST calls. Drawer shows an error without one. */
  authToken?: string;
  /** Card click → open the mark editor. `null` = new class. */
  onSelectAsset?: (uid: string | null) => void;
  /** Refresh the card list (e.g. after the editor saves). */
  refreshToken?: number;
};

/**
 * Object Library drawer (Phase 1.3): docked-left per-class (`Asset3D`) cards.
 * `B` toggles it; clicking a card opens the mark editor via `onSelectAsset`.
 *
 * Mount: <LibraryDrawer open={open} onClose={…} onOpen={…}
 *          authToken={token} onSelectAsset={(uid) => …} />
 */
export function LibraryDrawer({
  open,
  onClose,
  onOpen,
  authToken,
  onSelectAsset,
  refreshToken = 0,
}: Props) {
  const [assets, setAssets] = useState<Asset3DRecord[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const openRef = useRef(open);
  openRef.current = open;
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  const onOpenRef = useRef(onOpen);
  onOpenRef.current = onOpen;

  // `B` toggles the drawer; never hijack typing.
  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if (ev.metaKey || ev.ctrlKey || ev.altKey) {
        return;
      }
      if (ev.key !== "b" && ev.key !== "B") {
        return;
      }
      const t = ev.target as HTMLElement | null;
      if (
        t &&
        (t.tagName === "INPUT" ||
          t.tagName === "TEXTAREA" ||
          t.tagName === "SELECT" ||
          t.isContentEditable)
      ) {
        return;
      }
      // The mark editor is modal; don't toggle the drawer behind it.
      if (document.querySelector(".ss-mark-editor")) {
        return;
      }
      ev.preventDefault();
      if (openRef.current) {
        onCloseRef.current();
      } else {
        onOpenRef.current?.();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Escape closes when open (the mark editor handles its own Escape).
  useEffect(() => {
    if (!open) {
      return;
    }
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === "Escape") {
        // The mark editor is modal; leave Escape to it while it is up.
        if (document.querySelector(".ss-mark-editor")) {
          return;
        }
        onCloseRef.current();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open ]);

  useEffect(() => {
    if (!open) {
      return;
    }
    if (!authToken) {
      setError("Sign in required to load the object library.");
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    listAssets(authToken)
      .then((rows) => {
        if (!cancelled) {
          setAssets(rows);
        }
      })
      .catch((e: RestError) => {
        if (!cancelled) {
          setError(e.message || "Failed to load the object library");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [open, authToken, refreshToken]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) {
      return assets;
    }
    return assets.filter(
      (a) =>
        a.name.toLowerCase().includes(q) ||
        glbBasename(a.modelUrl).toLowerCase().includes(q),
    );
  }, [assets, query]);

  if (!open) {
    return null;
  }

  return (
    <aside
      className="ss-lib-drawer"
      aria-label="Object library"
      data-testid="library-drawer"
    >
      <div className="ss-lib-drawer-header">
        <div className="ss-lib-drawer-title">
          <span className="ss-lib-drawer-title-text">Object Library</span>
          <span className="ss-lib-drawer-count">{assets.length}</span>
        </div>
        <button
          type="button"
          className="ss-lib-drawer-close"
          aria-label="Close object library"
          onClick={onClose}
        >
          ×
        </button>
      </div>
      <div className="ss-lib-drawer-search">
        <input
          type="search"
          placeholder="Search classes or GLB files…"
          aria-label="Search object library"
          value={query}
          onChange={(ev) => setQuery(ev.target.value)}
        />
      </div>
      <div className="ss-lib-drawer-body">
        {loading ? (
          <p className="ss-lib-drawer-status">Loading classes…</p>
        ) : error ? (
          <p className="ss-lib-drawer-error">{error}</p>
        ) : filtered.length === 0 ? (
          <p className="ss-lib-drawer-status">
            {assets.length === 0
              ? "No classes yet. Create the first one below."
              : "No classes match the search."}
          </p>
        ) : (
          <ul className="ss-lib-card-list">
            {filtered.map((a) => (
              <li key={a.uid}>
                <button
                  type="button"
                  className="ss-lib-card"
                  onClick={() => onSelectAsset?.(a.uid)}
                  title={`Edit ${a.name}`}
                >
                  <span
                    className="ss-lib-card-swatch"
                    style={{ backgroundColor: a.markColor }}
                    aria-hidden="true"
                  />
                  <span className="ss-lib-card-main">
                    <span className="ss-lib-card-name">{a.name}</span>
                    <span className="ss-lib-card-glb">
                      {glbBasename(a.modelUrl)}
                    </span>
                  </span>
                  <span className="ss-lib-card-meta">
                    {a.xSize}×{a.ySize}×{a.zSize} m
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
      <div className="ss-lib-drawer-footer">
        <button
          type="button"
          className="ss-lib-new-btn"
          onClick={() => onSelectAsset?.(null)}
        >
          + New class
        </button>
        <span className="ss-lib-hint">
          <kbd>B</kbd> toggles
        </span>
      </div>
    </aside>
  );
}
