// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Open Scene dialog — the Phase 1 same-page scene browser.
 *
 * Compositor-style floating gallery: search, per-scene stat cards, and a
 * New Scene card. Clicking a card calls `onOpenScene(uid)`; the parent owns
 * navigation (the SPA never does a full page load here). Multi-scene tabs
 * are Phase 3 and intentionally not built.
 *
 * Mount API:
 *   const [open, setOpen] = useState(false);
 *   useOpenSceneHotkey(() => setOpen(true));   // ⌘O / Ctrl+O
 *   <OpenSceneDialog
 *     open={open}
 *     onClose={() => setOpen(false)}
 *     onOpenScene={(uid) => { setOpen(false); navigateToScene(uid); }}
 *     authToken={token}
 *   />
 */

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import { createPortal } from "react-dom";
import { api } from "../lib/rest";
import "../tokens/tokens.css";
import "./OpenSceneDialog.css";

/** Minimal shape of GET /api/v1/scenes (SceneSerializer, read side). */
export type SceneSummary = {
  uid: string;
  name: string;
  cameras?: unknown[];
  sensors?: unknown[];
  regions?: unknown[];
  tripwires?: unknown[];
  map_processed?: string | null;
};

export type OpenSceneDialogProps = {
  open: boolean;
  onClose: () => void;
  onOpenScene: (sceneId: string) => void;
  authToken: string;
};

/** Case-insensitive name filter. Pure — unit tested. */
export function filterScenes(
  scenes: SceneSummary[],
  query: string,
): SceneSummary[] {
  const q = query.trim().toLowerCase();
  if (!q) {
    return scenes;
  }
  return scenes.filter((s) => s.name.toLowerCase().includes(q));
}

/** Counts available from the list serializer. Pure — unit tested. */
export function sceneStats(scene: SceneSummary): {
  cameras: number;
  sensors: number;
  rois: number;
} {
  return {
    cameras: Array.isArray(scene.cameras) ? scene.cameras.length : 0,
    sensors: Array.isArray(scene.sensors) ? scene.sensors.length : 0,
    rois:
      (Array.isArray(scene.regions) ? scene.regions.length : 0) +
      (Array.isArray(scene.tripwires) ? scene.tripwires.length : 0),
  };
}

/** Human "modified" label from map_processed. Pure — unit tested. */
export function formatModified(iso: string | null | undefined): string {
  if (!iso) {
    return "—";
  }
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) {
    return "—";
  }
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

/**
 * Global ⌘O / Ctrl+O listener. Ignores keystrokes inside editable fields.
 * The parent passes its own open setter.
 */
export function useOpenSceneHotkey(onRequestOpen: () => void): void {
  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      const mod = ev.metaKey || ev.ctrlKey;
      if (!mod || ev.key.toLowerCase() !== "o") {
        return;
      }
      const t = ev.target as HTMLElement | null;
      if (
        t &&
        (t.tagName === "INPUT" ||
          t.tagName === "TEXTAREA" ||
          t.isContentEditable)
      ) {
        return;
      }
      ev.preventDefault();
      onRequestOpen();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onRequestOpen]);
}

function normalizeScenes(payload: unknown): SceneSummary[] {
  const list = Array.isArray(payload)
    ? payload
    : (payload as { results?: unknown[] } | null)?.results;
  if (!Array.isArray(list)) {
    return [];
  }
  return list
    .filter(
      (row): row is Record<string, unknown> =>
        !!row && typeof row === "object",
    )
    .map((row) => ({
      uid: String(row.uid ?? ""),
      name: String(row.name ?? "Untitled scene"),
      cameras: row.cameras as unknown[] | undefined,
      sensors: row.sensors as unknown[] | undefined,
      regions: row.regions as unknown[] | undefined,
      tripwires: row.tripwires as unknown[] | undefined,
      map_processed: (row.map_processed as string | null) ?? null,
    }))
    .filter((s) => s.uid.length > 0);
}

function SceneCard({
  scene,
  onOpen,
}: {
  scene: SceneSummary;
  onOpen: (uid: string) => void;
}) {
  const stats = sceneStats(scene);
  // NOTE: the read SceneSerializer exposes no thumbnail; render a
  // stylized placeholder until a thumbnail field exists server-side.
  const initials = scene.name
    .split(/\s+/)
    .slice(0, 2)
    .map((w) => w[0] ?? "")
    .join("")
    .toUpperCase();
  return (
    <button
      type="button"
      className="ss-open-scene-card"
      onClick={() => onOpen(scene.uid)}
      aria-label={`Open scene ${scene.name}`}
    >
      <span className="ss-open-scene-thumb" aria-hidden="true">
        <span className="ss-open-scene-thumb-initials">{initials}</span>
      </span>
      <span className="ss-open-scene-card-body">
        <span className="ss-open-scene-name">{scene.name}</span>
        <span className="ss-open-scene-stats">
          {stats.cameras} cameras · {stats.sensors} sensors · {stats.rois}{" "}
          ROIs
        </span>
        <span className="ss-open-scene-modified">
          Modified {formatModified(scene.map_processed)}
        </span>
      </span>
    </button>
  );
}

export function OpenSceneDialog({
  open,
  onClose,
  onOpenScene,
  authToken,
}: OpenSceneDialogProps) {
  const [scenes, setScenes] = useState<SceneSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [createError, setCreateError] = useState<string | null>(null);
  const [createBusy, setCreateBusy] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) {
      return;
    }
    setQuery("");
    setCreating(false);
    setCreateError(null);
    setError(null);
    setLoading(true);
    let cancelled = false;
    api
      .getScenes(authToken)
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setScenes(normalizeScenes(payload));
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (cancelled) {
          return;
        }
        setError(err instanceof Error ? err.message : "Failed to load scenes");
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, authToken]);

  useEffect(() => {
    if (open) {
      // Autofocus the search field once the dialog paints.
      const t = window.setTimeout(() => searchRef.current?.focus(), 0);
      return () => window.clearTimeout(t);
    }
    return undefined;
  }, [open ]);

  const onDialogKeyDown = useCallback(
    (ev: ReactKeyboardEvent) => {
      if (ev.key === "Escape") {
        ev.stopPropagation();
        onClose();
        return;
      }
      // Minimal focus trap: keep Tab cycling inside the dialog.
      if (ev.key === "Tab" && dialogRef.current) {
        const focusables = Array.from(
          dialogRef.current.querySelectorAll<HTMLElement>(
            'button, input, [tabindex]:not([tabindex="-1"])',
          ),
        ).filter((el) => !el.hasAttribute("disabled"));
        if (focusables.length === 0) {
          return;
        }
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        const active = document.activeElement as HTMLElement | null;
        if (ev.shiftKey && active === first) {
          ev.preventDefault();
          last.focus();
        } else if (!ev.shiftKey && active === last) {
          ev.preventDefault();
          first.focus();
        }
      }
    },
    [onClose],
  );

  const createScene = useCallback(async () => {
    const name = newName.trim();
    if (!name || createBusy) {
      return;
    }
    setCreateBusy(true);
    setCreateError(null);
    try {
      const form = new FormData();
      form.append("name", name);
      const created = (await api.createScene(authToken, form)) as {
        uid?: unknown;
      } | null;
      const uid = created && typeof created.uid === "string" ? created.uid : "";
      if (!uid) {
        throw new Error("Server did not return a scene id");
      }
      onOpenScene(uid);
    } catch (err: unknown) {
      setCreateError(
        err instanceof Error ? err.message : "Failed to create scene",
      );
    } finally {
      setCreateBusy(false);
    }
  }, [newName, createBusy, authToken, onOpenScene]);

  if (typeof document === "undefined" || !open) {
    return null;
  }

  const visible = filterScenes(scenes, query);

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
        className="ss-open-scene-dialog"
        role="dialog"
        aria-modal="true"
        aria-label="Open scene"
        onKeyDown={onDialogKeyDown}
      >
        <div className="ss-open-scene-header">
          <h2 className="ss-open-scene-title">Open scene</h2>
          <button
            type="button"
            className="ss-open-scene-close"
            onClick={onClose}
            aria-label="Close"
          >
            ×
          </button>
        </div>
        <div className="ss-open-scene-search-row">
          <input
            ref={searchRef}
            type="search"
            className="ss-open-scene-search"
            placeholder="Search scenes…"
            value={query}
            onChange={(ev) => setQuery(ev.target.value)}
            aria-label="Search scenes"
          />
          <span className="ss-open-scene-hint" aria-hidden="true">
            ⌘O
          </span>
        </div>
        {loading ? (
          <p className="ss-open-scene-status">Loading scenes…</p>
        ) : error ? (
          <p className="ss-open-scene-status ss-open-scene-error" role="alert">
            {error}
          </p>
        ) : (
          <div className="ss-open-scene-grid" role="list">
            <div className="ss-open-scene-new-wrap" role="listitem">
              {creating ? (
                <form
                  className="ss-open-scene-new-form"
                  onSubmit={(ev) => {
                    ev.preventDefault();
                    void createScene();
                  }}
                >
                  <input
                    type="text"
                    className="ss-open-scene-search"
                    placeholder="Scene name"
                    value={newName}
                    autoFocus
                    onChange={(ev) => setNewName(ev.target.value)}
                    aria-label="New scene name"
                  />
                  <div className="ss-open-scene-new-actions">
                    <button
                      type="submit"
                      className="ss-open-scene-btn-primary"
                      disabled={createBusy || !newName.trim()}
                    >
                      {createBusy ? "Creating…" : "Create"}
                    </button>
                    <button
                      type="button"
                      className="ss-open-scene-btn-ghost"
                      onClick={() => {
                        setCreating(false);
                        setNewName("");
                        setCreateError(null);
                      }}
                    >
                      Cancel
                    </button>
                  </div>
                  {createError ? (
                    <p
                      className="ss-open-scene-status ss-open-scene-error"
                      role="alert"
                    >
                      {createError}
                    </p>
                  ) : null}
                </form>
              ) : (
                <button
                  type="button"
                  className="ss-open-scene-card ss-open-scene-new"
                  onClick={() => setCreating(true)}
                  aria-label="New scene"
                >
                  <span
                    className="ss-open-scene-thumb ss-open-scene-thumb-new"
                    aria-hidden="true"
                  >
                    <span className="ss-open-scene-plus">+</span>
                  </span>
                  <span className="ss-open-scene-card-body">
                    <span className="ss-open-scene-name">New scene</span>
                    <span className="ss-open-scene-stats">
                      Start from a blank scene
                    </span>
                  </span>
                </button>
              )}
            </div>
            {visible.map((scene) => (
              <div key={scene.uid} role="listitem">
                <SceneCard scene={scene} onOpen={onOpenScene} />
              </div>
            ))}
          </div>
        )}
        {!loading && !error && visible.length === 0 && scenes.length > 0 ? (
          <p className="ss-open-scene-status">
            No scenes match “{query.trim()}”.
          </p>
        ) : null}
      </div>
    </div>
  );

  return createPortal(node, document.body);
}
