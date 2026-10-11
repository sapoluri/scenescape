// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Open Scene dialog — floating gallery over the 3D viewport.
 *
 * Home (`/`) opens this by default over an empty scene. In a workspace,
 * ⌘O / Open opens the same gallery. Opening a scene always full-navigates
 * to `/{uid}/`. Create / import / manage use parent sheets or `/?ss=…`.
 */

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import { createPortal } from "react-dom";
import { ACTION_ICONS } from "../components/actionIcons";
import { api } from "../lib/rest";
import { sceneMapBitmapUrl } from "../lib/sceneMapBitmap";
import "../tokens/tokens.css";
import "../components/Button.css";
import "./OpenSceneDialog.css";

/** Card shape used by the gallery (home bootstrap or REST-normalized). */
export type GalleryScene = {
  uid: string;
  name: string;
  detailUrl: string;
  manageUrl: string;
  deleteUrl: string | null;
  thumbnailUrl?: string | null;
  mapUrl?: string | null;
  counts: {
    sensors: number;
    regions: number;
    tripwires: number;
  };
  mapProcessed?: string | null;
};

/** Minimal shape of GET /api/v1/scenes (SceneSerializer, read side). */
export type SceneSummary = {
  uid: string;
  name: string;
  cameras?: unknown[];
  sensors?: unknown[];
  regions?: unknown[];
  tripwires?: unknown[];
  map_processed?: string | null;
  thumbnail?: string | null;
  map?: string | null;
};

export type OpenSceneDialogProps = {
  open: boolean;
  onClose: () => void;
  onOpenScene: (sceneId: string) => void;
  authToken: string;
  isSuperuser?: boolean;
  /** Prefetched cards (scenes-home bootstrap). */
  initialScenes?: GalleryScene[];
  /** Full SceneSheet create (home) or navigate to `/?ss=scene-create`. */
  onNewScene?: () => void;
  /** Import dialog (home) or navigate to `/?ss=scene-import`. */
  onImportScene?: () => void;
};

/** Case-insensitive name filter. Pure — unit tested. */
export function filterScenes<T extends { name: string }>(
  scenes: T[],
  query: string,
): T[] {
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

export function gallerySceneFromSummary(row: SceneSummary): GalleryScene {
  const stats = sceneStats(row);
  return {
    uid: row.uid,
    name: row.name,
    detailUrl: `/${row.uid}/`,
    manageUrl: `/?ss=scene-manage&id=${encodeURIComponent(row.uid)}`,
    deleteUrl: null,
    thumbnailUrl: row.thumbnail ?? null,
    mapUrl: row.map ?? null,
    counts: {
      sensors: stats.cameras + stats.sensors,
      regions: Array.isArray(row.regions) ? row.regions.length : 0,
      tripwires: Array.isArray(row.tripwires) ? row.tripwires.length : 0,
    },
    mapProcessed: row.map_processed ?? null,
  };
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

function normalizeScenes(payload: unknown): GalleryScene[] {
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
    .map((row) =>
      gallerySceneFromSummary({
        uid: String(row.uid ?? ""),
        name: String(row.name ?? "Untitled scene"),
        cameras: row.cameras as unknown[] | undefined,
        sensors: row.sensors as unknown[] | undefined,
        regions: row.regions as unknown[] | undefined,
        tripwires: row.tripwires as unknown[] | undefined,
        map_processed: (row.map_processed as string | null) ?? null,
        thumbnail:
          typeof row.thumbnail === "string" ? row.thumbnail : null,
        map: typeof row.map === "string" ? row.map : null,
      }),
    )
    .filter((s) => s.uid.length > 0);
}

function SceneCard({
  scene,
  isSuperuser,
  onOpen,
}: {
  scene: GalleryScene;
  isSuperuser: boolean;
  onOpen: (uid: string) => void;
}) {
  const thumb = sceneMapBitmapUrl({
    thumbnailUrl: scene.thumbnailUrl ?? null,
    mapUrl: scene.mapUrl ?? null,
  });
  const initials = scene.name
    .split(/\s+/)
    .slice(0, 2)
    .map((w) => w[0] ?? "")
    .join("")
    .toUpperCase();
  const openScene = (ev: { preventDefault: () => void }) => {
    ev.preventDefault();
    onOpen(scene.uid);
  };

  return (
    <div
      className="card scene-card ss-scene-card ss-open-scene-card"
      id={`scene_id_${scene.uid}`}
      data-scene-id={scene.uid}
      ref={(el) => {
        if (el) {
          el.setAttribute("name", scene.name);
        }
      }}
    >
      <div className="ss-open-scene-media">
        <a
          id={`scene-manage-${scene.name}`}
          href={scene.detailUrl}
          className="ss-open-scene-thumb-link"
          onClick={openScene}
          aria-label={`Open scene ${scene.name}`}
        >
          <span
            className={`ss-open-scene-thumb${thumb ? " has-image" : ""}`}
            aria-hidden="true"
          >
            {thumb ? (
              <img
                className="ss-open-scene-thumb-img"
                src={thumb}
                alt=""
                loading="lazy"
                decoding="async"
              />
            ) : (
              <span className="ss-open-scene-thumb-initials">{initials}</span>
            )}
          </span>
        </a>
        {isSuperuser ? (
          <div
            className="ss-open-scene-card-tools scene-card-actions ss-scene-card-actions"
            onClick={(ev) => ev.stopPropagation()}
          >
            <a
              className="ss-icon-btn"
              id={`scene-edit-${scene.uid}`}
              href={scene.manageUrl}
              title={`Edit ${scene.name} Scene Details`}
              aria-label={`Edit ${scene.name} Scene Details`}
            >
              <i className={`bi ${ACTION_ICONS.edit}`} aria-hidden="true" />
            </a>
            {scene.deleteUrl ? (
              <a
                className="ss-icon-btn ss-icon-btn--danger"
                id={`scene-delete-${scene.uid}`}
                href={scene.deleteUrl}
                title={`Delete ${scene.name} Scene`}
                aria-label={`Delete ${scene.name} Scene`}
              >
                <i className={`bi ${ACTION_ICONS.delete}`} aria-hidden="true" />
              </a>
            ) : null}
          </div>
        ) : null}
      </div>
      <a
        href={scene.detailUrl}
        className="ss-open-scene-card-body"
        onClick={openScene}
      >
        <h5 className="card-header ss-open-scene-name">{scene.name}</h5>
        <span className="ss-open-scene-stats">
          <span className="sensor-count">{scene.counts.sensors}</span> sensors
          {" · "}
          <span className="region-count">{scene.counts.regions}</span> regions
          {" · "}
          <span className="tripwire-count">{scene.counts.tripwires}</span>{" "}
          tripwires
        </span>
        <span className="ss-open-scene-modified">
          Modified {formatModified(scene.mapProcessed)}
        </span>
      </a>
    </div>
  );
}

export function OpenSceneDialog({
  open,
  onClose,
  onOpenScene,
  authToken,
  isSuperuser = false,
  initialScenes,
  onNewScene,
  onImportScene,
}: OpenSceneDialogProps) {
  const [scenes, setScenes] = useState<GalleryScene[]>(initialScenes ?? []);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const searchRef = useRef<HTMLInputElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) {
      return;
    }
    setQuery("");
    setError(null);
    if (initialScenes) {
      setScenes(initialScenes);
    }
    // Refresh from REST so ⌘O stays current; home also benefits after
    // create/import when bootstrap was stale.
    setLoading(!(initialScenes && initialScenes.length > 0));
    let cancelled = false;
    api
      .getScenes(authToken)
      .then((payload) => {
        if (cancelled) {
          return;
        }
        const fromApi = normalizeScenes(payload);
        if (initialScenes?.length) {
          const byId = new Map(initialScenes.map((s) => [s.uid, s]));
          setScenes(
            fromApi.map((s) => {
              const boot = byId.get(s.uid);
              if (!boot) {
                return {
                  ...s,
                  manageUrl: `/?ss=scene-manage&id=${encodeURIComponent(s.uid)}`,
                  deleteUrl: isSuperuser
                    ? `/scene/delete/${s.uid}/`
                    : null,
                };
              }
              return {
                ...s,
                detailUrl: boot.detailUrl || s.detailUrl,
                manageUrl: boot.manageUrl || s.manageUrl,
                deleteUrl: boot.deleteUrl,
                counts:
                  boot.counts.sensors ||
                  boot.counts.regions ||
                  boot.counts.tripwires
                    ? boot.counts
                    : s.counts,
              };
            }),
          );
        } else {
          setScenes(
            fromApi.map((s) => ({
              ...s,
              manageUrl: `/?ss=scene-manage&id=${encodeURIComponent(s.uid)}`,
              deleteUrl: isSuperuser ? `/scene/delete/${s.uid}/` : null,
            })),
          );
        }
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (cancelled) {
          return;
        }
        if (!(initialScenes && initialScenes.length > 0)) {
          setError(
            err instanceof Error ? err.message : "Failed to load scenes",
          );
        }
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, authToken, initialScenes, isSuperuser]);

  useEffect(() => {
    if (open) {
      const t = window.setTimeout(() => searchRef.current?.focus(), 0);
      return () => window.clearTimeout(t);
    }
    return undefined;
  }, [open]);

  const onDialogKeyDown = useCallback(
    (ev: ReactKeyboardEvent) => {
      if (ev.key === "Escape") {
        ev.stopPropagation();
        onClose();
        return;
      }
      if (ev.key === "Tab" && dialogRef.current) {
        const focusables = Array.from(
          dialogRef.current.querySelectorAll<HTMLElement>(
            'button, a[href], input, [tabindex]:not([tabindex="-1"])',
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

  const handleNew = useCallback(() => {
    if (onNewScene) {
      onNewScene();
      return;
    }
    window.location.assign("/?ss=scene-create");
  }, [onNewScene]);

  const handleImport = useCallback(() => {
    if (onImportScene) {
      onImportScene();
      return;
    }
    window.location.assign("/?ss=scene-import");
  }, [onImportScene]);

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
          <div className="ss-open-scene-header-actions">
            {isSuperuser ? (
              <button
                type="button"
                id="import-scene"
                className="ss-btn ss-btn--secondary"
                onClick={handleImport}
              >
                Import Scene
              </button>
            ) : null}
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
        ) : scenes.length === 0 && !isSuperuser ? (
          <p className="ss-open-scene-status scene-gallery-empty ss-scene-gallery-empty">
            No scenes are available. Ask an administrator to add one.
          </p>
        ) : (
          <div
            className="ss-open-scene-grid scene-gallery ss-scene-gallery"
            role="list"
          >
            {isSuperuser ? (
              <div className="ss-open-scene-new-wrap" role="listitem">
                <button
                  type="button"
                  id="new_scene"
                  className="ss-open-scene-card ss-open-scene-new"
                  onClick={handleNew}
                  aria-label="New scene"
                >
                  <span className="ss-open-scene-media" aria-hidden="true">
                    <span className="ss-open-scene-thumb ss-open-scene-thumb-new">
                      <span className="ss-open-scene-plus">+</span>
                    </span>
                  </span>
                  <span className="ss-open-scene-card-body">
                    <span className="ss-open-scene-name">New scene</span>
                    <span className="ss-open-scene-stats">
                      Start from a blank scene
                    </span>
                  </span>
                </button>
              </div>
            ) : null}
            {visible.map((scene) => (
              <div key={scene.uid} role="listitem">
                <SceneCard
                  scene={scene}
                  isSuperuser={isSuperuser}
                  onOpen={onOpenScene}
                />
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
