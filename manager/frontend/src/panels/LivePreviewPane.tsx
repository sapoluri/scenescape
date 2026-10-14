// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useRef, useState } from "react";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { useAppToast } from "../components/ToastProvider";
import { type RestError } from "../lib/rest";
import { restDeleteTarget } from "../lib/restDelete";
import { refreshCameraStrip } from "../mqtt";
import type { SceneCameraBootstrap } from "../scene/types";
import { enterCameraCalibrate } from "../viewport/calibrate";
import { getViewportState, useViewportStore } from "../viewport/store";
import "./LivePreviewPane.css";

type Props = {
  cameras: SceneCameraBootstrap[];
  cameraRates?: Record<string, string>;
  authToken?: string;
  isSuperuser?: boolean;
  /** Preview fills the whole dock (tabs hidden). */
  maximized?: boolean;
  onMaximizedChange?: (maximized: boolean) => void;
};

/** Case-insensitive match on camera name or sensor id. */
function cameraMatchesQuery(
  cam: SceneCameraBootstrap,
  query: string,
): boolean {
  const q = query.trim().toLowerCase();
  if (!q) {
    return true;
  }
  return (
    cam.name.toLowerCase().includes(q) ||
    cam.sensorId.toLowerCase().includes(q)
  );
}

function isLivePreview(img: HTMLImageElement | null): boolean {
  if (!img || img.classList.contains("display-none")) {
    return false;
  }
  const src = img.currentSrc || img.getAttribute("src") || "";
  if (!src || src.includes("offline.png")) {
    return false;
  }
  return img.naturalWidth > 0 || src.startsWith("data:image");
}

/**
 * Multi-camera Live glance pane for the single-pane right dock (bottom half).
 *
 * Mounted while Outliner Preview is on. `#live-view` is a hidden hard-contract
 * checkbox kept checked for MQTT; turning Outliner Preview off ends the feed.
 */
export function LivePreviewPane({
  cameras,
  cameraRates = {},
  authToken,
  isSuperuser = false,
  maximized = false,
  onMaximizedChange,
}: Props) {
  const toast = useAppToast();
  const [query, setQuery] = useState("");
  const [onlineBySensor, setOnlineBySensor] = useState<Record<string, boolean>>(
    {},
  );
  const [deleteTarget, setDeleteTarget] = useState<SceneCameraBootstrap | null>(
    null,
  );
  const [deleteBusy, setDeleteBusy] = useState(false);
  const selectedId = useViewportStore((s) => s.selectedId);
  const projectCameraIds = useViewportStore((s) => s.projectCameraIds);
  const entities = useViewportStore((s) => s.entities);
  const rootRef = useRef<HTMLDivElement>(null);

  const posedCameraIds = cameras
    .map((c) => c.id)
    .filter((id) => {
      const e = entities[id];
      return e?.type === "camera" && e.posed;
    });
  const projectAllFrames =
    posedCameraIds.length > 0 &&
    posedCameraIds.every((id) => projectCameraIds.includes(id));

  const visibleCameras = useMemo(
    () => cameras.filter((cam) => cameraMatchesQuery(cam, query)),
    [cameras, query],
  );
  const filtering = query.trim().length > 0;
  const visibleIdSet = useMemo(
    () => new Set(visibleCameras.map((c) => c.id)),
    [visibleCameras],
  );

  useEffect(() => {
    refreshCameraStrip();
    const t1 = window.setTimeout(refreshCameraStrip, 400);
    const t2 = window.setTimeout(refreshCameraStrip, 1200);
    return () => {
      window.clearTimeout(t1);
      window.clearTimeout(t2);
    };
  }, [cameras]);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) {
      return;
    }
    let raf = 0;
    const readOnline = (): Record<string, boolean> => {
      const next: Record<string, boolean> = {};
      for (const cam of cameras) {
        const img = root.querySelector<HTMLImageElement>(
          `img[data-ss-card-sensor="${CSS.escape(cam.sensorId)}"]`,
        );
        next[cam.sensorId] = isLivePreview(img);
      }
      return next;
    };
    const apply = () => {
      raf = 0;
      setOnlineBySensor((prev) => {
        const next = readOnline();
        const keys = Object.keys(next);
        if (
          keys.length === Object.keys(prev).length &&
          keys.every((k) => prev[k] === next[k])
        ) {
          return prev;
        }
        return next;
      });
    };
    const schedule = () => {
      if (!raf) {
        raf = window.requestAnimationFrame(apply);
      }
    };
    apply();
    const mo = new MutationObserver(schedule);
    mo.observe(root, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ["class", "src"],
    });
    const poll = window.setInterval(apply, 2000);
    return () => {
      mo.disconnect();
      window.clearInterval(poll);
      if (raf) {
        window.cancelAnimationFrame(raf);
      }
    };
  }, [cameras]);

  const handleSelect = (cam: SceneCameraBootstrap) => {
    getViewportState().select(cam.id);
  };

  const onConfirmDelete = async () => {
    if (!deleteTarget?.deleteUrl || !authToken) {
      return;
    }
    setDeleteBusy(true);
    try {
      const path = deleteTarget.deleteUrl.startsWith("http")
        ? new URL(deleteTarget.deleteUrl).pathname
        : deleteTarget.deleteUrl;
      const m = path.match(/\/cam\/delete\/([^/]+)\/?$/);
      if (!m) {
        throw new Error("Unrecognized camera delete URL");
      }
      await restDeleteTarget(authToken, {
        kind: "camera",
        uid: decodeURIComponent(m[1]),
      });
      getViewportState().removeEntity(deleteTarget.id);
      setDeleteTarget(null);
      toast.show(`Deleted ${deleteTarget.name}`, "ok");
      window.location.reload();
    } catch (err) {
      toast.show((err as RestError).message || "Delete failed", "bad");
      setDeleteBusy(false);
    }
  };

  return (
    <div
      ref={rootRef}
      className="ss-live-preview"
      role="region"
      aria-label="Live camera preview"
    >
      {/* Hard contract for useCameraStripMqtt — Preview on implies live. */}
      <input
        type="checkbox"
        id="live-view"
        className="ss-live-preview-live-contract"
        checked
        readOnly
        tabIndex={-1}
        aria-hidden="true"
      />
      <div className="ss-live-preview-head">
        <span className="ss-live-preview-title">Live</span>
        <span
          className="ss-live-preview-count"
          aria-label={
            filtering
              ? `${visibleCameras.length} of ${cameras.length} cameras`
              : `${cameras.length} cameras`
          }
        >
          {filtering
            ? `${visibleCameras.length}/${cameras.length}`
            : cameras.length}
        </span>
        <div className="ss-live-preview-head-actions">
          <button
            type="button"
            id="project-all-frames"
            className={`ss-live-preview-icon-btn${projectAllFrames ? " is-on" : ""}`}
            disabled={posedCameraIds.length === 0}
            title={
              posedCameraIds.length > 0
                ? projectAllFrames
                  ? "Stop projecting all cameras"
                  : "Project all calibrated cameras onto the scene map"
                : "Calibrate cameras first — projection needs a pose"
            }
            aria-label="Project all cameras"
            aria-pressed={projectAllFrames}
            onClick={() => {
              getViewportState().setProjectAllCameras(!projectAllFrames);
            }}
          >
            <i className="bi bi-cast" aria-hidden="true" />
          </button>
          {onMaximizedChange ? (
            <button
              type="button"
              className={`ss-live-preview-icon-btn${maximized ? " is-on" : ""}`}
              title={maximized ? "Restore split" : "Maximize preview"}
              aria-label={maximized ? "Restore split" : "Maximize preview"}
              aria-pressed={maximized}
              onClick={() => onMaximizedChange(!maximized)}
            >
              <i
                className={`bi ${maximized ? "bi-fullscreen-exit" : "bi-arrows-fullscreen"}`}
                aria-hidden="true"
              />
            </button>
          ) : null}
        </div>
      </div>

      {cameras.length > 0 ? (
        <div className="ss-live-preview-search-row">
          <input
            type="search"
            className="ss-live-preview-search"
            placeholder="Search cameras…"
            value={query}
            onChange={(ev) => setQuery(ev.target.value)}
            aria-label="Search cameras"
          />
        </div>
      ) : null}

      <div
        id="ss-strip-cards"
        className="ss-live-preview-grid"
        role="listbox"
        aria-label="Scene cameras"
      >
        {cameras.length === 0 && (
          <p className="ss-live-preview-empty">No cameras in this scene yet.</p>
        )}
        {cameras.length > 0 && filtering && visibleCameras.length === 0 ? (
          <p className="ss-live-preview-empty">No cameras match “{query.trim()}”.</p>
        ) : null}
        {cameras.map((cam) => {
          const online = Boolean(onlineBySensor[cam.sensorId]);
          const selected = selectedId === cam.id;
          const entity = entities[cam.id];
          const posed = entity?.type === "camera" && entity.posed;
          const projecting = projectCameraIds.includes(cam.id);
          const shown = visibleIdSet.has(cam.id);
          return (
            <div
              key={cam.id}
              role="option"
              aria-selected={selected}
              aria-hidden={!shown}
              tabIndex={shown ? 0 : -1}
              hidden={!shown}
              className={`ss-camera-card${selected ? " is-selected" : ""}`}
              onClick={() => handleSelect(cam)}
              onKeyDown={(ev) => {
                if (ev.key === "Enter" || ev.key === " ") {
                  ev.preventDefault();
                  handleSelect(cam);
                }
              }}
              title={`${cam.name} — select`}
            >
              <span className="ss-camera-card-thumb">
                <span
                  className="snapshot-image"
                  data-topic={cam.cmdTopic}
                  data-topic-name={`scenescape/cmd/camera/${cam.name}`}
                >
                  <span className="cam-offline">Offline</span>
                  <img
                    id={`live-preview-${cam.sensorId}`}
                    data-ss-card-sensor={cam.sensorId}
                    data-ss-card-name={cam.name}
                    className="display-none"
                    alt=""
                  />
                </span>
                <span
                  className={`ss-live-pill ss-live-pill-mini${online ? " is-live" : ""}`}
                >
                  <span className="ss-live-dot" aria-hidden="true" />
                  {online ? "Live" : "Offline"}
                </span>
              </span>
              <span className="ss-camera-card-foot">
                <span className="ss-camera-card-name" title={cam.name}>
                  {cam.name}
                </span>
                <span className="ss-camera-card-rate">
                  {online ? `${cameraRates[cam.sensorId] ?? "--"} fps` : "—"}
                </span>
                <button
                  type="button"
                  className={`ss-camera-card-cal${projecting ? " is-on" : ""}`}
                  disabled={!posed}
                  title={
                    posed
                      ? projecting
                        ? "Stop projecting onto the scene map"
                        : "Project live frame onto the scene map"
                      : "Calibrate this camera first — projection needs a pose"
                  }
                  aria-pressed={projecting}
                  aria-label={`Project ${cam.name}`}
                  onClick={(ev) => {
                    ev.stopPropagation();
                    getViewportState().setProjectCameraEnabled(
                      cam.id,
                      !projecting,
                    );
                  }}
                >
                  <i className="bi bi-cast" aria-hidden="true" />
                </button>
                {isSuperuser && (
                  <button
                    type="button"
                    className="ss-camera-card-cal"
                    title={`Calibrate ${cam.name}`}
                    aria-label={`Calibrate ${cam.name}`}
                    onClick={(ev) => {
                      ev.stopPropagation();
                      enterCameraCalibrate({
                        cameraId: cam.id,
                        sensorId: cam.sensorId,
                        cameraName: cam.name,
                      });
                    }}
                  >
                    <i className="bi bi-bullseye" aria-hidden="true" />
                  </button>
                )}
                {isSuperuser && cam.deleteUrl ? (
                  <button
                    type="button"
                    className="ss-camera-card-cal ss-camera-card-cal--danger"
                    title={`Delete ${cam.name}`}
                    aria-label={`Delete ${cam.name}`}
                    onClick={(ev) => {
                      ev.stopPropagation();
                      setDeleteTarget(cam);
                    }}
                  >
                    <i className="bi bi-trash" aria-hidden="true" />
                  </button>
                ) : null}
              </span>
            </div>
          );
        })}
      </div>

      <ConfirmDialog
        open={Boolean(deleteTarget)}
        title="Delete camera?"
        confirmLabel="Delete"
        danger
        busy={deleteBusy}
        onConfirm={() => void onConfirmDelete()}
        onCancel={() => {
          if (!deleteBusy) {
            setDeleteTarget(null);
          }
        }}
      >
        <p>
          Are you sure you want to delete{" "}
          <strong>{deleteTarget?.name}</strong>?
        </p>
        <p>This action cannot be undone.</p>
      </ConfirmDialog>
    </div>
  );
}
