// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from "react";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { useAppToast } from "../components/ToastProvider";
import { type RestError } from "../lib/rest";
import { restDeleteTarget } from "../lib/restDelete";
import { refreshCameraStrip } from "../mqtt";
import type { SceneCameraBootstrap } from "../scene/types";
import { enterCameraCalibrate } from "../viewport/calibrate";
import { getViewportState, useViewportStore } from "../viewport/store";
import "../scene/CameraStrip.css";
import "./CameraStripOverlay.css";

type Props = {
  cameras: SceneCameraBootstrap[];
  cameraRates?: Record<string, string>;
  /**
   * Kept for API parity with CamerasPanelContent. The strip itself needs no
   * token: live frames arrive through the page-level camera-strip MQTT hook
   * (useCameraStripMqtt), which feeds every img[data-ss-card-sensor].
   */
  authToken?: string;
  /** Gate the calibration launcher (mirrors the side-panel cards). */
  isSuperuser?: boolean;
  /** Called with the viewport entity id (=== SceneCameraBootstrap.id). */
  onSelectCamera?: (id: string) => void;
};

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
 * Floating camera panel for the single-pane 3D viewport.
 *
 * Compositor-style floating panel, bottom-left of the viewport: a header
 * ("Cameras" + count + Live View toggle + collapse), a Rerun-style live
 * preview viewer for the selected camera, and a horizontal strip of cards.
 *
 * `#live-view` is the hard-contract checkbox that `useCameraStripMqtt` /
 * `sscape.js` use to gate continuous `getimage` polling. Live frames still
 * land on `img[data-ss-card-sensor]` / `.snapshot-image` anchors.
 *
 * Calibrate opens Phase 2.3 in-viewport split via `enterCameraCalibrate`
 * (3D + live feed), not the legacy `?ss=calibrate-cam` sheet.
 *
 * Mount inside the viewport container element:
 *
 *   <div className="ss-viewport" ...>
 *     ...
 *     <CameraStripOverlay cameras={cameras} cameraRates={cameraRates} />
 *   </div>
 */
export function CameraStripOverlay({
  cameras,
  cameraRates = {},
  authToken,
  isSuperuser = false,
  onSelectCamera,
}: Props) {
  const toast = useAppToast();
  const [collapsed, setCollapsed] = useState(false);
  const [liveView, setLiveView] = useState(false);
  const [previewId, setPreviewId] = useState<string | null>(null);
  const [onlineBySensor, setOnlineBySensor] = useState<Record<string, boolean>>(
    {},
  );
  const [deleteTarget, setDeleteTarget] = useState<SceneCameraBootstrap | null>(
    null,
  );
  const [deleteBusy, setDeleteBusy] = useState(false);
  const selectedId = useViewportStore((s) => s.selectedId);
  const projectCameraId = useViewportStore((s) => s.projectCameraId);
  const entities = useViewportStore((s) => s.entities);
  const stripRef = useRef<HTMLDivElement>(null);

  const onLiveViewChange = (checked: boolean) => {
    setLiveView(checked);
    if (checked) {
      setCollapsed(false);
      refreshCameraStrip();
    }
  };

  // Drop the preview if its camera leaves the scene.
  useEffect(() => {
    if (previewId && !cameras.some((c) => c.id === previewId)) {
      setPreviewId(null);
    }
  }, [cameras, previewId]);

  // Project needs JPEG frames — turn Live View on and open the preview.
  useEffect(() => {
    if (!projectCameraId) {
      return;
    }
    setCollapsed(false);
    setPreviewId(projectCameraId);
    setLiveView(true);
    refreshCameraStrip();
  }, [projectCameraId]);

  // Request MQTT frames for the strip anchors (mirrors CamerasPanelContent).
  useEffect(() => {
    refreshCameraStrip();
    const t1 = window.setTimeout(refreshCameraStrip, 400);
    const t2 = window.setTimeout(refreshCameraStrip, 1200);
    return () => {
      window.clearTimeout(t1);
      window.clearTimeout(t2);
    };
  }, [cameras]);

  // Track live/offline per camera by observing img src/class changes,
  // scoped to this panel (mirrors CameraStripEnhancer, React-driven).
  useEffect(() => {
    const root = stripRef.current;
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
    onSelectCamera?.(cam.id);
    setPreviewId(cam.id);
  };

  const preview = cameras.find((c) => c.id === previewId) ?? null;
  const previewOnline = preview
    ? Boolean(onlineBySensor[preview.sensorId])
    : false;
  const previewEntity = preview ? entities[preview.id] : undefined;
  const previewPosed =
    previewEntity?.type === "camera" && previewEntity.posed;
  const previewProjecting = Boolean(
    preview && projectCameraId === preview.id,
  );

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
      getViewportState().setProjectCamera(null);
      setPreviewId((id) => (id === deleteTarget.id ? null : id));
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
      ref={stripRef}
      className={`ss-camera-panel${collapsed ? " is-collapsed" : ""}`}
      role="region"
      aria-label="Cameras"
    >
      <div className="ss-camera-panel-head">
        <button
          type="button"
          className="ss-camera-panel-toggle"
          aria-expanded={!collapsed}
          aria-controls="ss-camera-panel-body"
          onClick={() => setCollapsed((v) => !v)}
          title={collapsed ? "Expand cameras" : "Collapse cameras"}
        >
          <i
            className={`bi ${collapsed ? "bi-chevron-up" : "bi-chevron-down"}`}
            aria-hidden="true"
          />
          <span>Cameras</span>
          <span className="ss-camera-panel-count" aria-label={`${cameras.length} cameras`}>
            {cameras.length}
          </span>
        </button>
        <label
          className={`ss-camera-live-toggle${liveView ? " is-on" : ""}`}
          title="Continuously refresh camera JPEG previews over MQTT"
        >
          <input
            type="checkbox"
            id="live-view"
            className="ss-camera-live-input"
            checked={liveView}
            onChange={(ev) => onLiveViewChange(ev.target.checked)}
            aria-labelledby="live-view-label"
          />
          <span id="live-view-label" className="ss-camera-live-label">
            Live View
          </span>
        </label>
      </div>

      {!collapsed && (
        <div id="ss-camera-panel-body" className="ss-camera-panel-body">
          {preview && (
            <section
              className="ss-camera-viewer"
              aria-label={`${preview.name} live preview`}
            >
              <header className="ss-camera-viewer-bar">
                <span
                  className={`ss-live-pill${previewOnline ? " is-live" : ""}`}
                >
                  <span className="ss-live-dot" aria-hidden="true" />
                  {previewOnline ? "Live" : "Offline"}
                </span>
                <span className="ss-camera-viewer-name" title={preview.name}>
                  {preview.name}
                </span>
                <span className="ss-camera-viewer-rate">
                  {previewOnline
                    ? `${cameraRates[preview.sensorId] ?? "--"} fps`
                    : "—"}
                </span>
                <span className="ss-camera-viewer-actions">
                  <button
                    type="button"
                    className={`ss-camera-project${previewProjecting ? " is-on" : ""}`}
                    disabled={!previewPosed}
                    title={
                      previewPosed
                        ? previewProjecting
                          ? "Stop projecting frame onto the scene map"
                          : "Project live frame onto the scene map"
                        : "Calibrate this camera first — projection needs a pose"
                    }
                    aria-pressed={previewProjecting}
                    onClick={(ev) => {
                      ev.stopPropagation();
                      getViewportState().setProjectCamera(
                        previewProjecting ? null : preview.id,
                      );
                    }}
                  >
                    <i className="bi bi-cast" aria-hidden="true" />
                    <span>Project</span>
                  </button>
                  {isSuperuser && (
                    <button
                      type="button"
                      className="ss-camera-calibrate"
                      title={`Calibrate ${preview.name}`}
                      onClick={(ev) => {
                        ev.stopPropagation();
                        enterCameraCalibrate({
                          cameraId: preview.id,
                          sensorId: preview.sensorId,
                          cameraName: preview.name,
                        });
                      }}
                    >
                      <i className="bi bi-crosshair" aria-hidden="true" />
                      <span>Calibrate</span>
                    </button>
                  )}
                  {isSuperuser && preview.deleteUrl ? (
                    <button
                      type="button"
                      className="ss-camera-delete"
                      title={`Delete ${preview.name}`}
                      aria-label={`Delete ${preview.name}`}
                      onClick={(ev) => {
                        ev.stopPropagation();
                        setDeleteTarget(preview);
                      }}
                    >
                      <i className="bi bi-trash" aria-hidden="true" />
                    </button>
                  ) : null}
                  <button
                    type="button"
                    className="ss-camera-viewer-close"
                    aria-label="Close preview"
                    onClick={() => setPreviewId(null)}
                  >
                    <i className="bi bi-x-lg" aria-hidden="true" />
                  </button>
                </span>
              </header>
              <div className="ss-camera-viewer-feed">
                <span
                  className="snapshot-image"
                  data-topic={preview.cmdTopic}
                  data-topic-name={`scenescape/cmd/camera/${preview.name}`}
                >
                  <span className="cam-offline">Camera Offline</span>
                  <img
                    id={`strip-overlay-viewer-${preview.sensorId}`}
                    data-ss-card-sensor={preview.sensorId}
                    data-ss-card-name={preview.name}
                    className="display-none"
                    alt={`${preview.name} live view`}
                  />
                </span>
              </div>
            </section>
          )}

          <div
            id="ss-strip-cards"
            className="ss-camera-cards"
            role="listbox"
            aria-label="Scene cameras"
            aria-orientation="horizontal"
          >
            {cameras.length === 0 && (
              <p className="ss-camera-empty">No cameras in this scene yet.</p>
            )}
            {cameras.map((cam) => {
              const online = Boolean(onlineBySensor[cam.sensorId]);
              const selected = selectedId === cam.id;
              const previewing = previewId === cam.id;
              return (
                <div
                  key={cam.id}
                  role="option"
                  aria-selected={selected}
                  tabIndex={0}
                  className={`ss-camera-card${selected ? " is-selected" : ""}${
                    previewing ? " is-previewing" : ""
                  }`}
                  onClick={() => handleSelect(cam)}
                  onKeyDown={(ev) => {
                    if (ev.key === "Enter" || ev.key === " ") {
                      ev.preventDefault();
                      handleSelect(cam);
                    }
                  }}
                  title={`${cam.name} — select and preview`}
                >
                  <span className="ss-camera-card-thumb">
                    <span
                      className="snapshot-image"
                      data-topic={cam.cmdTopic}
                      data-topic-name={`scenescape/cmd/camera/${cam.name}`}
                    >
                      <span className="cam-offline">Offline</span>
                      <img
                        id={`strip-overlay-preview-${cam.sensorId}`}
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
                        <i className="bi bi-crosshair" aria-hidden="true" />
                      </button>
                    )}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
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
