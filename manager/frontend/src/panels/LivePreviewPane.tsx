// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useRef, useState } from "react";
import { refreshCameraStrip } from "../mqtt";
import type { SceneCameraBootstrap } from "../scene/types";
import { formatCameraFps } from "../scene/useLiveChrome";
import { useViewportStore } from "../viewport/store";
import { enterCameraVisualize } from "../viewport/visualize";
import { useVisualizeStore } from "../viewport/visualize/visualizeStore";
import "./LivePreviewPane.css";

type Props = {
  cameras: SceneCameraBootstrap[];
  cameraRates?: Record<string, string>;
  /** Preview fills the whole dock (tabs hidden). */
  maximized?: boolean;
  onMaximizedChange?: (maximized: boolean) => void;
};

/** Case-insensitive match on camera name or sensor id. */
function cameraMatchesQuery(cam: SceneCameraBootstrap, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) {
    return true;
  }
  return (
    cam.name.toLowerCase().includes(q) || cam.sensorId.toLowerCase().includes(q)
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
 * Off-screen JPEG hosts + `#live-view` so Project frame can pull MQTT frames
 * without opening the Preview dock pane (Properties stays open).
 */
export function ProjectionFeedHost({
  cameras,
  active,
}: {
  cameras: SceneCameraBootstrap[];
  active: boolean;
}) {
  useEffect(() => {
    if (!active) {
      return;
    }
    refreshCameraStrip();
    const t1 = window.setTimeout(refreshCameraStrip, 400);
    const t2 = window.setTimeout(refreshCameraStrip, 1200);
    return () => {
      window.clearTimeout(t1);
      window.clearTimeout(t2);
    };
  }, [active, cameras]);

  if (!active || cameras.length === 0) {
    return null;
  }

  return (
    <div className="ss-projection-feed" aria-hidden="true">
      <input
        type="checkbox"
        id="live-view"
        className="ss-live-preview-live-contract"
        checked
        readOnly
        tabIndex={-1}
      />
      {cameras.map((cam) => (
        <span
          key={cam.id}
          className="snapshot-image"
          data-topic={cam.cmdTopic}
          data-topic-name={`scenescape/cmd/camera/${cam.name}`}
        >
          <img
            id={`projection-feed-${cam.sensorId}`}
            data-ss-card-sensor={cam.sensorId}
            data-ss-card-name={cam.name}
            className="display-none"
            alt=""
          />
        </span>
      ))}
    </div>
  );
}

/**
 * Multi-camera Live glance pane for the single-pane right dock.
 *
 * Cards show thumbnail, name, and FPS only. Project all, Overlay, and
 * per-camera Project / Calibrate / Delete live on the Outliner Cameras group.
 */
export function LivePreviewPane({
  cameras,
  cameraRates = {},
  maximized = false,
  onMaximizedChange,
}: Props) {
  const [query, setQuery] = useState("");
  const [onlineBySensor, setOnlineBySensor] = useState<Record<string, boolean>>(
    {},
  );
  const selectedId = useViewportStore((s) => s.selectedId);
  const showTelemetry = useViewportStore((s) => s.showTelemetry);
  const liveCameraId = useVisualizeStore((state) => state.active ? state.cameraId : null);
  const rootRef = useRef<HTMLDivElement>(null);

  const availableCameras = useMemo(
    () => cameras.filter((camera) => camera.id !== liveCameraId),
    [cameras, liveCameraId],
  );
  const onlineCount = availableCameras.filter((c) => onlineBySensor[c.sensorId]).length;

  const visibleCameras = useMemo(
    () => availableCameras.filter((cam) => cameraMatchesQuery(cam, query)),
    [availableCameras, query],
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

  /** Thumbnail click opens the annotated Live visualize split. */
  const handleSelect = (cam: SceneCameraBootstrap) => {
    enterCameraVisualize({
      cameraId: cam.id,
      sensorId: cam.sensorId,
      cameraName: cam.name,
    });
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
        <span className="ss-live-preview-title">
          {onlineCount > 0 ? "Live" : "Offline"}
        </span>
        <span
          className="ss-live-preview-count"
          aria-label={
            filtering
              ? `${onlineCount} live of ${availableCameras.length} cameras, ${visibleCameras.length} shown`
              : `${onlineCount} live of ${availableCameras.length} cameras`
          }
        >
          {onlineCount}
        </span>
        <div className="ss-live-preview-head-actions">
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

      {availableCameras.length > 0 ? (
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
        {cameras.length > 0 && availableCameras.length === 0 ? (
          <p className="ss-live-preview-empty">No other cameras.</p>
        ) : null}
        {availableCameras.length > 0 && filtering && visibleCameras.length === 0 ? (
          <p className="ss-live-preview-empty">
            No cameras match “{query.trim()}”.
          </p>
        ) : null}
        {cameras.map((cam) => {
          const online = Boolean(onlineBySensor[cam.sensorId]);
          const selected = selectedId === cam.id;
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
              title={`${cam.name} — open live view`}
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
                {showTelemetry ? (
                  <span className="ss-camera-card-rate">
                    {online ? formatCameraFps(cameraRates[cam.sensorId]) : "—"}
                  </span>
                ) : null}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
