// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from "react";
import { refreshCameraStrip } from "../mqtt";
import { getViewportState, useViewportStore } from "../viewport/store";
import type { SceneCameraBootstrap } from "../scene/types";
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
 * Floating camera strip for the single-pane 3D viewport (Phase 1.3).
 *
 * Bottom-left overlay: camera cards in a horizontal strip, reusing the
 * side-panel card structure so the existing MQTT frame machinery
 * (refreshCameraStrip / applyCameraFrame, keyed on
 * img[data-ss-card-sensor] and .snapshot-image anchors) feeds it without
 * changes. Clicking a card selects the camera entity in the viewport store
 * (same selection model as viewport clicks and the Outliner) and expands
 * an inline live-feed preview in place.
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
  onSelectCamera,
}: Props) {
  const [collapsed, setCollapsed] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [onlineBySensor, setOnlineBySensor] = useState<Record<string, boolean>>(
    {},
  );
  const selectedId = useViewportStore((s) => s.selectedId);
  const stripRef = useRef<HTMLDivElement>(null);

  // Drop the expanded preview if its camera leaves the scene.
  useEffect(() => {
    if (expandedId && !cameras.some((c) => c.id === expandedId)) {
      setExpandedId(null);
    }
  }, [cameras, expandedId]);

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
  // scoped to this strip (mirrors CameraStripEnhancer, React-driven).
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
    setExpandedId((prev) => (prev === cam.id ? null : cam.id));
  };

  const expanded = cameras.find((c) => c.id === expandedId) ?? null;
  const expandedOnline = expanded
    ? Boolean(onlineBySensor[expanded.sensorId])
    : false;

  return (
    <div
      ref={stripRef}
      className={`ss-camera-strip-overlay ss-camera-strip${
        collapsed ? " is-collapsed" : ""
      }`}
      role="region"
      aria-label="Camera strip"
    >
      <div className="ss-strip-overlay-head">
        <button
          type="button"
          className="ss-strip-collapse"
          aria-expanded={!collapsed}
          aria-controls="ss-strip-cards"
          onClick={() => setCollapsed((v) => !v)}
          title={collapsed ? "Expand camera strip" : "Collapse camera strip"}
        >
          <i
            className={`bi ${collapsed ? "bi-chevron-up" : "bi-chevron-down"}`}
            aria-hidden="true"
          />
          <span>Cameras</span>
          <span className="ss-strip-count" aria-label={`${cameras.length} cameras`}>
            {cameras.length}
          </span>
        </button>
      </div>

      {!collapsed && (
        <>
          {expanded && (
            <div
              className="ss-strip-expanded"
              aria-label={`${expanded.name} live feed`}
            >
              <span
                className="snapshot-image"
                data-topic={expanded.cmdTopic}
                data-topic-name={`scenescape/cmd/camera/${expanded.name}`}
              >
                <span className="cam-offline">Camera Offline</span>
                <img
                  id={`strip-overlay-expanded-${expanded.sensorId}`}
                  data-ss-card-sensor={expanded.sensorId}
                  data-ss-card-name={expanded.name}
                  className="display-none"
                  alt={`${expanded.name} live view`}
                />
              </span>
              <span className="ss-strip-expanded-meta">
                <span className="ss-strip-expanded-name">{expanded.name}</span>
                <span
                  className={`ss-camera-strip-badge ${
                    expandedOnline ? "is-online" : "is-offline"
                  }`}
                >
                  {expandedOnline ? "Live" : "Offline"}
                </span>
                <span className="rate">
                  {expandedOnline
                    ? (cameraRates[expanded.sensorId] ?? "--")
                    : "--"}
                </span>
              </span>
            </div>
          )}

          <div
            id="ss-strip-cards"
            className="ss-strip-cards"
            role="listbox"
            aria-label="Scene cameras"
            aria-orientation="horizontal"
          >
            {cameras.length === 0 && (
              <p className="ss-strip-empty">No cameras in this scene yet.</p>
            )}
            {cameras.map((cam) => {
              const online = Boolean(onlineBySensor[cam.sensorId]);
              const selected = selectedId === cam.id;
              return (
                <button
                  key={cam.id}
                  type="button"
                  role="option"
                  aria-selected={selected}
                  aria-pressed={expandedId === cam.id}
                  aria-label={`Select camera ${cam.name}`}
                  title={`${cam.name} — select and preview`}
                  className={`card camera-card count-item ss-strip-card${
                    selected ? " is-selected" : ""
                  }`}
                  onClick={() => handleSelect(cam)}
                >
                  <span className="card-header">
                    <span className="ss-strip-card-name">{cam.name}</span>
                    <span
                      className={`ss-camera-strip-badge ${
                        online ? "is-online" : "is-offline"
                      }`}
                      aria-hidden="true"
                    >
                      {online ? "Live" : "Offline"}
                    </span>
                    <span
                      className={`rate${online ? "" : " telemetry-hide"}`}
                      aria-label={
                        online
                          ? `Frame rate ${cameraRates[cam.sensorId] ?? "--"}`
                          : "Frame rate unavailable"
                      }
                    >
                      {online ? (cameraRates[cam.sensorId] ?? "--") : "--"}
                    </span>
                  </span>
                  <span className="card-image">
                    <span
                      className="snapshot-image"
                      data-topic={cam.cmdTopic}
                      data-topic-name={`scenescape/cmd/camera/${cam.name}`}
                    >
                      <span className="cam-offline">Camera Offline</span>
                      <img
                        id={`strip-overlay-preview-${cam.sensorId}`}
                        data-ss-card-sensor={cam.sensorId}
                        data-ss-card-name={cam.name}
                        className="display-none"
                        alt={`${cam.name} view`}
                      />
                    </span>
                  </span>
                </button>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
