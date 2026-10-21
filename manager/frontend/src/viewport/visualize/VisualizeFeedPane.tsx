// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { formatCameraFps } from "../../scene/useLiveChrome";
import { useViewportStore } from "../store";
import { useLiveCameraImage } from "./useLiveCameraImage";
import { useVisualizeStore } from "./visualizeStore";
import "./VisualizeFeedPane.css";

const ZOOM_FACTOR = 1.1;
const MIN_SCALE = 0.1;
const FIT_INSET = 0.96;

type ViewState = {
  scale: number;
  panX: number;
  panY: number;
};

function fitScale(
  paneW: number,
  paneH: number,
  imgW: number,
  imgH: number,
): number {
  if (paneW <= 0 || paneH <= 0 || imgW <= 0 || imgH <= 0) {
    return 1;
  }
  return Math.min(paneW / imgW, paneH / imgH) * FIT_INSET;
}

function centerPan(
  paneW: number,
  paneH: number,
  imgW: number,
  imgH: number,
  base: number,
  scale: number,
): Pick<ViewState, "panX" | "panY"> {
  const displayW = imgW * base * scale;
  const displayH = imgH * base * scale;
  return {
    panX: (paneW - displayW) / 2,
    panY: (paneH - displayH) / 2,
  };
}

/**
 * Right pane of the visualization split: annotated live JPEG (getimage).
 * Wheel zooms; drag pans. No point editing.
 */
export function VisualizeFeedPane({ cameraRates = {} }: { cameraRates?: Record<string, string> }): React.JSX.Element {
  const sensorId = useVisualizeStore((s) => s.sensorId);
  const showTelemetry = useViewportStore((state) => state.showTelemetry);
  const cameraName = useVisualizeStore((s) => s.cameraName);
  const setFeedImageUrl = useVisualizeStore((s) => s.setFeedImageUrl);
  const { imageUrl, naturalSize } = useLiveCameraImage(sensorId, true);
  const imgW = naturalSize?.w || 1280;
  const imgH = naturalSize?.h || 720;

  const paneRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{
    startX: number;
    startY: number;
    originPanX: number;
    originPanY: number;
  } | null>(null);
  const userAdjustedRef = useRef(false);
  const viewRef = useRef<ViewState>({ scale: 1, panX: 0, panY: 0 });
  const baseFitRef = useRef(1);
  const [view, setView] = useState<ViewState>({ scale: 1, panX: 0, panY: 0 });
  const [paneSize, setPaneSize] = useState({ w: 0, h: 0 });
  const [panning, setPanning] = useState(false);

  viewRef.current = view;

  useEffect(() => {
    setFeedImageUrl(imageUrl);
    return () => setFeedImageUrl(null);
  }, [imageUrl, setFeedImageUrl]);

  useLayoutEffect(() => {
    const pane = paneRef.current;
    if (!pane) {
      return;
    }
    const apply = () => {
      const w = pane.clientWidth;
      const h = pane.clientHeight;
      setPaneSize((prev) => (prev.w === w && prev.h === h ? prev : { w, h }));
    };
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(pane);
    return () => ro.disconnect();
  }, []);

  useLayoutEffect(() => {
    if (paneSize.w <= 0 || paneSize.h <= 0) {
      return;
    }
    const base = fitScale(paneSize.w, paneSize.h, imgW, imgH);
    baseFitRef.current = base;
    if (userAdjustedRef.current) {
      return;
    }
    const next = {
      scale: 1,
      ...centerPan(paneSize.w, paneSize.h, imgW, imgH, base, 1),
    };
    viewRef.current = next;
    setView(next);
  }, [paneSize.w, paneSize.h, imgW, imgH, imageUrl]);

  useEffect(() => {
    const pane = paneRef.current;
    if (!pane) {
      return;
    }
    const onWheel = (ev: WheelEvent) => {
      ev.preventDefault();
      const rect = pane.getBoundingClientRect();
      const mouseX = ev.clientX - rect.left;
      const mouseY = ev.clientY - rect.top;
      const prev = viewRef.current;
      const nextScale =
        ev.deltaY < 0 ? prev.scale * ZOOM_FACTOR : prev.scale / ZOOM_FACTOR;
      const scale = Math.max(MIN_SCALE, nextScale);
      const ratio = scale / prev.scale;
      const next: ViewState = {
        scale,
        panX: mouseX - (mouseX - prev.panX) * ratio,
        panY: mouseY - (mouseY - prev.panY) * ratio,
      };
      userAdjustedRef.current = true;
      viewRef.current = next;
      setView(next);
    };
    pane.addEventListener("wheel", onWheel, { passive: false });
    return () => pane.removeEventListener("wheel", onWheel);
  }, []);

  const onPointerDown = (ev: React.PointerEvent<HTMLDivElement>) => {
    if (ev.button !== 0) {
      return;
    }
    const { panX, panY } = viewRef.current;
    dragRef.current = {
      startX: ev.clientX,
      startY: ev.clientY,
      originPanX: panX,
      originPanY: panY,
    };
    setPanning(true);
    ev.currentTarget.setPointerCapture(ev.pointerId);
  };

  const onPointerMove = (ev: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag) {
      return;
    }
    const next: ViewState = {
      ...viewRef.current,
      panX: drag.originPanX + (ev.clientX - drag.startX),
      panY: drag.originPanY + (ev.clientY - drag.startY),
    };
    userAdjustedRef.current = true;
    viewRef.current = next;
    setView(next);
  };

  const onPointerUp = (ev: React.PointerEvent<HTMLDivElement>) => {
    if (dragRef.current) {
      dragRef.current = null;
      setPanning(false);
      try {
        ev.currentTarget.releasePointerCapture(ev.pointerId);
      } catch {
        /* already released */
      }
    }
  };

  const base = baseFitRef.current;
  const displayScale = base * view.scale;

  return (
    <div className="ss-viz-feed" aria-label="Live camera feed">
      <div className="ss-viz-feed-bar">
        <span className="ss-viz-feed-title">
          {cameraName || "Camera"} · annotated
        </span>
        <span className="ss-viz-feed-hint">Scroll to zoom · drag to pan</span>
        {showTelemetry ? (
          <span className="ss-viz-feed-fps" role="group" aria-label="Active camera telemetry">
            {formatCameraFps(sensorId ? cameraRates[sensorId] : undefined)}
          </span>
        ) : null}
      </div>
      <div
        ref={paneRef}
        className="ss-viz-feed-body"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        {imageUrl ? (
          <div
            className={`ss-viz-feed-stack${panning ? " is-panning" : ""}`}
            style={{
              transform: `translate(${view.panX}px, ${view.panY}px) scale(${displayScale})`,
            }}
          >
            <img
              className="ss-viz-feed-img"
              src={imageUrl}
              alt=""
              width={imgW}
              height={imgH}
              draggable={false}
            />
          </div>
        ) : (
          <p className="ss-viz-feed-empty">
            Waiting for annotated frames… Enable the camera pipeline and check
            MQTT connectivity.
          </p>
        )}
      </div>
    </div>
  );
}
