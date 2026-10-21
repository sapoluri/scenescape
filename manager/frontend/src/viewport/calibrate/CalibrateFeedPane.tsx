// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { useCalibrateStore } from "./calibrateStore";
import { calibrationPointColor, CALIBRATION_POINT_RADIUS_PX, CALIBRATION_POINT_HIT_PX } from "./colors";
import { useCalibrationImage } from "./useCalibrationImage";
import "./CalibrateFeedPane.css";

const ZOOM_FACTOR = 1.1;
const MIN_SCALE = 0.1;
/** Match legacy CamCanvas fit: slightly inset from the pane. */
const FIT_INSET = 0.96;

type ViewState = {
  /** User zoom on top of fit-to-pane (1 = fitted). */
  scale: number;
  /** Top-left of the fitted image in pane coordinates. */
  panX: number;
  panY: number;
};

type DragMode =
  | { kind: "pan"; startX: number; startY: number; originPanX: number; originPanY: number }
  | { kind: "point"; index: number };

function nearestCamIndex(
  points: [number, number][],
  x: number,
  y: number,
  hitR: number,
): number {
  const lim = hitR * hitR;
  let best = -1;
  let bestD = lim;
  for (let i = 0; i < points.length; i++) {
    const dx = points[i][0] - x;
    const dy = points[i][1] - y;
    const d = dx * dx + dy * dy;
    if (d <= bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

function fitScale(paneW: number, paneH: number, imgW: number, imgH: number): number {
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
 * Right pane of Phase 2.3 calibrate split: live calibration JPEG with
 * point overlay. Wheel zooms, empty drag pans, double-click adds, drag
 * moves a point, right-click removes (legacy CamCanvas behaviour).
 */
export function CalibrateFeedPane(): React.JSX.Element {
  const camPoints = useCalibrateStore((s) => s.camPoints);
  const mapPoints = useCalibrateStore((s) => s.mapPoints);
  const sensorId = useCalibrateStore((s) => s.sensorId);
  const addCamPoint = useCalibrateStore((s) => s.addCamPoint);
  const moveCamPoint = useCalibrateStore((s) => s.moveCamPoint);
  const removeCamPointAt = useCalibrateStore((s) => s.removeCamPointAt);
  const optics = useCalibrateStore((s) => s.optics);

  const setFeedImageUrl = useCalibrateStore((s) => s.setFeedImageUrl);
  const setFeedImageSize = useCalibrateStore((s) => s.setFeedImageSize);
  const { imageUrl, naturalSize } = useCalibrationImage(sensorId, true);
  const imgW = naturalSize?.w || optics?.width || 1000;
  const imgH = naturalSize?.h || optics?.height || 1000;

  const paneRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<DragMode | null>(null);
  const userAdjustedRef = useRef(false);
  const viewRef = useRef<ViewState>({ scale: 1, panX: 0, panY: 0 });
  const baseFitRef = useRef(1);
  const [view, setView] = useState<ViewState>({ scale: 1, panX: 0, panY: 0 });
  const [paneSize, setPaneSize] = useState({ w: 0, h: 0 });
  const [cursor, setCursor] = useState<"default" | "crosshair" | "grabbing">("default");

  viewRef.current = view;

  useEffect(() => {
    setFeedImageUrl(imageUrl);
    return () => setFeedImageUrl(null);
  }, [imageUrl, setFeedImageUrl]);

  useEffect(() => {
    setFeedImageSize(naturalSize ? { width: naturalSize.w, height: naturalSize.h } : null);
    return () => setFeedImageSize(null);
  }, [naturalSize, setFeedImageSize]);

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

  // Fit + center when pane or image size changes, until the user pans/zooms.
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

  const clientToImage = (clientX: number, clientY: number): [number, number] => {
    const pane = paneRef.current;
    if (!pane) {
      return [0, 0];
    }
    const rect = pane.getBoundingClientRect();
    const mouseX = clientX - rect.left;
    const mouseY = clientY - rect.top;
    const { scale, panX, panY } = viewRef.current;
    const base = baseFitRef.current;
    const denom = base * scale;
    if (denom <= 0) {
      return [0, 0];
    }
    return [(mouseX - panX) / denom, (mouseY - panY) / denom];
  };

  // Non-passive wheel so we can prevent page scroll while zooming.
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
        ev.deltaY < 0
          ? prev.scale * ZOOM_FACTOR
          : prev.scale / ZOOM_FACTOR;
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

  const onDblClick = (ev: React.MouseEvent<HTMLDivElement>) => {
    ev.preventDefault();
    const [x, y] = clientToImage(ev.clientX, ev.clientY);
    if (x < 0 || y < 0 || x > imgW || y > imgH) {
      return;
    }
    addCamPoint([x, y]);
  };

  const onPointerDown = (ev: React.PointerEvent<HTMLDivElement>) => {
    if (ev.button !== 0) {
      return;
    }
    const el = ev.currentTarget;
    const [x, y] = clientToImage(ev.clientX, ev.clientY);
    const idx = nearestCamIndex(
      useCalibrateStore.getState().camPoints,
      x,
      y,
      CALIBRATION_POINT_HIT_PX / (baseFitRef.current * viewRef.current.scale),
    );
    if (idx >= 0) {
      useCalibrateStore.getState().beginPointEdit();
      dragRef.current = { kind: "point", index: idx };
      setCursor("grabbing");
    } else {
      const { panX, panY } = viewRef.current;
      dragRef.current = {
        kind: "pan",
        startX: ev.clientX,
        startY: ev.clientY,
        originPanX: panX,
        originPanY: panY,
      };
      setCursor("grabbing");
    }
    el.setPointerCapture(ev.pointerId);
    ev.preventDefault();
  };

  const onPointerMove = (ev: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag) {
      const [x, y] = clientToImage(ev.clientX, ev.clientY);
      const index = nearestCamIndex(
        useCalibrateStore.getState().camPoints, x, y,
        CALIBRATION_POINT_HIT_PX / (baseFitRef.current * viewRef.current.scale),
      );
      setCursor(index >= 0 ? "crosshair" : "default");
      return;
    }
    if (drag.kind === "pan") {
      const next: ViewState = {
        ...viewRef.current,
        panX: drag.originPanX + (ev.clientX - drag.startX),
        panY: drag.originPanY + (ev.clientY - drag.startY),
      };
      userAdjustedRef.current = true;
      viewRef.current = next;
      setView(next);
      return;
    }
    const [x, y] = clientToImage(ev.clientX, ev.clientY);
    moveCamPoint(drag.index, [
      Math.min(imgW, Math.max(0, x)),
      Math.min(imgH, Math.max(0, y)),
    ]);
  };

  const endDrag = (ev: React.PointerEvent<HTMLDivElement>) => {
    if (!dragRef.current) {
      return;
    }
    if (dragRef.current.kind === "point") useCalibrateStore.getState().endPointEdit();
    dragRef.current = null;
    const [x, y] = clientToImage(ev.clientX, ev.clientY);
    setCursor(nearestCamIndex(
      useCalibrateStore.getState().camPoints, x, y,
      CALIBRATION_POINT_HIT_PX / (baseFitRef.current * viewRef.current.scale),
    ) >= 0 ? "crosshair" : "default");
    try {
      ev.currentTarget.releasePointerCapture(ev.pointerId);
    } catch {
      /* already released */
    }
  };

  const onContextMenu = (ev: React.MouseEvent<HTMLDivElement>) => {
    ev.preventDefault();
    ev.stopPropagation();
    const [x, y] = clientToImage(ev.clientX, ev.clientY);
    const idx = nearestCamIndex(
      useCalibrateStore.getState().camPoints,
      x,
      y,
      CALIBRATION_POINT_HIT_PX / (baseFitRef.current * viewRef.current.scale),
    );
    if (idx >= 0) {
      removeCamPointAt(idx);
    }
  };

  const pairN = Math.min(camPoints.length, mapPoints.length);
  const hint =
    "Scroll to zoom · drag to pan · double-click to add · right-click to delete" +
    (camPoints.length !== mapPoints.length
      ? ` · cam ${camPoints.length} / map ${mapPoints.length}`
      : pairN > 0
        ? ` · ${pairN} pair${pairN === 1 ? "" : "s"}`
        : "");

  const base = baseFitRef.current;
  const fittedW = imgW * base;
  const fittedH = imgH * base;
  const markR = CALIBRATION_POINT_RADIUS_PX / (base * view.scale);

  return (
    <div className="ss-cal-feed" role="region" aria-label="Camera calibration feed">
      <div className="ss-cal-feed-bar">
        <span className="ss-cal-feed-title">Camera feed</span>
        <span className="ss-cal-feed-hint">{hint}</span>
      </div>
      <div
        className="ss-cal-feed-body"
        ref={paneRef}
        style={{ cursor }}
        onDoubleClick={onDblClick}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onPointerLeave={() => { if (!dragRef.current) setCursor("default"); }}
        onContextMenu={onContextMenu}
      >
        {imageUrl ? (
          <div
            className="ss-cal-feed-stack is-active"
            style={{
              width: fittedW,
              height: fittedH,
              transform: `translate(${view.panX}px, ${view.panY}px) scale(${view.scale})`,
              cursor,
            }}
            role="presentation"
          >
            <img src={imageUrl} alt="Calibration frame" draggable={false} />
            <svg
              className="ss-cal-feed-overlay"
              viewBox={`0 0 ${imgW} ${imgH}`}
              preserveAspectRatio="none"
            >
              {camPoints.map((pt, i) => (
                <g key={`c-${i}`}>
                  <circle
                    cx={pt[0]}
                    cy={pt[1]}
                    r={markR}
                    fill={calibrationPointColor(i)}
                  />
                </g>
              ))}
            </svg>
          </div>
        ) : (
          <div className="ss-cal-feed-empty">Waiting for calibration frame…</div>
        )}
        {imageUrl && camPoints.map((point, index) => (
          <div
            key={`label-${index}`}
            className="ss-vlabel ss-vlabel--cal"
            style={{
              left: view.panX + point[0] * base * view.scale,
              top: view.panY + (point[1] - markR) * base * view.scale - 4,
              transform: "translate(-50%, -100%)",
              borderColor: calibrationPointColor(index),
            }}
          >
            {index + 1}
          </div>
        ))}
      </div>
    </div>
  );
}
