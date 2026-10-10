// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef } from "react";
import { useCalibrateStore } from "./calibrateStore";
import { calibrationPointColor } from "./colors";
import { useCalibrationImage } from "./useCalibrationImage";
import "./CalibrateFeedPane.css";

function clientToImage(
  el: HTMLElement,
  clientX: number,
  clientY: number,
  imgW: number,
  imgH: number,
): [number, number] {
  const rect = el.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * imgW;
  const y = ((clientY - rect.top) / rect.height) * imgH;
  return [x, y];
}

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

/**
 * Right pane of Phase 2.3 calibrate split: live calibration JPEG with
 * point overlay. Double-click to add, drag to move, right-click to remove
 * (same as the legacy 2D calibrator).
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
  const { imageUrl, naturalSize } = useCalibrationImage(sensorId, true);
  const imgW = naturalSize?.w || optics?.width || 1000;
  const imgH = naturalSize?.h || optics?.height || 1000;
  const markR = Math.max(imgW, imgH) * 0.008;
  const hitR = Math.max(markR * 4, Math.max(imgW, imgH) * 0.02);
  const dragIndex = useRef<number | null>(null);

  useEffect(() => {
    setFeedImageUrl(imageUrl);
    return () => setFeedImageUrl(null);
  }, [imageUrl, setFeedImageUrl]);

  const onDblClick = (ev: React.MouseEvent<HTMLDivElement>) => {
    ev.preventDefault();
    const el = ev.currentTarget;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return;
    }
    const [x, y] = clientToImage(el, ev.clientX, ev.clientY, imgW, imgH);
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
    const [x, y] = clientToImage(el, ev.clientX, ev.clientY, imgW, imgH);
    const idx = nearestCamIndex(
      useCalibrateStore.getState().camPoints,
      x,
      y,
      hitR,
    );
    if (idx < 0) {
      return;
    }
    dragIndex.current = idx;
    el.setPointerCapture(ev.pointerId);
    ev.preventDefault();
  };

  const onPointerMove = (ev: React.PointerEvent<HTMLDivElement>) => {
    if (dragIndex.current === null) {
      return;
    }
    const el = ev.currentTarget;
    const [x, y] = clientToImage(el, ev.clientX, ev.clientY, imgW, imgH);
    moveCamPoint(dragIndex.current, [
      Math.min(imgW, Math.max(0, x)),
      Math.min(imgH, Math.max(0, y)),
    ]);
  };

  const endDrag = (ev: React.PointerEvent<HTMLDivElement>) => {
    if (dragIndex.current === null) {
      return;
    }
    dragIndex.current = null;
    try {
      ev.currentTarget.releasePointerCapture(ev.pointerId);
    } catch {
      /* already released */
    }
  };

  const onContextMenu = (ev: React.MouseEvent<HTMLDivElement>) => {
    ev.preventDefault();
    ev.stopPropagation();
    const el = ev.currentTarget;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return;
    }
    const [x, y] = clientToImage(el, ev.clientX, ev.clientY, imgW, imgH);
    const idx = nearestCamIndex(
      useCalibrateStore.getState().camPoints,
      x,
      y,
      hitR,
    );
    if (idx >= 0) {
      removeCamPointAt(idx);
    }
  };

  const pairN = Math.min(camPoints.length, mapPoints.length);
  const hint =
    "Double-click to add · drag to move · right-click to delete" +
    (camPoints.length !== mapPoints.length
      ? ` · cam ${camPoints.length} / map ${mapPoints.length}`
      : pairN > 0
        ? ` · ${pairN} pair${pairN === 1 ? "" : "s"}`
        : "");

  return (
    <div className="ss-cal-feed" role="region" aria-label="Camera calibration feed">
      <div className="ss-cal-feed-bar">
        <span className="ss-cal-feed-title">Camera feed</span>
        <span className="ss-cal-feed-hint">{hint}</span>
      </div>
      <div className="ss-cal-feed-body">
        <div
          className="ss-cal-feed-stack is-active"
          onDoubleClick={onDblClick}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          onContextMenu={onContextMenu}
          role="presentation"
        >
          {imageUrl ? (
            <img src={imageUrl} alt="Calibration frame" draggable={false} />
          ) : (
            <div className="ss-cal-feed-empty">Waiting for calibration frame…</div>
          )}
          {imageUrl ? (
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
                  <text
                    x={pt[0] + markR * 1.2}
                    y={pt[1] - markR}
                    fill="#e9eaee"
                    fontSize={markR * 3.2}
                    fontWeight="700"
                  >
                    {i + 1}
                  </text>
                </g>
              ))}
            </svg>
          ) : null}
        </div>
      </div>
    </div>
  );
}
