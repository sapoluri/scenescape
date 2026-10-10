// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from "react";
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

/**
 * Right pane of Phase 2.3 calibrate split: live calibration JPEG with
 * point-picking overlay. Click either this feed or the 3D map first;
 * the other side completes the pair. Right-click a marker to remove.
 */
export function CalibrateFeedPane(): React.JSX.Element {
  const sensorId = useCalibrateStore((s) => s.sensorId);
  const pairs = useCalibrateStore((s) => s.pairs);
  const pending = useCalibrateStore((s) => s.pending);
  const draftCam = useCalibrateStore((s) => s.draftCam);
  const draftMap = useCalibrateStore((s) => s.draftMap);
  const pickCam = useCalibrateStore((s) => s.pickCam);
  const clearDraft = useCalibrateStore((s) => s.clearDraft);
  const removePairAt = useCalibrateStore((s) => s.removePairAt);
  const optics = useCalibrateStore((s) => s.optics);

  const setFeedImageUrl = useCalibrateStore((s) => s.setFeedImageUrl);
  const { imageUrl, naturalSize } = useCalibrationImage(sensorId, true);
  const imgW = naturalSize?.w || optics?.width || 1000;
  const imgH = naturalSize?.h || optics?.height || 1000;
  const markR = Math.max(imgW, imgH) * 0.008;
  // Generous hit target so right-click still works as points accumulate.
  const hitR = Math.max(markR * 4, Math.max(imgW, imgH) * 0.02);
  const nextColor = calibrationPointColor(pairs.length);

  useEffect(() => {
    setFeedImageUrl(imageUrl);
    return () => setFeedImageUrl(null);
  }, [imageUrl, setFeedImageUrl]);

  const feedAcceptsClick = pending !== "scene";

  const onClick = (ev: React.MouseEvent<HTMLDivElement>) => {
    if (!feedAcceptsClick) {
      return;
    }
    const el = ev.currentTarget;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return;
    }
    pickCam(clientToImage(el, ev.clientX, ev.clientY, imgW, imgH));
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
    // Read latest store — avoids stale pair lists after rapid edits.
    const s = useCalibrateStore.getState();
    const curPairs = s.pairs;
    const curDraft = s.draftCam;
    const lim = hitR * hitR;

    if (curDraft) {
      const dx = curDraft[0] - x;
      const dy = curDraft[1] - y;
      if (dx * dx + dy * dy <= lim) {
        clearDraft();
        return;
      }
    }

    let best = -1;
    let bestD = lim;
    for (let i = 0; i < curPairs.length; i++) {
      const dx = curPairs[i].cam[0] - x;
      const dy = curPairs[i].cam[1] - y;
      const d = dx * dx + dy * dy;
      if (d <= bestD) {
        bestD = d;
        best = i;
      }
    }
    if (best >= 0) {
      removePairAt(best);
    }
  };

  let hint = "Click either this feed or the 3D map · right-click to remove";
  if (pending === "scene") {
    hint = "Click the matching floor point in the 3D view";
  } else if (pending === "cam" && draftMap) {
    hint = "Click the matching point in this feed";
  }

  return (
    <div className="ss-cal-feed" role="region" aria-label="Camera calibration feed">
      <div className="ss-cal-feed-bar">
        <span className="ss-cal-feed-title">Camera feed</span>
        <span className="ss-cal-feed-hint">{hint}</span>
      </div>
      <div className="ss-cal-feed-body">
        <div
          className={`ss-cal-feed-stack${feedAcceptsClick ? " is-active" : ""}`}
          onClick={onClick}
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
              {pairs.map((p, i) => (
                <g key={`c-${i}`}>
                  <circle
                    cx={p.cam[0]}
                    cy={p.cam[1]}
                    r={markR}
                    fill={calibrationPointColor(i)}
                  />
                  <text
                    x={p.cam[0] + markR * 1.2}
                    y={p.cam[1] - markR}
                    fill="#e9eaee"
                    fontSize={markR * 3.2}
                    fontWeight="700"
                  >
                    {i + 1}
                  </text>
                </g>
              ))}
              {draftCam ? (
                <g>
                  <circle
                    cx={draftCam[0]}
                    cy={draftCam[1]}
                    r={markR}
                    fill={nextColor}
                    stroke="#ffffff"
                    strokeWidth={markR * 0.35}
                  />
                  <text
                    x={draftCam[0] + markR * 1.2}
                    y={draftCam[1] - markR}
                    fill="#e9eaee"
                    fontSize={markR * 3.2}
                    fontWeight="700"
                  >
                    {pairs.length + 1}
                  </text>
                </g>
              ) : null}
            </svg>
          ) : null}
        </div>
      </div>
    </div>
  );
}
