// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCalibrateStore } from "./calibrateStore";
import { useCalibrationImage } from "./useCalibrationImage";
import "./CalibrateFeedPane.css";

/**
 * Right pane of Phase 2.3 calibrate split: live calibration JPEG with
 * point-picking overlay (click → then click matching floor point in 3D).
 */
export function CalibrateFeedPane(): React.JSX.Element {
  const sensorId = useCalibrateStore((s) => s.sensorId);
  const pairs = useCalibrateStore((s) => s.pairs);
  const pending = useCalibrateStore((s) => s.pending);
  const draftCam = useCalibrateStore((s) => s.draftCam);
  const setDraftCam = useCalibrateStore((s) => s.setDraftCam);
  const setPending = useCalibrateStore((s) => s.setPending);
  const optics = useCalibrateStore((s) => s.optics);

  const { imageUrl, naturalSize } = useCalibrationImage(sensorId, true);
  const imgW = naturalSize?.w || optics?.width || 1000;
  const imgH = naturalSize?.h || optics?.height || 1000;
  const markR = Math.max(imgW, imgH) * 0.008;

  const onClick = (ev: React.MouseEvent<HTMLDivElement>) => {
    if (pending !== "cam") {
      return;
    }
    const el = ev.currentTarget;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return;
    }
    const x = ((ev.clientX - rect.left) / rect.width) * imgW;
    const y = ((ev.clientY - rect.top) / rect.height) * imgH;
    setDraftCam([x, y]);
    setPending("scene");
  };

  return (
    <div className="ss-cal-feed" role="region" aria-label="Camera calibration feed">
      <div className="ss-cal-feed-bar">
        <span className="ss-cal-feed-title">Camera feed</span>
        <span className="ss-cal-feed-hint">
          {pending === "cam"
            ? "Click a known point in the feed"
            : "Click the matching floor point in the 3D view"}
        </span>
      </div>
      <div className="ss-cal-feed-body">
        <div
          className={`ss-cal-feed-stack${pending === "cam" ? " is-active" : ""}`}
          onClick={onClick}
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
                  <circle cx={p.cam[0]} cy={p.cam[1]} r={markR} fill="#0a84ff" />
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
                <circle
                  cx={draftCam[0]}
                  cy={draftCam[1]}
                  r={markR}
                  fill="#ff453a"
                />
              ) : null}
            </svg>
          ) : null}
        </div>
      </div>
    </div>
  );
}
