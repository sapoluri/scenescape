// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from "react";
import type { SceneCameraBootstrap } from "../../scene/types";
import { enterCameraCalibrate } from "../calibrate";
import { getViewportState, useViewportStore } from "../store";
import { enterCameraVisualize } from "./index";
import { useVisualizeStore } from "./visualizeStore";
import "../calibrate/CalibrateWizard.css";

type Props = {
  cameras: SceneCameraBootstrap[];
  isSuperuser?: boolean;
};

/**
 * Mode strip for the visualization split: camera switcher, Project,
 * Calibrate handoff, Exit.
 */
export function VisualizeBar({
  cameras,
  isSuperuser = false,
}: Props): React.JSX.Element {
  const cameraId = useVisualizeStore((s) => s.cameraId);
  const cameraName = useVisualizeStore((s) => s.cameraName);
  const exit = useVisualizeStore((s) => s.exit);
  const entities = useViewportStore((s) => s.entities);
  const projectCameraIds = useViewportStore((s) => s.projectCameraIds);
  const projecting = cameraId
    ? projectCameraIds.includes(cameraId)
    : false;
  const posed =
    cameraId != null &&
    entities[cameraId]?.type === "camera" &&
    entities[cameraId].posed;

  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      const t = ev.target as HTMLElement | null;
      if (
        t &&
        (t.tagName === "INPUT" ||
          t.tagName === "TEXTAREA" ||
          t.isContentEditable)
      ) {
        return;
      }
      if (ev.key === "Escape") {
        exit();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [exit]);

  const onCalibrate = () => {
    const cam = cameras.find((c) => c.id === cameraId);
    if (!cam) {
      return;
    }
    enterCameraCalibrate({
      cameraId: cam.id,
      sensorId: cam.sensorId,
      cameraName: cam.name,
    });
  };

  return (
    <div className="ss-cal-wizard ss-viz-bar" role="toolbar" aria-label="Live view">
      <span className="ss-cal-wizard-badge">Live</span>
      <label className="ss-cal-wizard-cam">
        <span className="ss-viewport-bar-label">Camera</span>
        <select
          value={cameraId || ""}
          onChange={(ev) => {
            const id = ev.target.value;
            const cam = cameras.find((c) => c.id === id);
            if (!cam) {
              return;
            }
            enterCameraVisualize({
              cameraId: cam.id,
              sensorId: cam.sensorId,
              cameraName: cam.name,
            });
          }}
          aria-label="Select camera to view"
        >
          {cameras.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </select>
      </label>
      <span className="ss-cal-wizard-step" title={cameraName}>
        Annotated pipeline frames
      </span>
      <span className="ss-viewport-bar-sep" aria-hidden="true" />
      <button
        type="button"
        className={`ss-tool-btn${projecting ? " on" : ""}`}
        disabled={!posed}
        title={
          posed
            ? projecting
              ? "Stop projecting onto the scene map"
              : "Project live frame onto the scene map"
            : "Calibrate this camera first — projection needs a pose"
        }
        onClick={() => {
          if (!cameraId) {
            return;
          }
          getViewportState().setProjectCameraEnabled(cameraId, !projecting);
        }}
      >
        <i className="bi bi-cast" aria-hidden="true" />
        <span className="ss-tool-label">Project</span>
      </button>
      {isSuperuser ? (
        <button
          type="button"
          className="ss-tool-btn"
          onClick={onCalibrate}
          title="Open calibrate for this camera (raw calibration frames)"
        >
          <i className="bi bi-bullseye" aria-hidden="true" />
          <span className="ss-tool-label">Calibrate</span>
        </button>
      ) : null}
      <button
        type="button"
        className="ss-tool-btn ss-cal-wizard-exit"
        onClick={() => exit()}
        title="Exit live view (Esc)"
      >
        Exit
      </button>
    </div>
  );
}
