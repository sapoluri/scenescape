// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useWorkspaceEditing } from "../../lib/editMode";

import { useEffect } from "react";
import { Button } from "../../components/Button";
import type { SceneCameraBootstrap } from "../../scene/types";
import { enterCameraCalibrate } from "../calibrate";
import { ProjectionOpacitySlider } from "../ProjectionOpacitySlider";
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
  const canEdit = useWorkspaceEditing();
  const cameraName = useVisualizeStore((s) => s.cameraName);
  const exit = useVisualizeStore((s) => s.exit);
  const entities = useViewportStore((s) => s.entities);
  const projectCameraIds = useViewportStore((s) => s.projectCameraIds);
  const projecting = cameraId ? projectCameraIds.includes(cameraId) : false;
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
    <div
      className="ss-cal-wizard ss-cal-wizard--calibration ss-viz-bar"
      role="toolbar"
      aria-label="Live view"
    >
      <div className="ss-cal-wizard-main">
        <label className="ss-cal-wizard-cam">
          <span className="ss-viewport-bar-label">Live Camera</span>
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
        <div className="ss-viz-opacity-slot">
          {projecting && cameraId ? (
            <ProjectionOpacitySlider cameraId={cameraId} />
          ) : null}
        </div>
      </div>
      <div className="ss-cal-wizard-actions">
        <div className="ss-cal-wizard-buttons">
          <Button
            variant={projecting ? "primary" : "secondary"}
            className="ss-btn--sm"
            aria-pressed={projecting}
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
            <span>Project</span>
          </Button>
          {isSuperuser && canEdit ? (
            <Button
              variant="secondary"
              className="ss-btn--sm"
              onClick={onCalibrate}
              title="Open calibrate for this camera (raw calibration frames)"
            >
              <i className="bi bi-bullseye" aria-hidden="true" />
              <span>Calibrate</span>
            </Button>
          ) : null}
          <button
            type="button"
            className="ss-sp-dock-props-close"
            onClick={() => exit()}
            title="Close live view (Esc)"
            aria-label="Close live view"
          >
            <i className="bi bi-x" aria-hidden="true" />
          </button>
        </div>
      </div>
    </div>
  );
}
