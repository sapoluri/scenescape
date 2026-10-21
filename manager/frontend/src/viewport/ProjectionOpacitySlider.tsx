// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getViewportState, projectOpacityOf, useViewportStore } from "./store";

type Props = {
  /** Extra class on the label (e.g. properties-panel / outliner layout). */
  className?: string;
  /**
   * Per-camera stored opacity (Properties / calibrate / Live). Applies whenever
   * Project-all is off.
   */
  cameraId?: string;
  /**
   * Project-all overlay override. Does not change per-camera stored values.
   */
  projectAll?: boolean;
};

/**
 * Map-projection overlay opacity. Use `cameraId` for one camera’s stored
 * value, or `projectAll` for the temporary Project-all override.
 */
export function ProjectionOpacitySlider({
  className,
  cameraId,
  projectAll = false,
}: Props): React.JSX.Element | null {
  const projectOpacityById = useViewportStore((s) => s.projectOpacityById);
  const projectAllOpacity = useViewportStore((s) => s.projectAllOpacity);
  if (!projectAll && !cameraId) {
    return null;
  }
  const opacity = projectAll
    ? projectAllOpacity
    : projectOpacityOf(projectOpacityById, cameraId!);
  const pct = Math.round(opacity * 100);
  const cls = ["ss-cal-wizard-opacity", className].filter(Boolean).join(" ");
  return (
    <label
      className={cls}
      title={
        projectAll
          ? "Overlay opacity while Project all is on (override)"
          : "Map projection opacity for this camera"
      }
      onClick={(ev) => ev.stopPropagation()}
      onPointerDown={(ev) => ev.stopPropagation()}
    >
      <span className="ss-viewport-bar-label">Overlay</span>
      <span className="ss-cal-wizard-opacity-track">
        <input
          type="range"
          min={0}
          max={100}
          step={1}
          value={pct}
          onChange={(ev) => {
            const next = Number(ev.target.value) / 100;
            if (projectAll) {
              getViewportState().setProjectAllOpacity(next);
            } else if (cameraId) {
              getViewportState().setProjectOpacity(cameraId, next);
            }
          }}
          aria-label={
            projectAll
              ? "Projection opacity override for all cameras"
              : "Projection opacity"
          }
        />
      </span>
      <span className="ss-cal-wizard-opacity-val">{pct}%</span>
    </label>
  );
}
