// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Per-index correspondence colors — same palette as legacy
 * `CALIBRATION_POINT_COLORS` in manager/static/js/constants.js so camera
 * feed markers and 3D map spheres stay visually paired.
 */
export const CALIBRATION_POINT_COLORS = [
  "#ff0000",
  "#00ff00",
  "#0000ff",
  "#ffff00",
  "#ff00ff",
  "#00ffff",
  "#ffa500",
  "#800080",
] as const;

export function calibrationPointColor(index: number): string {
  const i =
    ((index % CALIBRATION_POINT_COLORS.length) +
      CALIBRATION_POINT_COLORS.length) %
    CALIBRATION_POINT_COLORS.length;
  return CALIBRATION_POINT_COLORS[i];
}

export const CALIBRATION_POINT_RADIUS_PX = 5;
export const CALIBRATION_POINT_HIT_PX = 10;

export function calibrationPointColorHex(index: number): number {
  return Number.parseInt(calibrationPointColor(index).slice(1), 16);
}
