// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export const DEFAULT_DOCK_WIDTH = 360;
const WIDTH_KEY = "ss-sp-dock-width-px";

export function clampDockWidth(width: number, availableWidth: number): number {
  const available = Number.isFinite(availableWidth)
    ? Math.max(0, availableWidth)
    : 720;
  const maximum = Math.max(0, available - Math.min(320, available / 2) - 16);
  const minimum = Math.min(240, maximum);
  return Math.min(
    maximum,
    Math.max(minimum, Number.isFinite(width) ? width : DEFAULT_DOCK_WIDTH),
  );
}

export function readDockWidth(): number {
  try {
    const width = Number(sessionStorage.getItem(WIDTH_KEY));
    if (Number.isFinite(width) && width >= 96) return width;
  } catch {
    return DEFAULT_DOCK_WIDTH;
  }
  return DEFAULT_DOCK_WIDTH;
}

export function writeDockWidth(width: number): void {
  try {
    sessionStorage.setItem(WIDTH_KEY, String(Math.round(width)));
  } catch {
    return;
  }
}
