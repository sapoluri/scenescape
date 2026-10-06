// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Typed React → hybrid bridges.
 *
 * Single-pane cutover: the 2D map is gone, so fit / number helpers are
 * no-ops. Geometry persist uses React-installed `ssPersistGeometry`.
 */

import type { PersistGeometryResult } from "./roiPersist";

type PersistOptions = { preferHidden?: boolean } | string[];

type RoiColorSectors = {
  thresholds: { color: string; color_min: number }[];
  range_max: number;
};

export function fitSceneMapDisplay(): void {
  // No 2D map in the single pane; kept for WorkspaceSplitter callers.
}

export function numberRois(): void {
  // No 2D map in the single pane.
}

export function numberTripwires(): void {
  // No 2D map in the single pane.
}

export function syncRoiColorSectors(
  uuid: string,
  sectors: RoiColorSectors,
): void {
  window.ssSyncRoiColorSectors?.(uuid, sectors);
}

export function reapplyRoiColors(): void {
  window.ssReapplyRoiColors?.();
}

export function persistGeometry(
  options?: PersistOptions,
): void | Promise<PersistGeometryResult | void> {
  return window.ssPersistGeometry?.(options);
}

declare global {
  interface Window {
    ssUseReactMap?: boolean;
    ssInitSceneMap?: (attempt?: number) => boolean;
    ssSyncRoiColorSectors?: (uuid: string, sectors: RoiColorSectors) => void;
    ssReapplyRoiColors?: () => void;
  }
}
