// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Typed React → hybrid `sscape.js` bridges for Snap map / ROI helpers.
 *
 * MQTT connect, camera strip, and local sensor draw are React-owned
 * (`src/mqtt/`, `SensorLayer`). Do not reintroduce those via this module.
 */

type PersistOptions = { preferHidden?: boolean } | string[];

type RoiColorSectors = {
  thresholds: { color: string; color_min: number }[];
  range_max: number;
};

export function fitSceneMapDisplay(): void {
  if (typeof window.ssMap?.fit === "function") {
    window.ssMap.fit();
    return;
  }
  window.fitSceneMapDisplay?.();
}

export function numberRois(): void {
  if (typeof window.ssMap?.numberRois === "function") {
    window.ssMap.numberRois();
    return;
  }
  window.numberRois?.();
}

export function numberTripwires(): void {
  if (typeof window.ssMap?.numberTripwires === "function") {
    window.ssMap.numberTripwires();
    return;
  }
  window.numberTripwires?.();
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
): void | Promise<void> {
  return window.ssPersistGeometry?.(options);
}

declare global {
  interface Window {
    ssSyncRoiColorSectors?: (uuid: string, sectors: RoiColorSectors) => void;
    ssReapplyRoiColors?: () => void;
  }
}
