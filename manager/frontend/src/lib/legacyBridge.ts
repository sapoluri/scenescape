// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Typed React → hybrid `sscape.js` bridges.
 *
 * Prefer this module over ad-hoc `window.ss*` / `window.fitSceneMapDisplay`
 * in React. Implementations still live on `window` until MQTT/map ownership
 * moves fully into React (plan item D remainder).
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

export function refreshCameraSnapshots(): void {
  window.ssRefreshCameraSnapshots?.();
}

export function drawSingletonSensors(): void {
  window.ssDrawSingletonSensors?.();
}

export function removeSingletonSensor(sensorId: string): void {
  window.ssRemoveSingletonSensor?.(sensorId);
}

export function ensureMqttScene(): void {
  window.ssEnsureMqttScene?.();
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
    ssRefreshCameraSnapshots?: () => void;
    ssDrawSingletonSensors?: () => void;
    ssRemoveSingletonSensor?: (sensorId: string) => void;
    ssEnsureMqttScene?: () => void;
    ssSyncRoiColorSectors?: (uuid: string, sectors: RoiColorSectors) => void;
    ssReapplyRoiColors?: () => void;
  }
}
