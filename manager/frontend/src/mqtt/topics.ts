// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/** MQTT topic fragments — keep in sync with manager/static/js/constants.js. */
export const APP_NAME = "scenescape";
export const CMD_CAMERA = "/cmd/camera/";
export const DATA_REGULATED = "/regulated/scene/";
export const DATA_SCENE = "/data/scene/";
export const DATA_CAMERA = "/data/camera/";
export const DATA_SENSOR = "/data/sensor/";
export const IMAGE_CAMERA = "/image/camera/";
export const IMAGE_CALIBRATE = "/image/calibration/camera/";
export const SYS_CHILDSCENE_STATUS = "/sys/child/status";

export function sceneRegulatedTopic(sceneId: string): string {
  return `${APP_NAME}${DATA_REGULATED}${sceneId}`;
}

/** Unregulated scene output: `scenescape/data/scene/{sceneId}/+`. */
export function sceneDataOutputTopic(sceneId: string): string {
  return `${APP_NAME}${DATA_SCENE}${sceneId}/+`;
}

export function sceneEventTopic(sceneId: string): string {
  return `${APP_NAME}/event/+/${sceneId}/+/+`;
}

export function sceneRegionEventTopic(sceneId: string): string {
  return `${APP_NAME}/event/region/${sceneId}/+/+`;
}

export function sceneTripwireEventTopic(sceneId: string): string {
  return `${APP_NAME}/event/tripwire/${sceneId}/+/+`;
}

export function cameraImageTopic(): string {
  return `${APP_NAME}${IMAGE_CAMERA}+`;
}

export function cameraImageTopicFor(sensorId: string): string {
  return `${APP_NAME}${IMAGE_CAMERA}${sensorId}`;
}

export function cameraDataTopic(sensorId: string): string {
  return `${APP_NAME}${DATA_CAMERA}${sensorId}`;
}

export function cameraCmdTopic(sensorId: string): string {
  return `${APP_NAME}${CMD_CAMERA}${sensorId}`;
}

export function sensorDataTopic(sensorId: string): string {
  return `${APP_NAME}${DATA_SENSOR}${sensorId}`;
}

export function childStatusTopic(): string {
  return `${APP_NAME}${SYS_CHILDSCENE_STATUS}/+`;
}
