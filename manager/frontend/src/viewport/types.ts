// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Single-pane 3D viewport entity model.
 *
 * World convention is Z-up, matching the Scenescape domain (MobileObject x/y,
 * Asset3D sizes, and the legacy scenescape3d.js viewer): positions are
 * [x, y, z] meters with z = height above the floor plane. This mirrors the
 * SceneEulerPose convention in src/placement (XYZ degrees, Z-up).
 */

export type ViewportEntityType =
  | "region"
  | "tripwire"
  | "camera"
  | "sensor"
  | "mark"
  | "child";

export interface ViewportEntityBase {
  id: string;
  type: ViewportEntityType;
  name: string;
  visible: boolean;
}

export interface RegionEntity extends ViewportEntityBase {
  type: "region";
  /** Floor-plane corners in meters: [x, y]. */
  min: [number, number];
  max: [number, number];
  height: number;
  color: string;
  volumetric: boolean;
  bufferSize: number;
}

export interface TripwireEntity extends ViewportEntityBase {
  type: "tripwire";
  /** Endpoints on the floor plane in meters: [x, y]. */
  a: [number, number];
  b: [number, number];
  color: string;
}

export interface CameraEntity extends ViewportEntityBase {
  type: "camera";
  /** Camera position [x, y, z] meters, Z-up. */
  position: [number, number, number];
  /** Look-at target [x, y, z] meters. */
  target: [number, number, number];
  fov: number;
  color: string;
}

export interface SensorEntity extends ViewportEntityBase {
  type: "sensor";
  position: [number, number, number];
  color: string;
}

/**
 * Live tracked-object mark, drawn as tracker transform × Asset3D default
 * pose × GLB (see docs/design/single-pane-ui.md). For Phase 1.1 marks render
 * as colored primitives; GLB bodies arrive with the Object Library drawer
 * (Phase 1.3).
 */
export interface MarkEntity extends ViewportEntityBase {
  type: "mark";
  /** Asset3D class name, e.g. "person", "forklift". */
  className: string;
  color: string;
  /** Tracker position [x, y, z] meters, Z-up. */
  position: [number, number, number];
  /** Heading radians about the Z axis, derived from velocity. NaN if unknown. */
  heading: number;
  speed: number;
}

export interface ChildSceneEntity extends ViewportEntityBase {
  type: "child";
  position: [number, number, number];
  color: string;
}

export type ViewportEntity =
  | RegionEntity
  | TripwireEntity
  | CameraEntity
  | SensorEntity
  | MarkEntity
  | ChildSceneEntity;

export type ViewPreset = "persp" | "top" | "front" | "side";

export function isMark(e: ViewportEntity): e is MarkEntity {
  return e.type === "mark";
}
