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
  | "child"
  | "map";

export interface ViewportEntityBase {
  id: string;
  type: ViewportEntityType;
  name: string;
  visible: boolean;
}

export interface RegionEntity extends ViewportEntityBase {
  type: "region";
  /** Floor-plane polygon corners in meters: [x, y]. */
  points: [number, number][];
  /** Derived bbox of points in meters: [x, y]. */
  min: [number, number];
  max: [number, number];
  height: number;
  color: string;
  volumetric: boolean;
  bufferSize: number;
}

export interface TripwireEntity extends ViewportEntityBase {
  type: "tripwire";
  /** Polyline points on the floor plane in meters: [x, y]. */
  points: [number, number][];
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
  /**
   * Rig orientation as XYZ euler degrees, Z-up (REST pose × Rx(π) when
   * calibrated; look-at fallback when not).
   */
  rotation: [number, number, number];
  /**
   * REST / solve XYZ euler degrees for map projection. Meaningful only when
   * `posed` is true (matches calibrate previewPose.rotation).
   */
  restRotation: [number, number, number];
  /** True when REST calibration translation is known (projection eligible). */
  posed: boolean;
  fov: number;
  /** Calibrated image aspect, derived from centered cx/cy when available. */
  aspect?: number;
  color: string;
}

/** Sensor coverage area (matches AREA_CHOICES / SensorCalibratePanel). */
export type SensorAreaMode = "scene" | "circle" | "poly";

/** Singleton sensor kind (matches SINGLETON_CHOICES). */
export type SensorSingletonType = "environmental" | "attribute";

export interface SensorEntity extends ViewportEntityBase {
  type: "sensor";
  position: [number, number, number];
  /** Coverage area type: entire scene, circle, or polygon. */
  areaMode: SensorAreaMode;
  /** Coverage radius in meters (circle mode). */
  radius: number;
  /** Floor polygon vertices in meters (poly mode). Empty while drawing. */
  points: [number, number][];
  /** Environmental vs attribute analytics role. */
  singletonType: SensorSingletonType;
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
  regionDwells?: { regionId: string; seconds: number }[];
}

export interface ChildSceneEntity extends ViewportEntityBase {
  type: "child";
  geometry?: import("../placement/sceneGeometry").SceneGeometrySpec | null;
  /** Parent-scene pose translation [x, y, z] meters, Z-up. */
  position: [number, number, number];
  /** Euler degrees XYZ (Z-up), matching SceneEulerPose / REST transform. */
  rotation: [number, number, number];
  /** Non-uniform scale from the child link transform. */
  scale: [number, number, number];
  /** Local AABB size in meters before scale (map footprint). */
  size: [number, number, number];
  color: string;
  childType: string;
  restUid: string;
  remoteChildId: string | null;
  detailUrl: string | null;
  editHref: string;
  transformSource: string | null;
  deleteUrl: string | null;
}

/**
 * Singleton scene map GLB. Pose matches REST `mesh_translation` /
 * `mesh_rotation` / `mesh_scale` (Scene Manage → Pose).
 */
export interface MapEntity extends ViewportEntityBase {
  type: "map";
  position: [number, number, number];
  /** Euler degrees XYZ (Z-up), matching SceneEulerPose. */
  rotation: [number, number, number];
  scale: [number, number, number];
}

export type ViewportEntity =
  | RegionEntity
  | TripwireEntity
  | CameraEntity
  | SensorEntity
  | MarkEntity
  | ChildSceneEntity
  | MapEntity;

/** Stable viewport id for the scene map GLB singleton. */
export const SCENE_MAP_ENTITY_ID = "scene-map";

export type ViewPreset = "persp" | "top" | "front" | "side";

/** Phase 1.2 tool ids. `live` is the default navigate + click-to-select tool. */
export type ToolId =
  | "move"
  | "rotate"
  | "scale"
  | "region"
  | "tripwire"
  | "camera"
  | "sensor"
  | "measure"
  | "live";

export function isMark(e: ViewportEntity): e is MarkEntity {
  return e.type === "mark";
}
