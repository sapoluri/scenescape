// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { restPoseToRigEuler } from "../viewport/entities";
import {
  listDirtyPersistableEntities,
  markEntityPersistClean,
} from "../viewport/persistStore";
import { getViewportState } from "../viewport/store";
import type {
  CameraEntity,
  RegionEntity,
  SensorEntity,
  TripwireEntity,
  ViewportEntity,
} from "../viewport/types";
import { persistGeometry } from "./legacyBridge";
import { api } from "./rest";
import {
  writeRegionToModel,
  writeTripwireToModel,
} from "../viewport/geometrySync";

export type PersistBootstrapCam = {
  id: string;
  sensorId: string;
};

export type PersistBootstrapSensor = {
  id: string;
  sensorId: string;
};

export type SaveAllResult = {
  saved: number;
  errors: string[];
  /** Draft → server uid maps from geometry persist (when geometry ran). */
  roiIds?: Record<string, string>;
  tripIds?: Record<string, string>;
};

/** Serializable fingerprint for dirty detection in Properties. */
export function entityPersistKey(e: ViewportEntity): string {
  switch (e.type) {
    case "camera":
      return JSON.stringify({
        name: e.name,
        position: e.position,
        rotation: e.rotation,
        fov: e.fov,
      });
    case "sensor":
      return JSON.stringify({
        name: e.name,
        position: e.position,
        areaMode: e.areaMode,
        radius: e.radius,
        color: e.color,
      });
    case "region":
      return JSON.stringify({
        name: e.name,
        color: e.color,
        height: e.height,
        bufferSize: e.bufferSize,
        volumetric: e.volumetric,
        points: e.points,
      });
    case "tripwire":
      return JSON.stringify({
        name: e.name,
        color: e.color,
        points: e.points,
      });
    default:
      return e.id;
  }
}

export async function persistCameraEntity(
  authToken: string,
  sceneId: string,
  sensorId: string,
  e: CameraEntity,
  optics?: {
    fx: number;
    fy: number;
    cx: number;
    cy: number;
    k1: number;
    k2: number;
    p1: number;
    p2: number;
    k3: number;
    width: number;
    height: number;
  },
): Promise<void> {
  // Rig euler → REST euler (same Rx(π) involution as restPoseToRigEuler).
  const restRotation = restPoseToRigEuler(e.rotation);
  const payload: Record<string, unknown> = {
    name: e.name,
    sensor_id: sensorId,
    scene: sceneId,
    transform_type: "euler",
    translation: e.position,
    rotation: restRotation,
    scale: [1, 1, 1],
  };
  if (optics) {
    payload.intrinsics = {
      fx: optics.fx,
      fy: optics.fy,
      cx: optics.cx,
      cy: optics.cy,
    };
    payload.distortion = {
      k1: optics.k1,
      k2: optics.k2,
      p1: optics.p1,
      p2: optics.p2,
      k3: optics.k3,
    };
    payload.resolution = { width: optics.width, height: optics.height };
  }
  await api.updateCamera(authToken, sensorId, payload);
}

export async function persistSensorEntity(
  authToken: string,
  sceneId: string,
  sensorId: string,
  e: SensorEntity,
): Promise<void> {
  const payload: Record<string, unknown> = {
    name: e.name,
    sensor_id: sensorId,
    scene: sceneId,
    area: e.areaMode,
  };
  if (e.areaMode === "circle") {
    payload.center = [e.position[0], e.position[1]];
    payload.radius = e.radius;
  }
  if (e.areaMode === "poly") {
    const row = (await api.getSensor(authToken, sensorId)) ?? {};
    const points = row.points;
    if (!Array.isArray(points) || points.length < 3) {
      throw new Error(
        "Polygon coverage needs vertices — use Edit polygon on map first",
      );
    }
    payload.points = points;
    // Keep map position aligned with the polygon centroid when possible.
    payload.center = [e.position[0], e.position[1]];
  }
  if (e.areaMode === "scene") {
    payload.center = [e.position[0], e.position[1]];
  }
  await api.updateSensor(authToken, sensorId, payload);
}

export async function persistRegionEntity(e: RegionEntity): Promise<void> {
  writeRegionToModel(e);
  await Promise.resolve(persistGeometry());
}

export async function persistTripwireEntity(e: TripwireEntity): Promise<void> {
  writeTripwireToModel(e);
  await Promise.resolve(persistGeometry());
}

/**
 * Bulk-save every dirty camera/sensor/region/tripwire (and flush geometry
 * once for regions/tripwires). Continues after individual failures.
 */
export async function saveAllDirtyEntities(args: {
  authToken: string;
  sceneId: string;
  cameras: PersistBootstrapCam[];
  sensors: PersistBootstrapSensor[];
  /** Also flush geometry when the 2D dirty flags are set. */
  geometryDirty?: boolean;
}): Promise<SaveAllResult> {
  const dirty = listDirtyPersistableEntities();
  const errors: string[] = [];
  let saved = 0;

  const camById = new Map(args.cameras.map((c) => [c.id, c]));
  const senById = new Map(args.sensors.map((s) => [s.id, s]));

  const regions: RegionEntity[] = [];
  const trips: TripwireEntity[] = [];

  for (const e of dirty) {
    try {
      if (e.type === "camera") {
        const boot = camById.get(e.id);
        if (!boot) {
          errors.push(`${e.name}: not linked to a server camera`);
          continue;
        }
        await persistCameraEntity(
          args.authToken,
          args.sceneId,
          boot.sensorId,
          e,
        );
        const restRotation = restPoseToRigEuler(e.rotation);
        getViewportState().updateEntity(e.id, {
          restRotation,
          posed: true,
        });
        markEntityPersistClean(e.id);
        saved += 1;
      } else if (e.type === "sensor") {
        const boot = senById.get(e.id);
        if (!boot) {
          errors.push(`${e.name}: not linked to a server sensor`);
          continue;
        }
        await persistSensorEntity(
          args.authToken,
          args.sceneId,
          boot.sensorId,
          e,
        );
        markEntityPersistClean(e.id);
        saved += 1;
      } else if (e.type === "region") {
        regions.push(e);
      } else if (e.type === "tripwire") {
        trips.push(e);
      }
    } catch (err) {
      errors.push(
        `${e.name}: ${err instanceof Error ? err.message : "Save failed"}`,
      );
    }
  }

  let roiIds: Record<string, string> | undefined;
  let tripIds: Record<string, string> | undefined;

  const needGeometry =
    regions.length > 0 || trips.length > 0 || Boolean(args.geometryDirty);
  if (needGeometry) {
    try {
      for (const r of regions) {
        writeRegionToModel(r);
      }
      for (const t of trips) {
        writeTripwireToModel(t);
      }
      const geo = await Promise.resolve(persistGeometry());
      if (geo && typeof geo === "object") {
        roiIds = geo.roiIds;
        tripIds = geo.tripIds;
      }
      // Mark clean under current (pre-remap) ids; caller remaps store + baselines.
      for (const r of regions) {
        markEntityPersistClean(r.id);
        saved += 1;
      }
      for (const t of trips) {
        markEntityPersistClean(t.id);
        saved += 1;
      }
      // Geometry-only dirty (no entity fingerprint change) still counts.
      if (
        regions.length === 0 &&
        trips.length === 0 &&
        args.geometryDirty
      ) {
        saved += 1;
      }
    } catch (err) {
      errors.push(
        `Geometry: ${err instanceof Error ? err.message : "Save failed"}`,
      );
    }
  }

  return { saved, errors, roiIds, tripIds };
}
