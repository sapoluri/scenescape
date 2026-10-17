// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { restPoseToRigEuler } from "../viewport/entities";
import {
  listDirtyPersistableEntities,
  markEntityPersistClean,
  getPersistState,
  usePersistStore,
} from "../viewport/persistStore";
import { getViewportState, useViewportStore } from "../viewport/store";
import type {
  CameraEntity,
  MapEntity,
  RegionEntity,
  SensorEntity,
  TripwireEntity,
  ViewportEntity,
} from "../viewport/types";
import { persistGeometry } from "./legacyBridge";
import type { PersistGeometryResult } from "./roiPersist";
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
        points: e.points,
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
    case "map":
      return JSON.stringify({
        name: e.name,
        position: e.position,
        rotation: e.rotation,
        scale: e.scale,
      });
    default:
      return e.id;
  }
}

/** Persist scene map GLB TRS (same fields as Scene Manage → Pose). */
export async function persistMapEntity(
  authToken: string,
  sceneId: string,
  e: MapEntity,
): Promise<void> {
  await api.updateSceneJson(authToken, sceneId, {
    mesh_translation: e.position,
    mesh_rotation: e.rotation,
    mesh_scale: e.scale,
  });
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
    if (!Array.isArray(e.points) || e.points.length < 3) {
      throw new Error(
        "Polygon coverage needs at least 3 vertices — click the floor to draw",
      );
    }
    payload.points = e.points;
    payload.center = [e.position[0], e.position[1]];
  }
  if (e.areaMode === "scene") {
    payload.center = [e.position[0], e.position[1]];
  }
  await api.updateSensor(authToken, sensorId, payload);
}

/**
 * Rekey temp `draft-*` viewport ids to server uids after a geometry save.
 * Prevents duplicate Outliner rows when the geometry model is rekeyed and
 * reseeded while the store still holds the pre-save draft id.
 */
export function remapGeometryEntityIds(idMap: Record<string, string>): void {
  const s = getViewportState();
  let changed = false;
  const next = { ...s.entities };
  const baselines = { ...getPersistState().baselines };
  for (const [oldId, newId] of Object.entries(idMap)) {
    if (!oldId || !newId || oldId === newId) {
      continue;
    }
    const entity = next[oldId];
    if (!entity) {
      continue;
    }
    delete next[oldId];
    next[newId] = { ...entity, id: newId };
    if (oldId in baselines) {
      baselines[newId] = baselines[oldId];
      delete baselines[oldId];
    }
    changed = true;
  }
  if (!changed) {
    return;
  }
  usePersistStore.setState({ baselines });
  useViewportStore.setState((state) => ({
    entities: next,
    selectedId:
      state.selectedId && idMap[state.selectedId]
        ? idMap[state.selectedId]
        : state.selectedId,
    structVersion: state.structVersion + 1,
  }));
  getPersistState().captureBaselines(Object.values(idMap));
}

function applyGeometryPersistResult(
  geo: PersistGeometryResult | void | undefined,
): PersistGeometryResult | undefined {
  if (!geo || typeof geo !== "object") {
    return undefined;
  }
  if (geo.roiIds) {
    remapGeometryEntityIds(geo.roiIds);
  }
  if (geo.tripIds) {
    remapGeometryEntityIds(geo.tripIds);
  }
  return geo;
}

export async function persistRegionEntity(
  e: RegionEntity,
): Promise<PersistGeometryResult | undefined> {
  writeRegionToModel(e);
  const geo = await Promise.resolve(persistGeometry());
  const result = applyGeometryPersistResult(geo || undefined);
  const newId = result?.roiIds?.[e.id] ?? e.id;
  markEntityPersistClean(newId);
  return result;
}

export async function persistTripwireEntity(
  e: TripwireEntity,
): Promise<PersistGeometryResult | undefined> {
  writeTripwireToModel(e);
  const geo = await Promise.resolve(persistGeometry());
  const result = applyGeometryPersistResult(geo || undefined);
  const newId = result?.tripIds?.[e.id] ?? e.id;
  markEntityPersistClean(newId);
  return result;
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
      } else if (e.type === "map") {
        await persistMapEntity(args.authToken, args.sceneId, e);
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
        // Rekey draft-* store ids before marking clean so Outliner/baselines
        // do not keep a pre-save duplicate beside the server uid.
        applyGeometryPersistResult(geo);
      }
      for (const r of regions) {
        const newId = roiIds?.[r.id] ?? r.id;
        markEntityPersistClean(newId);
        saved += 1;
      }
      for (const t of trips) {
        const newId = tripIds?.[t.id] ?? t.id;
        markEntityPersistClean(newId);
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
