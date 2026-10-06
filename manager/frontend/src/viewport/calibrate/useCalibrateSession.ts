// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from "react";
import {
  Mesh,
  MeshBasicMaterial,
  Plane,
  Raycaster,
  SphereGeometry,
  Vector2,
  Vector3,
} from "three";
import { api } from "../../lib/rest";
import { restPoseToRigEuler } from "../entities";
import { getViewportState } from "../store";
import type { ViewportWorld } from "../world";
import { useCalibrateStore } from "./calibrateStore";
import { solveCameraPose } from "./poseSolve";
import type { CameraOptics } from "./types";

const FLOOR = new Plane(new Vector3(0, 0, 1), 0);

function parseOptics(row: Record<string, unknown>, fallbackName: string): CameraOptics {
  const intr = (row.intrinsics as Record<string, unknown>) || {};
  const dist = (row.distortion as Record<string, unknown>) || {};
  const res = (row.resolution as Record<string, unknown>) || {};
  const num = (v: unknown, d: number) => {
    const n = Number(v);
    return Number.isFinite(n) ? n : d;
  };
  const width = num(res.width, 1280);
  const height = num(res.height, 720);
  return {
    name: String(row.name || fallbackName || ""),
    fx: num(intr.fx, width),
    fy: num(intr.fy, width),
    cx: num(intr.cx, width / 2),
    cy: num(intr.cy, height / 2),
    k1: num(dist.k1, 0),
    k2: num(dist.k2, 0),
    p1: num(dist.p1, 0),
    p2: num(dist.p2, 0),
    k3: num(dist.k3, 0),
    width,
    height,
  };
}

/**
 * Loads optics, captures prior frustum pose, floor-picks scene points,
 * draws pair markers, and live-solves the camera frustum.
 */
export function useCalibrateSession(
  world: ViewportWorld | null,
  authToken: string,
): void {
  const active = useCalibrateStore((s) => s.active);
  const cameraId = useCalibrateStore((s) => s.cameraId);
  const sensorId = useCalibrateStore((s) => s.sensorId);
  const cameraName = useCalibrateStore((s) => s.cameraName);
  const pairs = useCalibrateStore((s) => s.pairs);
  const pending = useCalibrateStore((s) => s.pending);
  const draftCam = useCalibrateStore((s) => s.draftCam);
  const optics = useCalibrateStore((s) => s.optics);

  // Load optics + snapshot prior pose when entering / switching camera.
  useEffect(() => {
    if (!active || !cameraId || !sensorId || !authToken) {
      return;
    }
    let cancelled = false;
    const entity = getViewportState().entities[cameraId];
    if (entity?.type === "camera") {
      useCalibrateStore.getState().setPriorPose({
        position: [...entity.position] as [number, number, number],
        rotation: [...entity.rotation] as [number, number, number],
        fov: entity.fov,
      });
      getViewportState().select(cameraId);
    }
    (async () => {
      try {
        const row = (await api.getCamera(authToken, sensorId)) ?? {};
        if (cancelled) {
          return;
        }
        useCalibrateStore
          .getState()
          .setOptics(parseOptics(row, cameraName));
      } catch {
        if (!cancelled) {
          useCalibrateStore.getState().setOptics(
            parseOptics({}, cameraName),
          );
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [active, cameraId, sensorId, authToken, cameraName]);

  // Floor click → complete pair when waiting for scene point.
  useEffect(() => {
    if (!active || !world || pending !== "scene" || !draftCam) {
      return;
    }
    const host = world.host;
    const onPointerDown = (ev: PointerEvent) => {
      if (ev.button !== 0) {
        return;
      }
      const rect = host.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) {
        return;
      }
      const ndc = new Vector2(
        ((ev.clientX - rect.left) / rect.width) * 2 - 1,
        -(((ev.clientY - rect.top) / rect.height) * 2 - 1),
      );
      const raycaster = new Raycaster();
      raycaster.setFromCamera(ndc, world.getActiveCamera());
      const hit = new Vector3();
      if (!raycaster.ray.intersectPlane(FLOOR, hit)) {
        return;
      }
      ev.preventDefault();
      ev.stopPropagation();
      useCalibrateStore.getState().addPair({
        cam: draftCam,
        map: [hit.x, hit.y, hit.z],
      });
    };
    host.addEventListener("pointerdown", onPointerDown, true);
    return () => host.removeEventListener("pointerdown", onPointerDown, true);
  }, [active, world, pending, draftCam]);

  // Floor markers for completed pairs.
  useEffect(() => {
    if (!world) {
      return;
    }
    const group = world.entityRoot;
    const markers: Mesh[] = [];
    const mat = new MeshBasicMaterial({ color: 0x30d158 });
    const geo = new SphereGeometry(0.18, 12, 12);
    if (active) {
      pairs.forEach((p, i) => {
        const m = new Mesh(geo, mat);
        m.position.set(p.map[0], p.map[1], p.map[2] + 0.18);
        m.name = `ss-cal-marker-${i}`;
        m.userData.ssCalMarker = true;
        group.add(m);
        markers.push(m);
      });
    }
    return () => {
      markers.forEach((m) => {
        group.remove(m);
      });
      geo.dispose();
      mat.dispose();
    };
  }, [world, active, pairs]);

  // Live solve → update frustum.
  useEffect(() => {
    if (!active || !cameraId || !optics || pairs.length < 4) {
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const pose = await solveCameraPose(pairs, optics);
        if (cancelled || !pose) {
          return;
        }
        getViewportState().updateEntity(cameraId, {
          position: pose.position,
          rotation: restPoseToRigEuler(pose.rotation),
          fov: pose.fov,
        });
      } catch (err) {
        console.warn("Calibrate pose solve failed", err);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [active, cameraId, optics, pairs]);
}
