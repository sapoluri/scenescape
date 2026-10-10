// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef } from "react";
import { Box3, Vector3 } from "three";
import { api } from "../lib/rest";
import { loadSceneObject } from "../placement/loadSceneObject";
import {
  isGlbUrl,
  sceneGeometryFromRest,
  type SceneGeometrySpec,
} from "../placement/sceneGeometry";
import { IDENTITY_POSE } from "../placement/placementTypes";
import {
  createCameraProjection,
  type CameraProjectionHandle,
} from "./projection/projectOntoMap";
import type { ViewportWorld } from "./world";

export type SceneMapProps = {
  sceneId: string;
  sceneName: string;
  mapUrl: string | null;
  mapScale: number | null;
  authToken: string;
};

/**
 * Load the scene floor map (PNG plane or posed GLB) into `world.mapRoot` and
 * expose a camera-projection handle for coverage overlays (calibrate / live).
 */
export function useSceneMap(
  world: ViewportWorld | null,
  props: SceneMapProps,
): { projectionReady: boolean } {
  const projectionRef = useRef<CameraProjectionHandle | null>(null);
  const readyRef = useRef(false);

  useEffect(() => {
    if (!world) {
      return;
    }
    let cancelled = false;

    const clearMap = () => {
      projectionRef.current?.dispose();
      projectionRef.current = null;
      world.cameraProjection = null;
      while (world.mapRoot.children.length) {
        const child = world.mapRoot.children[0];
        world.mapRoot.remove(child);
        child.traverse((o) => {
          const mesh = o as {
            geometry?: { dispose?: () => void };
            material?:
              | { dispose?: () => void }
              | { dispose?: () => void }[];
          };
          mesh.geometry?.dispose?.();
          if (Array.isArray(mesh.material)) {
            mesh.material.forEach((m) => m.dispose?.());
          } else {
            mesh.material?.dispose?.();
          }
        });
      }
      readyRef.current = false;
    };

    const frameOrbit = () => {
      const box = new Box3().setFromObject(world.mapRoot);
      if (box.isEmpty()) {
        return;
      }
      const size = new Vector3();
      const center = new Vector3();
      box.getSize(size);
      box.getCenter(center);
      const span = Math.max(size.x, size.y, 8);
      world.orbit.target.copy(center);
      world.perspCam.position.set(
        center.x,
        center.y - span * 0.85,
        center.z + span * 0.65,
      );
      world.perspCam.near = 0.1;
      world.perspCam.far = Math.max(5000, span * 20);
      world.perspCam.updateProjectionMatrix();
      world.orbit.update();
    };

    const bootstrapSpec = (): SceneGeometrySpec | null => {
      if (!props.mapUrl) {
        return null;
      }
      return {
        id: props.sceneId,
        name: props.sceneName,
        mapUrl: props.mapUrl,
        scale: props.mapScale,
        isGlb: isGlbUrl(props.mapUrl),
        meshPose: { ...IDENTITY_POSE },
      };
    };

    (async () => {
      clearMap();
      let spec = bootstrapSpec();
      // GLB mesh pose lives on REST, not bootstrap — fetch when needed.
      if (props.authToken && (spec?.isGlb || !spec)) {
        try {
          const row = await api.getScene(props.authToken, props.sceneId);
          if (cancelled) {
            return;
          }
          const fromRest = sceneGeometryFromRest(row ?? {});
          if (fromRest.mapUrl) {
            spec = fromRest;
          }
        } catch (err) {
          console.warn("Scene map REST fetch failed", err);
        }
      }
      if (!spec?.mapUrl || cancelled) {
        return;
      }
      try {
        const object = await loadSceneObject(spec);
        if (cancelled) {
          return;
        }
        world.mapRoot.add(object);
        world.sceneMesh = object;
        try {
          projectionRef.current = createCameraProjection(
            object,
            world.mapRoot,
          );
          world.cameraProjection = projectionRef.current;
        } catch (err) {
          console.warn("Camera projection layer unavailable", err);
        }
        frameOrbit();
        readyRef.current = true;
      } catch (err) {
        console.warn("Failed to load scene map into 3D viewport", err);
      }
    })();

    return () => {
      cancelled = true;
      clearMap();
      world.sceneMesh = null;
    };
  }, [
    world,
    props.sceneId,
    props.sceneName,
    props.mapUrl,
    props.mapScale,
    props.authToken,
  ]);

  return { projectionReady: readyRef.current };
}
