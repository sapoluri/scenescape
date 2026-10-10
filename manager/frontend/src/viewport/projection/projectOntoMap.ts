// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
  Box3,
  BufferGeometry,
  Mesh,
  MeshPhysicalMaterial,
  Object3D,
  PerspectiveCamera,
  SRGBColorSpace,
  Texture,
  TextureLoader,
  Vector3,
} from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { ProjectedMaterial } from "./ProjectedMaterial";

/** Vendored projected material behaves as MeshPhysicalMaterial + project(). */
type ProjMat = MeshPhysicalMaterial & {
  project: (mesh: Mesh) => void;
  texture: Texture;
};

export type CameraProjectionHandle = {
  mesh: Mesh;
  material: ProjMat;
  camera: PerspectiveCamera;
  setOpacity: (opacity: number) => void;
  setVisible: (visible: boolean) => void;
  updateFrame: (
    imageUrl: string,
    opts?: {
      fov?: number;
      aspect?: number;
      position?: [number, number, number];
      /**
       * Rig XYZ euler degrees (Z-up), same as CameraEntity.rotation /
       * frustum — REST solve pose × Rx(π). Not raw REST euler.
       */
      rotationDeg?: [number, number, number];
      /**
       * When false, only swap the JPEG (pose uniforms already saved).
       * Default true. Use false for live-frame refresh to avoid main-thread
       * stalls every MQTT tick after a solve.
       */
      reproject?: boolean;
    },
  ) => Promise<void>;
  /** Update projector pose without reloading the texture. */
  updatePose: (opts: {
    fov?: number;
    aspect?: number;
    position?: [number, number, number];
    rotationDeg?: [number, number, number];
  }) => void;
  dispose: () => void;
};

function extractGeometry(root: Object3D): BufferGeometry {
  const geos: BufferGeometry[] = [];
  root.traverse((child) => {
    const mesh = child as Mesh;
    if (mesh.isMesh && mesh.geometry) {
      const g = mesh.geometry.clone();
      mesh.updateWorldMatrix(true, false);
      g.applyMatrix4(mesh.matrixWorld);
      geos.push(g);
    }
  });
  if (geos.length === 0) {
    throw new Error("Scene map has no mesh geometry to project onto");
  }
  const merged = mergeGeometries(geos, false);
  geos.forEach((g) => g.dispose());
  if (!merged) {
    throw new Error("Failed to merge scene map geometry for projection");
  }
  merged.computeVertexNormals();
  return merged;
}

const textureLoader = new TextureLoader();
textureLoader.setCrossOrigin("anonymous");

/**
 * Load a projection frame. MQTT live previews are `data:image/jpeg;base64,…`
 * — TextureLoader with crossOrigin:"anonymous" fails on data:/blob: URLs in
 * Chromium, so those go through a plain Image (legacy scenecamera.js path).
 */
function loadProjectionTexture(imageUrl: string): Promise<Texture> {
  if (imageUrl.startsWith("data:") || imageUrl.startsWith("blob:")) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const texture = new Texture(img);
        texture.colorSpace = SRGBColorSpace;
        texture.needsUpdate = true;
        resolve(texture);
      };
      img.onerror = () =>
        reject(new Error("Failed to decode projection frame image"));
      img.src = imageUrl;
    });
  }
  return textureLoader.loadAsync(imageUrl).then((texture) => {
    texture.colorSpace = SRGBColorSpace;
    texture.needsUpdate = true;
    return texture;
  });
}

/**
 * Create a camera-coverage overlay that projects a JPEG onto the scene map
 * mesh (same contract as legacy Draw.createProjectionMaterial +
 * Viewport.projectImage).
 */
export function createCameraProjection(
  mapRoot: Object3D,
  parent: Object3D,
): CameraProjectionHandle {
  const projCam = new PerspectiveCamera(60, 4 / 3, 0.1, 100);
  projCam.up.set(0, 0, 1);
  const placeholder = new Texture();
  const projectedMaterial = new ProjectedMaterial({
    camera: projCam,
    texture: placeholder,
    opacity: 0.55,
    transparent: true,
    depthWrite: false,
    toneMapped: false,
  }) as unknown as ProjMat;
  const physicalMaterial = new MeshPhysicalMaterial({
    color: 0xffffff,
    opacity: 0.01,
    transparent: true,
    depthWrite: false,
  });
  const geometry = extractGeometry(mapRoot);
  // Three.js skips draws when group.count === Infinity; use real index/vertex
  // count (legacy draw.js used Infinity against older three builds).
  geometry.clearGroups();
  const groupCount = geometry.index?.count ?? geometry.attributes.position.count;
  const mesh = new Mesh(geometry, [physicalMaterial, projectedMaterial]);
  for (let i = 0; i < 2; i++) {
    geometry.addGroup(0, groupCount, i);
  }
  mesh.position.z += 0.002;
  mesh.renderOrder = 2;
  mesh.visible = false;
  projectedMaterial.project(mesh);
  parent.add(mesh);

  const d2r = (d: number) => (d * Math.PI) / 180;

  const setOpacity = (opacity: number) => {
    const clamped = Math.min(1, Math.max(0, opacity));
    projectedMaterial.transparent = true;
    projectedMaterial.opacity = clamped;
    projectedMaterial.depthWrite = false;
    projectedMaterial.needsUpdate = true;
    const u = (
      projectedMaterial as unknown as {
        uniforms?: { opacity?: { value: number } };
      }
    ).uniforms;
    if (u?.opacity) {
      u.opacity.value = clamped;
    }
    mesh.visible = clamped > 0;
    projectedMaterial.visible = clamped > 0;
  };

  const setVisible = (visible: boolean) => {
    mesh.visible = visible && projectedMaterial.opacity > 0;
    projectedMaterial.visible = mesh.visible;
  };

  let currentTexture: Texture | null = null;

  const applyPose = (opts: {
    fov?: number;
    aspect?: number;
    position?: [number, number, number];
    rotationDeg?: [number, number, number];
  }) => {
    if (opts.aspect != null && Number.isFinite(opts.aspect)) {
      projCam.aspect = opts.aspect;
    }
    if (opts.fov != null && Number.isFinite(opts.fov)) {
      projCam.fov = opts.fov;
    }
    if (opts.position) {
      projCam.position.set(
        opts.position[0],
        opts.position[1],
        opts.position[2],
      );
    }
    if (opts.rotationDeg) {
      projCam.rotation.set(
        d2r(opts.rotationDeg[0]),
        d2r(opts.rotationDeg[1]),
        d2r(opts.rotationDeg[2]),
        "XYZ",
      );
    }
    projCam.updateProjectionMatrix();
  };

  const updatePose = (opts: {
    fov?: number;
    aspect?: number;
    position?: [number, number, number];
    rotationDeg?: [number, number, number];
  }) => {
    applyPose(opts);
    projectedMaterial.project(mesh);
    setVisible(true);
  };

  const updateFrame = async (
    imageUrl: string,
    opts: {
      fov?: number;
      aspect?: number;
      position?: [number, number, number];
      rotationDeg?: [number, number, number];
      reproject?: boolean;
    } = {},
  ) => {
    const texture = await loadProjectionTexture(imageUrl);
    const img = texture.image as {
      width?: number;
      height?: number;
      naturalWidth?: number;
      naturalHeight?: number;
    };
    const w = img?.naturalWidth || img?.width || 1280;
    const h = img?.naturalHeight || img?.height || 720;
    applyPose({
      ...opts,
      aspect: opts.aspect ?? w / h,
    });
    const prev = currentTexture;
    currentTexture = texture;
    projectedMaterial.texture = texture;
    // Texture-only refreshes skip project() — camera matrices stay valid.
    if (opts.reproject !== false) {
      projectedMaterial.project(mesh);
    }
    setVisible(true);
    if (prev && prev !== placeholder) {
      prev.dispose();
    }
  };

  return {
    mesh,
    material: projectedMaterial,
    camera: projCam,
    setOpacity,
    setVisible,
    updateFrame,
    updatePose,
    dispose: () => {
      parent.remove(mesh);
      geometry.dispose();
      physicalMaterial.dispose();
      projectedMaterial.dispose();
      if (currentTexture && currentTexture !== placeholder) {
        currentTexture.dispose();
      }
      placeholder.dispose();
    },
  };
}

/** World-space AABB of the map (for orbit framing). */
export function mapExtentMeters(mapRoot: Object3D): {
  width: number;
  height: number;
  center: Vector3;
} {
  const box = new Box3().setFromObject(mapRoot);
  const size = new Vector3();
  const center = new Vector3();
  box.getSize(size);
  box.getCenter(center);
  return { width: size.x, height: size.y, center };
}
