// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type {
  Camera,
  Mesh,
  MeshPhysicalMaterialParameters,
  Texture,
  Vector2,
} from "three";

declare module "./ProjectedMaterial.module.js" {
  export type ProjectedMaterialParameters = MeshPhysicalMaterialParameters & {
    camera: Camera;
    texture?: Texture;
    textureScale?: number;
    textureOffset?: Vector2;
    cover?: boolean;
    backgroundOpacity?: number;
  };

  export default class ProjectedMaterial extends import("three").MeshPhysicalMaterial {
    constructor(options: ProjectedMaterialParameters);
    texture: Texture;
    camera: Camera;
    project(mesh: Mesh): void;
    isProjectedMaterial: true;
  }

  export function allocateProjectionData(
    geometry: import("three").BufferGeometry,
    instancesCount: number,
  ): void;
}
