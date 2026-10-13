// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Group, Object3D } from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import {
  applyDefaultPose,
  listAssets,
  type Asset3DRecord,
} from "../library/AssetLibrary";

/**
 * Object Library cache for live 3D marks.
 *
 * Runtime composition (same as MarkEditor):
 *   tracker transform × Asset3D default pose × GLB
 *
 * Templates are cloned per mark instance. Assets without a GLB leave
 * `template` null so the renderer can fall back to a footprint box.
 */

export type MarkAssetTemplate = {
  record: Asset3DRecord;
  /** Pose frame holding the GLB; null when the class has no model. */
  template: Group | null;
};

type Listener = () => void;

export class MarkAssetCatalog {
  private byName = new Map<string, MarkAssetTemplate>();
  private byNameLower = new Map<string, string>();
  private loading: Promise<void> | null = null;
  private listeners = new Set<Listener>();
  private disposed = false;
  private readonly loader = new GLTFLoader();

  onChange(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private emit(): void {
    for (const l of this.listeners) {
      try {
        l();
      } catch {
        /* listener must not break the catalog */
      }
    }
  }

  lookup(className: string): MarkAssetTemplate | null {
    const exact = this.byName.get(className);
    if (exact) {
      return exact;
    }
    const key = this.byNameLower.get(className.toLowerCase());
    return key ? (this.byName.get(key) ?? null) : null;
  }

  /** Clone the class GLB under a fresh pose frame, or null if none. */
  cloneModel(className: string): Group | null {
    const entry = this.lookup(className);
    if (!entry?.template) {
      return null;
    }
    return entry.template.clone(true);
  }

  assetOf(className: string): Asset3DRecord | null {
    return this.lookup(className)?.record ?? null;
  }

  /**
   * Load (or reload) Object Library rows into the mark template cache.
   * Pass `force: true` after a library save so GLBs/poses refresh even if a
   * prior load is still in flight; model URLs are cache-busted on force.
   */
  async load(
    authToken: string,
    opts: { force?: boolean } = {},
  ): Promise<void> {
    if (!authToken || this.disposed) {
      return;
    }
    if (this.loading) {
      await this.loading;
      if (this.disposed) {
        return;
      }
      if (!opts.force) {
        return;
      }
    }
    this.loading = this.loadInner(authToken, Boolean(opts.force)).finally(
      () => {
        this.loading = null;
      },
    );
    return this.loading;
  }

  private async loadInner(
    authToken: string,
    bustCache: boolean,
  ): Promise<void> {
    let rows: Asset3DRecord[] = [];
    try {
      rows = await listAssets(authToken);
    } catch (err) {
      console.warn("Object Library load failed for 3D marks", err);
      return;
    }
    if (this.disposed) {
      return;
    }

    const next = new Map<string, MarkAssetTemplate>();
    const nextLower = new Map<string, string>();
    const cacheTag = bustCache ? `t=${Date.now()}` : "";
    await Promise.all(
      rows.map(async (rec) => {
        if (!rec.name) {
          return;
        }
        let template: Group | null = null;
        if (rec.modelUrl) {
          try {
            const url = cacheTag
              ? `${rec.modelUrl}${rec.modelUrl.includes("?") ? "&" : "?"}${cacheTag}`
              : rec.modelUrl;
            const gltf = await this.loader.loadAsync(url);
            if (this.disposed) {
              return;
            }
            template = new Group();
            template.name = "default-pose";
            const model = gltf.scene;
            model.name = "model";
            template.add(model);
            applyDefaultPose(template, rec);
          } catch (err) {
            console.warn(`Failed to load mark GLB for ${rec.name}`, err);
            template = null;
          }
        }
        next.set(rec.name, { record: rec, template });
        nextLower.set(rec.name.toLowerCase(), rec.name);
      }),
    );

    if (this.disposed) {
      return;
    }
    this.byName = next;
    this.byNameLower = nextLower;
    this.emit();
  }

  dispose(): void {
    this.disposed = true;
    this.listeners.clear();
    this.byName.clear();
    this.byNameLower.clear();
  }
}

/** Apply the Asset3D default pose to a pose-frame object (exported for tests). */
export function poseMarkModel(poseGroup: Object3D, rec: Asset3DRecord): void {
  applyDefaultPose(poseGroup, rec);
}

/** Frozen event name: Object Library row saved / deleted — reload live marks. */
export const LIBRARY_CHANGED_EVENT = "ss-library-changed";

/** Notify the 3D mark renderer that Asset3D definitions changed. */
export function notifyLibraryChanged(): void {
  window.dispatchEvent(new CustomEvent(LIBRARY_CHANGED_EVENT));
}
