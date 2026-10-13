// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
  Box3,
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  Color,
  Group,
  Line,
  LineBasicMaterial,
  Mesh,
  MeshStandardMaterial,
  Object3D,
  Sprite,
  SpriteMaterial,
  Vector3,
} from "three";
import {
  applyDefaultPose,
  createMarkPrimitiveGeometry,
  footprintOf,
  parseMarkGeometry,
  type Asset3DRecord,
} from "../library/AssetLibrary";
import {
  LIBRARY_CHANGED_EVENT,
  MarkAssetCatalog,
} from "./markAssets";
import {
  colorFromTrackId,
  trackIdFromMarkKey,
} from "./marks";
import { getViewportState, useViewportStore } from "./store";
import type { MarkEntity } from "./types";
import type { ViewportWorld } from "./world";

/**
 * Imperative three.js rendering for live tracked-object marks.
 *
 * Composition matches the Object Library mark editor:
 *   tracker transform × Asset3D default pose × (GLB | mark geometry)
 *
 * Classes without a GLB use Asset3D `mark_geometry` (box / capsule /
 * cylinder) sized to the class footprint, with the default pose applied.
 *
 * Trail + hover triangle use the 2D UUID track color (`colorFromTrackId`);
 * the body keeps the Asset3D mark color.
 */

const TRAIL_POINTS = 40;
const LABEL_Z_FALLBACK = 1.6;
const INDICATOR_GAP = 0.5;
const LABEL_GAP = 1.2;

/** Fingerprint of library fields that affect mark body geometry/appearance. */
function assetBodySig(asset: Asset3DRecord | null, className: string): string {
  if (!asset) {
    return `none:${className}`;
  }
  return [
    asset.uid,
    asset.modelUrl || "",
    asset.markColor,
    parseMarkGeometry(asset.markGeometry),
    asset.xSize,
    asset.ySize,
    asset.zSize,
    asset.rotation.join(","),
    asset.translation.join(","),
    asset.scale.join(","),
  ].join("|");
}

interface MarkNodes {
  group: Group;
  /** Pose / model holder; replaced when the library asset resolves. */
  bodyRoot: Group;
  /** Downward triangle sprite above the body (UUID track color). */
  indicator: Sprite;
  trail: Line;
  className: string;
  /** True when bodyRoot holds a cloned library GLB (not a primitive). */
  hasGlb: boolean;
  /** Library pose/size/color/model fingerprint for this body. */
  assetSig: string;
  /** World/local Z for the HTML label (above the indicator). */
  labelZ: number;
  trackColor: string;
}

function disposeObjectTree(root: Object3D): void {
  root.traverse((o) => {
    const mesh = o as Mesh;
    mesh.geometry?.dispose?.();
    const sprite = o as Sprite;
    const mat = (mesh.material ?? sprite.material) as
      | { dispose?: () => void; map?: { dispose?: () => void } }
      | { dispose?: () => void; map?: { dispose?: () => void } }[]
      | undefined;
    if (Array.isArray(mat)) {
      mat.forEach((m) => {
        m.map?.dispose?.();
        m.dispose?.();
      });
    } else {
      mat?.map?.dispose?.();
      mat?.dispose?.();
    }
  });
}

/**
 * Downward triangle hover marker — port of legacy `createIndicator` in
 * draw.js, colored with the 2D track id rule.
 */
function createTrackIndicator(trackColor: string): Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.fillStyle = trackColor;
    ctx.beginPath();
    ctx.moveTo(32, 56); // bottom point
    ctx.lineTo(8, 8); // top left
    ctx.lineTo(56, 8); // top right
    ctx.closePath();
    ctx.fill();
  }
  const texture = new CanvasTexture(canvas);
  const sprite = new Sprite(
    new SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
    }),
  );
  sprite.name = "indicator";
  sprite.scale.set(1, 1, 1);
  return sprite;
}

/**
 * Build the mark body root: a pose frame holding either a cloned library GLB
 * or a footprint/primitive placeholder. Always applies Asset3D default pose
 * when a library row exists — including when there is no GLB.
 */
function buildMarkBody(
  entity: MarkEntity,
  catalog: MarkAssetCatalog,
): { bodyRoot: Group; hasGlb: boolean } {
  const asset = catalog.assetOf(entity.className);
  const glb = catalog.cloneModel(entity.className);
  if (glb) {
    // cloneModel already includes applyDefaultPose on the pose frame.
    glb.name = "mark-body";
    return { bodyRoot: glb, hasGlb: true };
  }

  const pose = new Group();
  pose.name = "mark-body";
  const color = new Color(asset?.markColor || entity.color);
  const kind = asset
    ? parseMarkGeometry(asset.markGeometry)
    : entity.className.toLowerCase().includes("person")
      ? "capsule"
      : "box";
  const size = asset
    ? (() => {
        const fp = footprintOf(asset);
        return { x: fp.inner.x, y: fp.inner.y, z: fp.height };
      })()
    : kind === "capsule"
      ? { x: 0.56, y: 0.56, z: 1.7 }
      : { x: 0.9, y: 0.9, z: 1.1 };
  const body = new Mesh(
    createMarkPrimitiveGeometry(kind, size),
    new MeshStandardMaterial({
      color,
      roughness: 0.55,
      transparent: true,
      opacity: 0.9,
    }),
  );
  body.name = "model";
  body.position.z = size.z / 2;
  pose.add(body);
  if (asset) {
    // Geometry already uses absolute x/y/z meters — skip GLB pose scale.
    applyDefaultPose(pose, asset, { applyScale: false });
  }
  return { bodyRoot: pose, hasGlb: false };
}

function bodyTopZ(root: Object3D): number {
  root.updateMatrixWorld(true);
  const box = new Box3().setFromObject(root);
  if (box.isEmpty()) {
    return LABEL_Z_FALLBACK - LABEL_GAP;
  }
  return box.max.z;
}

/** Place indicator + label height from the current body bounds (legacy gaps). */
function layoutChrome(nodes: MarkNodes): void {
  const top = bodyTopZ(nodes.bodyRoot);
  nodes.indicator.position.set(0, 0, top + INDICATOR_GAP);
  nodes.labelZ = Math.max(top + LABEL_GAP, LABEL_Z_FALLBACK);
}

function trackColorFor(entity: MarkEntity): string {
  return colorFromTrackId(trackIdFromMarkKey(entity.id));
}

function buildMarkNodes(
  entity: MarkEntity,
  catalog: MarkAssetCatalog,
): MarkNodes {
  const group = new Group();
  const asset = catalog.assetOf(entity.className);
  const { bodyRoot, hasGlb } = buildMarkBody(entity, catalog);
  const trackColor = trackColorFor(entity);
  const track = new Color(trackColor);

  const indicator = createTrackIndicator(trackColor);
  const trailGeo = new BufferGeometry();
  trailGeo.setAttribute(
    "position",
    new BufferAttribute(new Float32Array(TRAIL_POINTS * 3), 3),
  );
  const trail = new Line(
    trailGeo,
    new LineBasicMaterial({ color: track, transparent: true, opacity: 0.5 }),
  );
  trail.frustumCulled = false;
  group.add(bodyRoot, indicator, trail);
  const nodes: MarkNodes = {
    group,
    bodyRoot,
    indicator,
    trail,
    className: entity.className,
    hasGlb,
    assetSig: assetBodySig(asset, entity.className),
    labelZ: LABEL_Z_FALLBACK,
    trackColor,
  };
  layoutChrome(nodes);
  return nodes;
}

function rebuildBody(
  nodes: MarkNodes,
  entity: MarkEntity,
  catalog: MarkAssetCatalog,
): void {
  nodes.group.remove(nodes.bodyRoot);
  disposeObjectTree(nodes.bodyRoot);
  const asset = catalog.assetOf(entity.className);
  const built = buildMarkBody(entity, catalog);
  nodes.bodyRoot = built.bodyRoot;
  nodes.hasGlb = built.hasGlb;
  nodes.className = entity.className;
  nodes.assetSig = assetBodySig(asset, entity.className);
  nodes.group.add(nodes.bodyRoot, nodes.indicator, nodes.trail);
  layoutChrome(nodes);
}

function disposeMarkNodes(nodes: MarkNodes): void {
  disposeObjectTree(nodes.group);
}

export type AttachMarkRendererOptions = {
  authToken?: string;
};

/** Attach mark rendering to a live world. Returns a dispose function. */
export function attachMarkRenderer(
  world: ViewportWorld,
  options: AttachMarkRendererOptions = {},
): () => void {
  const { markRoot, labelLayer, host, onFrame } = world;
  const catalog = new MarkAssetCatalog();
  const markNodes = new Map<string, MarkNodes>();
  const labelEls = new Map<string, HTMLDivElement>();
  const trailPts = new Map<string, Vector3[]>();

  const ensureLabel = (id: string, name: string) => {
    let el = labelEls.get(id);
    if (!el) {
      el = document.createElement("div");
      el.className = "ss-vlabel ss-vlabel--mark";
      el.textContent = name;
      labelLayer.appendChild(el);
      labelEls.set(id, el);
    }
    return el;
  };

  const updateTrail = (id: string, x: number, y: number, show: boolean) => {
    const nodes = markNodes.get(id);
    if (!nodes) {
      return;
    }
    let pts = trailPts.get(id);
    if (!pts) {
      pts = [];
      trailPts.set(id, pts);
    }
    // Trail samples are stored in world XY; the Line is parented under the
    // mark group, so write local coords (world − current mark position) or
    // the trail becomes random squiggles as the object moves.
    const last = pts[0];
    if (!last || Math.hypot(last.x - x, last.y - y) > 0.05) {
      pts.unshift(new Vector3(x, y, 0.08));
      if (pts.length > TRAIL_POINTS) {
        pts.pop();
      }
    }
    const attr = nodes.trail.geometry.getAttribute(
      "position",
    ) as BufferAttribute;
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      attr.setXYZ(i, p.x - x, p.y - y, p.z);
    }
    // Clear unused slots so a shorter trail doesn't leave a stale segment.
    for (let i = pts.length; i < TRAIL_POINTS; i++) {
      attr.setXYZ(i, 0, 0, 0.08);
    }
    attr.needsUpdate = true;
    nodes.trail.geometry.setDrawRange(0, Math.max(pts.length, 1));
    nodes.trail.visible = show && pts.length >= 2;
  };

  const reconcileMarks = () => {
    const { entities, showTrails } = getViewportState();
    const seen = new Set<string>();
    for (const e of Object.values(entities)) {
      if (e.type !== "mark" || !e.visible) {
        continue;
      }
      seen.add(e.id);
      let nodes = markNodes.get(e.id);
      if (!nodes) {
        nodes = buildMarkNodes(e, catalog);
        markRoot.add(nodes.group);
        markNodes.set(e.id, nodes);
        ensureLabel(e.id, e.name);
      }
      const asset = catalog.assetOf(e.className);
      const nextSig = assetBodySig(asset, e.className);
      if (
        nodes.className !== e.className ||
        nodes.assetSig !== nextSig ||
        (!nodes.hasGlb && catalog.lookup(e.className)?.template)
      ) {
        // Class / library pose-size-color-model changed, or GLB arrived later.
        rebuildBody(nodes, e, catalog);
      }
      nodes.group.position.set(e.position[0], e.position[1], 0);
      // When the asset disables velocity heading, keep yaw at 0 (pose holds
      // the class orientation). Unknown classes keep the derived heading.
      const applyHeading =
        (!asset || asset.rotationFromVelocity) && Number.isFinite(e.heading);
      nodes.group.rotation.z = applyHeading ? e.heading : 0;
      // Trail stays on the UUID track color (same as the hover triangle).
      const track = new Color(nodes.trackColor);
      (nodes.trail.material as LineBasicMaterial).color.copy(track);
      if (!nodes.hasGlb) {
        const bodyColor = new Color(asset?.markColor || e.color);
        nodes.bodyRoot.traverse((o) => {
          const mesh = o as Mesh;
          const mat = mesh.material as MeshStandardMaterial | undefined;
          if (mat && "color" in mat && mat.color) {
            mat.color.copy(bodyColor);
          }
        });
      }
      layoutChrome(nodes);
      const label = labelEls.get(e.id);
      if (label && label.textContent !== e.name) {
        label.textContent = e.name;
      }
      updateTrail(e.id, e.position[0], e.position[1], showTrails);
    }
    for (const [id, nodes] of markNodes) {
      if (!seen.has(id)) {
        markRoot.remove(nodes.group);
        disposeMarkNodes(nodes);
        markNodes.delete(id);
        trailPts.delete(id);
        const label = labelEls.get(id);
        label?.remove();
        labelEls.delete(id);
      }
    }
  };
  // Marks update at event rate; reconcile on any store change (cheap diff).
  const unsubscribe = useViewportStore.subscribe(() => {
    reconcileMarks();
  });
  const unsubCatalog = catalog.onChange(() => {
    // Library (re)loaded — rebuild every mark so pose/color/GLB edits apply.
    const colorUpdates: MarkEntity[] = [];
    for (const [id, nodes] of markNodes) {
      const e = getViewportState().entities[id];
      if (!e || e.type !== "mark") {
        continue;
      }
      rebuildBody(nodes, e, catalog);
      const asset = catalog.assetOf(e.className);
      if (asset?.markColor && asset.markColor !== e.color) {
        colorUpdates.push({ ...e, color: asset.markColor });
      }
    }
    if (colorUpdates.length) {
      getViewportState().upsertEntities(colorUpdates, { fromMarks: true });
    }
    reconcileMarks();
  });
  const authToken = options.authToken || "";
  if (authToken) {
    void catalog.load(authToken);
  }
  const onLibraryChanged = () => {
    if (authToken) {
      void catalog.load(authToken, { force: true });
    }
  };
  window.addEventListener(LIBRARY_CHANGED_EVENT, onLibraryChanged);
  reconcileMarks();

  const projV = new Vector3();
  const updateLabels = () => {
    const show = getViewportState().showLabels;
    if (!show) {
      return;
    }
    const cam = world.getActiveCamera();
    const w = host.clientWidth;
    const h = host.clientHeight;
    for (const [id, el] of labelEls) {
      const nodes = markNodes.get(id);
      if (!nodes) {
        el.style.display = "none";
        continue;
      }
      projV
        .set(
          nodes.group.position.x,
          nodes.group.position.y,
          nodes.labelZ,
        )
        .project(cam);
      if (projV.z > 1 || projV.z < -1) {
        el.style.display = "none";
        continue;
      }
      el.style.display = "";
      el.style.transform = `translate(-50%, -100%) translate(${(projV.x * 0.5 + 0.5) * w}px, ${(-projV.y * 0.5 + 0.5) * h}px)`;
    }
  };
  onFrame.add(updateLabels);

  // Hide mark labels when the label overlay is off.
  const unsubscribeLabels = useViewportStore.subscribe((s, prev) => {
    if (s.showLabels !== prev.showLabels) {
      for (const el of labelEls.values()) {
        el.style.display = s.showLabels ? "" : "none";
      }
    }
  });

  return () => {
    unsubscribe();
    unsubscribeLabels();
    unsubCatalog();
    window.removeEventListener(LIBRARY_CHANGED_EVENT, onLibraryChanged);
    catalog.dispose();
    onFrame.delete(updateLabels);
    for (const [, nodes] of markNodes) {
      markRoot.remove(nodes.group);
      disposeMarkNodes(nodes);
    }
    markNodes.clear();
    for (const el of labelEls.values()) {
      el.remove();
    }
    labelEls.clear();
  };
}
