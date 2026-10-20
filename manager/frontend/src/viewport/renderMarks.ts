// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
  Box3,
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  Color,
  DoubleSide,
  Group,
  Mesh,
  MeshBasicMaterial,
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
  regionDwellRows,
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
/**
 * Square tube half-extents (meters). A flat XY ribbon disappears when the
 * camera looks along the floor; a small 3D tube stays visible from top and
 * side views.
 */
const TRAIL_HALF_WIDTH_M = 0.02;
const TRAIL_HALF_HEIGHT_M = 0.02;
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
  /**
   * Yaw group for body + indicator. Trail stays outside so heading does not
   * warp the path ribbon.
   */
  posed: Group;
  /** Pose / model holder; replaced when the library asset resolves. */
  bodyRoot: Group;
  /** Downward triangle sprite above the body (UUID track color). */
  indicator: Sprite;
  /** Thick floor ribbon (mesh) — WebGL Line linewidth is not portable. */
  trail: Mesh;
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
 * draw.js (main). Local scale stays 1; `layoutChrome` sizes it to the body
 * footprint because legacy applied `thisMark.scale` to the whole mark.
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

/** Build a tracked-object label with an optional per-region dwell list. */
export function createMarkLabelElement(title: string): HTMLDivElement {
  const el = document.createElement("div");
  el.className = "ss-vlabel ss-vlabel--mark";

  const header = document.createElement("div");
  header.className = "ss-vlabel--mark-id";
  header.textContent = title;

  const dwell = document.createElement("div");
  dwell.className = "ss-vlabel--mark-dwell";
  dwell.hidden = true;
  const heading = document.createElement("div");
  heading.className = "ss-vlabel--mark-row";
  heading.textContent = "Dwell Time:";
  const list = document.createElement("ul");
  list.className = "ss-vlabel--mark-dwell-list";
  dwell.append(heading, list);

  el.append(header, dwell);
  return el;
}

export function updateMarkDwellLabel(label: HTMLDivElement, rows: ReturnType<typeof regionDwellRows>): void {
  const dwell = label.querySelector<HTMLDivElement>(".ss-vlabel--mark-dwell")!;
  const signature = JSON.stringify(rows.map(row => [row.regionId, row.name, row.seconds.toFixed(1)]));
  if (dwell.dataset.signature === signature) {
    return;
  }
  dwell.dataset.signature = signature;
  dwell.hidden = rows.length === 0;
  const list = dwell.querySelector<HTMLUListElement>(".ss-vlabel--mark-dwell-list")!;
  list.replaceChildren(...rows.map(row => {
    const item = document.createElement("li");
    item.className = "ss-vlabel--mark-row";
    const name = document.createElement("span");
    name.className = "ss-vlabel--mark-region-name";
    name.textContent = row.name;
    const seconds = document.createElement("span");
    seconds.textContent = `${row.seconds.toFixed(1)} s`;
    item.append(name, seconds);
    return item;
  }));
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


/**
 * Place indicator + label from body bounds (legacy gaps from assetmanager.js).
 * Triangle world size follows the body’s XY footprint so it matches marks that
 * were uniformly scaled on main (`sprite.scale = 1` under `thisMark.scale`).
 */
function layoutChrome(nodes: MarkNodes): void {
  nodes.bodyRoot.updateMatrixWorld(true);
  const box = new Box3().setFromObject(nodes.bodyRoot);
  const size = new Vector3();
  if (!box.isEmpty()) {
    box.getSize(size);
  }
  const top = box.isEmpty() ? LABEL_Z_FALLBACK - LABEL_GAP : box.max.z;
  const footprint = Math.max(size.x, size.y, 0.35);
  const indicatorScale = Math.min(Math.max(footprint, 0.35), 1);
  nodes.indicator.scale.set(indicatorScale, indicatorScale, indicatorScale);
  nodes.indicator.position.set(0, 0, top + INDICATOR_GAP);
  nodes.labelZ = Math.max(top + LABEL_GAP, LABEL_Z_FALLBACK);
}

function trackColorFor(entity: MarkEntity): string {
  return colorFromTrackId(trackIdFromMarkKey(entity.id));
}

export function createTrailRibbon(trackColor: string): Mesh {
  const geometry = new BufferGeometry();
  // Placeholder attributes so the mesh is valid before the first sample.
  geometry.setAttribute("position", new BufferAttribute(new Float32Array(9), 3));
  geometry.setIndex([0, 1, 2]);
  const material = new MeshBasicMaterial({
    color: trackColor,
    transparent: true,
    opacity: 0.85,
    depthWrite: false,
    depthTest: true,
    side: DoubleSide,
  });
  const mesh = new Mesh(geometry, material);
  mesh.frustumCulled = false;
  mesh.renderOrder = -1;
  mesh.visible = false;
  mesh.name = "trail";
  return mesh;
}

/**
 * Rebuild a square-tube trail from world-space samples (newest first).
 * Written in local coords relative to the mark origin. Four verts per sample
 * (left/right × bottom/top) so the path is visible from top-down and side
 * perspectives — a flat XY ribbon is invisible when edge-on.
 */
function writeTrailRibbon(
  mesh: Mesh,
  pts: Vector3[],
  originX: number,
  originY: number,
): void {
  const n = pts.length;
  if (n < 2) {
    mesh.visible = false;
    return;
  }
  // 4 corners per sample: 0 LB, 1 RB, 2 RT, 3 LT
  const verts = new Float32Array(n * 4 * 3);
  // 4 sides × 2 tris × 3 indices per segment
  const indices = new Uint16Array((n - 1) * 24);
  for (let i = 0; i < n; i++) {
    const p = pts[i];
    const lx = p.x - originX;
    const ly = p.y - originY;
    const lz = p.z;
    let dx: number;
    let dy: number;
    if (i < n - 1) {
      dx = pts[i + 1].x - p.x;
      dy = pts[i + 1].y - p.y;
    } else {
      dx = p.x - pts[i - 1].x;
      dy = p.y - pts[i - 1].y;
    }
    const len = Math.hypot(dx, dy) || 1;
    const nx = (-dy / len) * TRAIL_HALF_WIDTH_M;
    const ny = (dx / len) * TRAIL_HALF_WIDTH_M;
    const hz = TRAIL_HALF_HEIGHT_M;
    const base = i * 4;
    // left-bottom
    verts[base * 3] = lx + nx;
    verts[base * 3 + 1] = ly + ny;
    verts[base * 3 + 2] = lz - hz;
    // right-bottom
    verts[(base + 1) * 3] = lx - nx;
    verts[(base + 1) * 3 + 1] = ly - ny;
    verts[(base + 1) * 3 + 2] = lz - hz;
    // right-top
    verts[(base + 2) * 3] = lx - nx;
    verts[(base + 2) * 3 + 1] = ly - ny;
    verts[(base + 2) * 3 + 2] = lz + hz;
    // left-top
    verts[(base + 3) * 3] = lx + nx;
    verts[(base + 3) * 3 + 1] = ly + ny;
    verts[(base + 3) * 3 + 2] = lz + hz;
    if (i > 0) {
      const prev = (i - 1) * 4;
      const cur = base;
      let o = (i - 1) * 24;
      const quad = (a: number, b: number, c: number, d: number) => {
        indices[o++] = a;
        indices[o++] = b;
        indices[o++] = c;
        indices[o++] = a;
        indices[o++] = c;
        indices[o++] = d;
      };
      // bottom, right, top, left
      quad(prev, prev + 1, cur + 1, cur);
      quad(prev + 1, prev + 2, cur + 2, cur + 1);
      quad(prev + 2, prev + 3, cur + 3, cur + 2);
      quad(prev + 3, prev, cur, cur + 3);
    }
  }
  const geometry = mesh.geometry;
  geometry.setAttribute("position", new BufferAttribute(verts, 3));
  geometry.setIndex(new BufferAttribute(indices, 1));
  geometry.attributes.position.needsUpdate = true;
  geometry.computeBoundingSphere();
  geometry.computeBoundingBox();
  mesh.visible = true;
}

function buildMarkNodes(
  entity: MarkEntity,
  catalog: MarkAssetCatalog,
): MarkNodes {
  const group = new Group();
  const posed = new Group();
  posed.name = "posed";
  const asset = catalog.assetOf(entity.className);
  const { bodyRoot, hasGlb } = buildMarkBody(entity, catalog);
  const trackColor = trackColorFor(entity);

  const indicator = createTrackIndicator(trackColor);
  const trail = createTrailRibbon(trackColor);
  posed.add(bodyRoot, indicator);
  group.add(posed, trail);
  const nodes: MarkNodes = {
    group,
    posed,
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
  nodes.posed.remove(nodes.bodyRoot);
  disposeObjectTree(nodes.bodyRoot);
  const asset = catalog.assetOf(entity.className);
  const built = buildMarkBody(entity, catalog);
  nodes.bodyRoot = built.bodyRoot;
  nodes.hasGlb = built.hasGlb;
  nodes.className = entity.className;
  nodes.assetSig = assetBodySig(asset, entity.className);
  nodes.posed.add(nodes.bodyRoot, nodes.indicator);
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

  const ensureLabel = (entity: MarkEntity) => {
    const { id, name } = entity;
    let el = labelEls.get(id);
    if (!el) {
      el = createMarkLabelElement(name);
      labelLayer.appendChild(el);
      labelEls.set(id, el);
    }
    const header = el.querySelector(".ss-vlabel--mark-id");
    if (header && header.textContent !== name) {
      header.textContent = name;
    }
    updateMarkDwellLabel(el, regionDwellRows(entity, getViewportState().entities));
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
    // Trail samples are stored in world XY; the ribbon is parented under the
    // mark group (outside yaw), so write local coords (world − mark origin).
    const last = pts[0];
    if (!last || Math.hypot(last.x - x, last.y - y) > 0.05) {
      // Center the tube above the floor so the bottom face does not z-fight.
      pts.unshift(new Vector3(x, y, TRAIL_HALF_HEIGHT_M + 0.02));
      if (pts.length > TRAIL_POINTS) {
        pts.pop();
      }
    }
    if (show && pts.length >= 2) {
      writeTrailRibbon(nodes.trail, pts, x, y);
    } else {
      nodes.trail.visible = false;
    }
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
        ensureLabel(e);
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
      // Apply yaw only to posed (body+indicator) so the trail ribbon stays put.
      const applyHeading =
        (!asset || asset.rotationFromVelocity) && Number.isFinite(e.heading);
      nodes.posed.rotation.z = applyHeading ? e.heading : 0;
      // Trail stays on the UUID track color (same as the hover triangle).
      const track = new Color(nodes.trackColor);
      (nodes.trail.material as MeshBasicMaterial).color.copy(track);
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
      ensureLabel(e);
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
    const show = getViewportState().showLabels && world.markRoot.visible;
    if (!show) {
      for (const element of labelEls.values()) {
        element.style.display = "none";
      }
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
