// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
  BoxGeometry,
  BufferAttribute,
  BufferGeometry,
  CapsuleGeometry,
  Color,
  Group,
  Line,
  LineBasicMaterial,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  RingGeometry,
  Vector3,
} from "three";
import type { ViewportWorld } from "./world";
import { getViewportState, useViewportStore } from "./store";
import type { MarkEntity } from "./types";

/**
 * Imperative three.js rendering for live tracked-object marks (Phase 1.1).
 * Consumes the store populated by `useViewportMarks`; reconciles meshes on
 * store updates and projects name labels every frame. No React re-render.
 */

const TRAIL_POINTS = 40;
const LABEL_Z = 1.6;

interface MarkNodes {
  group: Group;
  body: Mesh;
  ring: Mesh;
  trail: Line;
}

function buildMarkNodes(entity: MarkEntity): MarkNodes {
  const color = new Color(entity.color);
  const group = new Group();
  let body: Mesh;
  if (entity.className.toLowerCase().includes("person")) {
    const geo = new CapsuleGeometry(0.28, 0.9, 6, 12);
    geo.rotateX(Math.PI / 2); // capsule axis Y -> Z (Z-up standing figure)
    body = new Mesh(geo, new MeshStandardMaterial({ color, roughness: 0.5 }));
    body.position.z = 0.85;
  } else {
    body = new Mesh(
      new BoxGeometry(0.9, 0.9, 1.1),
      new MeshStandardMaterial({ color, roughness: 0.55 }),
    );
    body.position.z = 0.55;
  }
  // RingGeometry already lies in the XY plane = the Z-up floor plane.
  const ring = new Mesh(
    new RingGeometry(0.5, 0.66, 32),
    new MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.8,
      side: 2, // DoubleSide
    }),
  );
  ring.position.z = 0.05;
  const trailGeo = new BufferGeometry();
  trailGeo.setAttribute(
    "position",
    new BufferAttribute(new Float32Array(TRAIL_POINTS * 3), 3),
  );
  const trail = new Line(
    trailGeo,
    new LineBasicMaterial({ color, transparent: true, opacity: 0.5 }),
  );
  trail.frustumCulled = false;
  group.add(body, ring, trail);
  return { group, body, ring, trail };
}

function disposeMarkNodes(nodes: MarkNodes): void {
  nodes.group.traverse((o) => {
    const mesh = o as Mesh;
    if (mesh.geometry) {
      mesh.geometry.dispose();
    }
    const mat = mesh.material as { dispose?: () => void } | undefined;
    mat?.dispose?.();
  });
}

/** Attach mark rendering to a live world. Returns a dispose function. */
export function attachMarkRenderer(world: ViewportWorld): () => void {
  const { markRoot, labelLayer, host, onFrame } = world;
  const markNodes = new Map<string, MarkNodes>();
  const labelEls = new Map<string, HTMLDivElement>();
  const trailPts = new Map<string, Vector3[]>();

  const ensureLabel = (id: string, name: string) => {
    let el = labelEls.get(id);
    if (!el) {
      el = document.createElement("div");
      el.className = "ss-vlabel";
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
    for (let i = 0; i < TRAIL_POINTS; i++) {
      const p = pts[Math.min(i, pts.length - 1)] ?? new Vector3(x, y, 0.08);
      attr.setXYZ(i, p.x, p.y, p.z);
    }
    attr.needsUpdate = true;
    nodes.trail.visible = show;
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
        nodes = buildMarkNodes(e);
        markRoot.add(nodes.group);
        markNodes.set(e.id, nodes);
        ensureLabel(e.id, e.name);
      }
      nodes.group.position.set(e.position[0], e.position[1], 0);
      nodes.group.rotation.z = Number.isFinite(e.heading) ? e.heading : 0;
      const c = new Color(e.color);
      (nodes.body.material as MeshStandardMaterial).color.copy(c);
      (nodes.ring.material as MeshBasicMaterial).color.copy(c);
      (nodes.trail.material as LineBasicMaterial).color.copy(c);
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
        .set(nodes.group.position.x, nodes.group.position.y, LABEL_Z)
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
