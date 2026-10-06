// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from "react";
import {
  AmbientLight,
  BoxGeometry,
  BufferAttribute,
  BufferGeometry,
  CapsuleGeometry,
  Color,
  DirectionalLight,
  GridHelper,
  Group,
  Line,
  LineBasicMaterial,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  OrthographicCamera,
  PerspectiveCamera,
  RingGeometry,
  Scene,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { applyBlenderBindings, applyViewPreset } from "./controls";
import { useViewportMarks } from "./marks";
import { getViewportState, useViewportStore } from "./store";
import type { MarkEntity, ViewPreset } from "./types";
import "./viewport.css";

/**
 * Single-pane 3D viewport (Phase 1.1 foundation).
 *
 * Z-up world matching the Scenescape domain. Renders live tracked-object
 * marks from the frozen `ss-scene-objects` event; tools and panels arrive in
 * Phase 1.2 / 1.3. The three.js layer reconciles imperatively from the
 * zustand scene-graph store — no React re-render per frame.
 */

interface SceneViewportProps {
  sceneId: string;
  assetMarkColors?: Record<string, string>;
}

type Theme = "light" | "dark";

function readTheme(): Theme {
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

function useTheme(): Theme {
  const [theme, setTheme] = useState<Theme>(readTheme);
  useEffect(() => {
    const mo = new MutationObserver(() => setTheme(readTheme()));
    mo.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    return () => mo.disconnect();
  }, []);
  return theme;
}

const THEME_COLORS: Record<Theme, { bg: number; gridA: number; gridB: number }> = {
  dark: { bg: 0x1b1f23, gridA: 0x4b5563, gridB: 0x2d333b },
  light: { bg: 0xe9ebee, gridA: 0x9aa0a8, gridB: 0xc7ccd2 },
};

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

const PRESET_LABELS: Record<ViewPreset, string> = {
  top: "Top",
  front: "Front",
  side: "Side",
  persp: "Persp",
};

function OverlayCheck({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: () => void;
  label: string;
}) {
  return (
    <label className={`vchk${checked ? " on" : ""}`}>
      <input type="checkbox" checked={checked} onChange={onChange} />
      <span className="box" aria-hidden="true" />
      {label}
    </label>
  );
}

export function SceneViewport({ sceneId, assetMarkColors }: SceneViewportProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const labelLayerRef = useRef<HTMLDivElement>(null);
  const theme = useTheme();
  useViewportMarks({ assetMarkColors });

  const viewPreset = useViewportStore((s) => s.viewPreset);
  const ortho = useViewportStore((s) => s.ortho);
  const showGrid = useViewportStore((s) => s.showGrid);
  const showLabels = useViewportStore((s) => s.showLabels);
  const showTrails = useViewportStore((s) => s.showTrails);
  const setViewPreset = useViewportStore((s) => s.setViewPreset);
  const setOrtho = useViewportStore((s) => s.setOrtho);
  const toggleOverlay = useViewportStore((s) => s.toggleOverlay);

  const themeRef = useRef(theme);
  themeRef.current = theme;

  // --- three.js scene ----------------------------------------------------
  useEffect(() => {
    const host = hostRef.current;
    const labelLayer = labelLayerRef.current;
    if (!host || !labelLayer) {
      return;
    }
    let disposed = false;

    const scene = new Scene();
    const applyTheme = () => {
      const c = THEME_COLORS[themeRef.current];
      scene.background = new Color(c.bg);
    };
    applyTheme();

    const makeZUpCamera = <T extends PerspectiveCamera | OrthographicCamera>(
      cam: T,
    ): T => {
      cam.up.set(0, 0, 1);
      return cam;
    };
    const perspCam = makeZUpCamera(
      new PerspectiveCamera(
        50,
        Math.max(host.clientWidth, 1) / Math.max(host.clientHeight, 1),
        0.1,
        5000,
      ),
    );
    const orthoCam = makeZUpCamera(
      new OrthographicCamera(-20, 20, 20, -20, 0.1, 5000),
    );

    let renderer: WebGLRenderer | null = null;
    try {
      renderer = new WebGLRenderer({ antialias: true });
    } catch {
      host.replaceChildren();
      const note = document.createElement("p");
      note.className = "ss-viewport-error";
      note.textContent = "WebGL is not available in this browser.";
      host.appendChild(note);
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(host.clientWidth, host.clientHeight, false);
    host.replaceChildren(renderer.domElement);

    scene.add(new AmbientLight(0xffffff, 0.7));
    const key = new DirectionalLight(0xffffff, 0.8);
    key.position.set(4, -6, 10);
    scene.add(key);

    // GridHelper builds in the XZ plane (Y-up ground); rotate to the XY
    // plane for the Z-up floor, same as the placement canvas.
    let grid: GridHelper | null = null;
    const rebuildGrid = () => {
      const c = THEME_COLORS[themeRef.current];
      if (grid) {
        scene.remove(grid);
        grid.geometry.dispose();
        (grid.material as LineBasicMaterial).dispose();
      }
      grid = new GridHelper(60, 60, c.gridA, c.gridB);
      grid.rotation.x = Math.PI / 2;
      grid.visible = getViewportState().showGrid;
      scene.add(grid);
    };
    rebuildGrid();

    const markRoot = new Group();
    markRoot.name = "marks";
    scene.add(markRoot);

    const orbit = new OrbitControls(perspCam, renderer.domElement);
    applyBlenderBindings(orbit);

    // TransformControls rides along for Phase 1.2 tools; unattended here.
    const gizmo = new TransformControls(perspCam, renderer.domElement);
    gizmo.setMode("translate");
    scene.add(gizmo.getHelper());

    const target = new Vector3(0, 0, 0);
    const setActiveCamera = (useOrtho: boolean) => {
      const cam = useOrtho ? orthoCam : perspCam;
      (orbit as unknown as { object: unknown }).object = cam;
      try {
        (gizmo as unknown as { camera: unknown }).camera = cam;
      } catch {
        /* older TransformControls keep the constructor camera */
      }
      const s = getViewportState();
      applyViewPreset(cam, orbit, s.viewPreset, target, 28);
    };
    setActiveCamera(getViewportState().ortho);

    const updateOrthoFrustum = () => {
      const w = Math.max(host.clientWidth, 1);
      const h = Math.max(host.clientHeight, 1);
      const halfH = 24;
      orthoCam.left = (-halfH * w) / h;
      orthoCam.right = (halfH * w) / h;
      orthoCam.top = halfH;
      orthoCam.bottom = -halfH;
      orthoCam.updateProjectionMatrix();
      perspCam.aspect = w / h;
      perspCam.updateProjectionMatrix();
      renderer?.setSize(w, h, false);
    };
    updateOrthoFrustum();
    const ro = new ResizeObserver(updateOrthoFrustum);
    ro.observe(host);

    // --- mark reconciliation (imperative, off the store) ------------------
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
    const unsubscribe = useViewportStore.subscribe(reconcileMarks);
    reconcileMarks();

    // --- per-frame: controls, labels, render ------------------------------
    const projV = new Vector3();
    const updateLabels = () => {
      const show = getViewportState().showLabels;
      labelLayer.style.display = show ? "" : "none";
      if (!show) {
        return;
      }
      const s = getViewportState();
      const cam = s.ortho ? orthoCam : perspCam;
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

    let frame = 0;
    const tick = () => {
      if (disposed || !renderer) {
        return;
      }
      orbit.update();
      updateLabels();
      const s = getViewportState();
      renderer.render(scene, s.ortho ? orthoCam : perspCam);
      frame = window.requestAnimationFrame(tick);
    };
    tick();

    // --- UI state -> camera ------------------------------------------------
    const syncCamera = () => {
      if (disposed) {
        return;
      }
      setActiveCamera(getViewportState().ortho);
    };
    const unsubscribeUi = useViewportStore.subscribe((s, prev) => {
      if (s.viewPreset !== prev.viewPreset || s.ortho !== prev.ortho) {
        syncCamera();
      }
      if (s.showGrid !== prev.showGrid && grid) {
        grid.visible = s.showGrid;
      }
    });

    const themeMo = new MutationObserver(() => {
      applyTheme();
      rebuildGrid();
    });
    themeMo.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    return () => {
      disposed = true;
      window.cancelAnimationFrame(frame);
      ro.disconnect();
      themeMo.disconnect();
      unsubscribe();
      unsubscribeUi();
      gizmo.detach();
      gizmo.dispose();
      orbit.dispose();
      for (const [, nodes] of markNodes) {
        disposeMarkNodes(nodes);
      }
      markNodes.clear();
      labelLayer.replaceChildren();
      renderer?.dispose();
      renderer?.domElement.remove();
      renderer = null;
    };
  }, [sceneId]);

  // --- keyboard: 1/3/7 views, 5 persp/ortho --------------------------------
  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      const t = ev.target as HTMLElement | null;
      if (
        t &&
        (t.tagName === "INPUT" ||
          t.tagName === "TEXTAREA" ||
          t.isContentEditable)
      ) {
        return;
      }
      const s = getViewportState();
      if (ev.key === "1") {
        s.setViewPreset("front");
      } else if (ev.key === "3") {
        s.setViewPreset("side");
      } else if (ev.key === "7") {
        s.setViewPreset("top");
      } else if (ev.key === "5") {
        s.setOrtho(!s.ortho);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="ss-viewport" data-scene-id={sceneId}>
      <div className="ss-viewport-bar" role="toolbar" aria-label="Viewport">
        <span className="ss-viewport-bar-label">View</span>
        <div className="seg" role="group" aria-label="View preset">
          {(Object.keys(PRESET_LABELS) as ViewPreset[]).map((p) => (
            <button
              key={p}
              type="button"
              className={viewPreset === p ? "on" : ""}
              onClick={() => setViewPreset(p)}
              title={`${PRESET_LABELS[p]} view`}
            >
              {PRESET_LABELS[p]}
            </button>
          ))}
        </div>
        <div className="seg" role="group" aria-label="Projection">
          <button
            type="button"
            className={!ortho ? "on" : ""}
            onClick={() => setOrtho(false)}
            title="Perspective (5)"
          >
            Persp
          </button>
          <button
            type="button"
            className={ortho ? "on" : ""}
            onClick={() => setOrtho(true)}
            title="Orthographic (5)"
          >
            Ortho
          </button>
        </div>
        <span className="ss-viewport-bar-sep" aria-hidden="true" />
        <span className="ss-viewport-bar-label">Overlays</span>
        <OverlayCheck
          checked={showGrid}
          onChange={() => toggleOverlay("showGrid")}
          label="Grid"
        />
        <OverlayCheck
          checked={showTrails}
          onChange={() => toggleOverlay("showTrails")}
          label="Trails"
        />
        <OverlayCheck
          checked={showLabels}
          onChange={() => toggleOverlay("showLabels")}
          label="Labels"
        />
        <span className="ss-viewport-beta">3D beta</span>
      </div>
      <div className="ss-viewport-host" ref={hostRef} />
      <div className="ss-viewport-labels" ref={labelLayerRef} aria-hidden="true" />
    </div>
  );
}
