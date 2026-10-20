// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
  AmbientLight,
  AxesHelper,
  Box3,
  Color,
  DirectionalLight,
  GridHelper,
  Group,
  LineBasicMaterial,
  MOUSE,
  Object3D,
  OrthographicCamera,
  PerspectiveCamera,
  Scene,
  TOUCH,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { applyBlenderBindings, applyViewPreset } from "./controls";
import type { EntityNodes } from "./entities";
import type { CameraProjectionHandle } from "./projection/projectOntoMap";
import type { ProjectionPool } from "./projection/projectionPool";
import { getViewportState } from "./store";
import { useLightingStore } from "./lighting";

export type ViewportTheme = "light" | "dark";

const THEME_COLORS: Record<
  ViewportTheme,
  { bg: number; gridA: number; gridB: number }
> = {
  dark: { bg: 0x1b1f23, gridA: 0x4b5563, gridB: 0x2d333b },
  light: { bg: 0xe9ebee, gridA: 0x9aa0a8, gridB: 0xc7ccd2 },
};

export interface ViewportWorld {
  host: HTMLDivElement;
  labelLayer: HTMLDivElement;
  scene: Scene;
  perspCam: PerspectiveCamera;
  orthoCam: OrthographicCamera;
  orbit: OrbitControls;
  gizmo: TransformControls;
  markRoot: Group;
  entityRoot: Group;
  /** Scene floor map (PNG plane or posed GLB). */
  mapRoot: Group;
  /** Loaded map object (for projection / raycasts); null until loaded. */
  sceneMesh: Object3D | null;
  /**
   * Per-camera map projectors (live "project frame" / "project all frames").
   * Null until the scene map mesh is loaded.
   */
  projectionPool: ProjectionPool | null;
  /**
   * Alias for the calibrate session's projector (pool handle for that camera).
   * Prefer projectionPool.acquire(id) for live multi-camera coverage.
   */
  cameraProjection: CameraProjectionHandle | null;
  /** Entity id → three nodes, owned by the entity reconciler. */
  entityNodes: Map<string, EntityNodes>;
  /** Per-frame callbacks registered by renderers (label projection…). */
  onFrame: Set<() => void>;
  getActiveCamera: () => PerspectiveCamera | OrthographicCamera;
  setActiveCamera: (useOrtho: boolean) => void;
  setTheme: (theme: ViewportTheme) => void;
  setGridVisible: (visible: boolean) => void;
  dispose: () => void;
}

function makeZUpCamera<T extends PerspectiveCamera | OrthographicCamera>(
  cam: T,
): T {
  cam.up.set(0, 0, 1);
  return cam;
}

/**
 * Owns the three.js world: renderer, Z-up cameras, Blender-bound orbit
 * controls, the shared TransformControls gizmo, lights, grid, and the
 * render loop. Entity/mark renderers attach roots and per-frame callbacks;
 * React never re-renders per frame.
 *
 * Returns null when WebGL is unavailable (caller shows a fallback note).
 */
export function createViewportWorld(
  host: HTMLDivElement,
  labelLayer: HTMLDivElement,
): ViewportWorld | null {
  let renderer: WebGLRenderer | null = null;
  try {
    renderer = new WebGLRenderer({ antialias: true });
  } catch {
    return null;
  }

  const scene = new Scene();
  let theme: ViewportTheme =
    document.documentElement.dataset.theme === "light" ? "light" : "dark";
  const applyThemeColors = () => {
    scene.background = new Color(THEME_COLORS[theme].bg);
  };
  applyThemeColors();

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
  let activeCamera: PerspectiveCamera | OrthographicCamera = perspCam;

  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(host.clientWidth, host.clientHeight, false);
  host.replaceChildren(renderer.domElement);

  const ambient = new AmbientLight(0xa0a0a0, useLightingStore.getState().intensity);
  ambient.name = "ambient-light";
  scene.add(ambient);
  const applyLighting = () => {
    const intensity = useLightingStore.getState().intensity;
    ambient.intensity = intensity;
    if (renderer) renderer.toneMappingExposure = intensity;
  };
  applyLighting();
  const unsubscribeLighting = useLightingStore.subscribe(applyLighting);
  const key = new DirectionalLight(0xffffff, 0.8);
  key.position.set(4, -6, 10);
  scene.add(key);

  // GridHelper builds in the XZ plane (Y-up ground); rotate to the XY
  // plane for the Z-up floor, same as the placement canvas.
  let grid: GridHelper | null = null;
  const rebuildGrid = () => {
    const c = THEME_COLORS[theme];
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

  // Axis helpers: RGB XYZ axes at the world origin (Z-up).
  const axes = new AxesHelper(4);
  scene.add(axes);

  const mapRoot = new Group();
  mapRoot.name = "scene-map";
  scene.add(mapRoot);

  const markRoot = new Group();
  markRoot.name = "marks";
  scene.add(markRoot);

  const entityRoot = new Group();
  entityRoot.name = "entities";
  scene.add(entityRoot);

  const orbit = new OrbitControls(perspCam, renderer.domElement);
  applyBlenderBindings(orbit);

  const gizmo = new TransformControls(perspCam, renderer.domElement);
  gizmo.setMode("translate");
  scene.add(gizmo.getHelper());
  // While a gizmo handle drags, the orbit control must not fight it.
  gizmo.addEventListener("dragging-changed", (ev) => {
    orbit.enabled = !(ev as { value?: boolean }).value;
  });

  /** Half of the orthographic vertical frustum in world meters. */
  let orthoHalfExtent = 24;

  /** Perspective pose captured when entering Ortho; restored on return to Persp. */
  let savedPerspPose: {
    position: Vector3;
    up: Vector3;
    target: Vector3;
  } | null = null;

  const snapshotPerspPose = () => {
    savedPerspPose = {
      position: perspCam.position.clone(),
      up: perspCam.up.clone(),
      target: orbit.target.clone(),
    };
  };

  const restorePerspPose = (): boolean => {
    if (!savedPerspPose) {
      return false;
    }
    perspCam.position.copy(savedPerspPose.position);
    perspCam.up.copy(savedPerspPose.up);
    orbit.target.copy(savedPerspPose.target);
    perspCam.lookAt(orbit.target);
    orbit.update();
    return true;
  };

  const updateFrustums = () => {
    const w = Math.max(host.clientWidth, 1);
    const h = Math.max(host.clientHeight, 1);
    const halfH = orthoHalfExtent;
    orthoCam.left = (-halfH * w) / h;
    orthoCam.right = (halfH * w) / h;
    orthoCam.top = halfH;
    orthoCam.bottom = -halfH;
    orthoCam.updateProjectionMatrix();
    perspCam.aspect = w / h;
    perspCam.updateProjectionMatrix();
    renderer?.setSize(w, h, false);
  };

  const applyOrthoNavigation = () => {
    // Map-style: left-drag pans; no orbit (keeps the view top-down).
    orbit.enableRotate = false;
    orbit.mouseButtons = {
      LEFT: MOUSE.PAN,
      MIDDLE: MOUSE.DOLLY,
      RIGHT: MOUSE.PAN,
    };
    orbit.touches = {
      ONE: TOUCH.PAN,
      TWO: TOUCH.DOLLY_PAN,
    };
  };

  const applyPerspNavigation = () => {
    orbit.enableRotate = true;
    applyBlenderBindings(orbit);
  };

  /**
   * Size the ortho frustum so the scene map (or a default span) fills the
   * view — the point of Ortho is a usable top-down plan for ROI/tripwire.
   */
  const frameOrthoTopDown = (target: Vector3): number => {
    const box = new Box3().setFromObject(mapRoot);
    let span = 48;
    if (!box.isEmpty()) {
      const size = new Vector3();
      const center = new Vector3();
      box.getSize(size);
      box.getCenter(center);
      target.copy(center);
      span = Math.max(size.x, size.y, 8);
    } else if (target.lengthSq() === 0) {
      target.set(0, 0, 0);
    }
    orthoHalfExtent = span * 0.55;
    orthoCam.zoom = 1;
    orthoCam.near = 0.1;
    orthoCam.far = Math.max(5000, span * 20);
    updateFrustums();
    return Math.max(span, 8);
  };

  const setActiveCamera = (useOrtho: boolean) => {
    const wasOrtho = activeCamera === orthoCam;
    // Capture the live perspective pose before we swap cameras / reframe.
    if (useOrtho && !wasOrtho) {
      snapshotPerspPose();
    }
    const cam = useOrtho ? orthoCam : perspCam;
    activeCamera = cam;
    (orbit as unknown as { object: unknown }).object = cam;
    try {
      (gizmo as unknown as { camera: unknown }).camera = cam;
    } catch {
      /* older TransformControls keep the constructor camera */
    }
    if (useOrtho) {
      const target = orbit.target.clone();
      applyOrthoNavigation();
      const distance = frameOrthoTopDown(target);
      applyViewPreset(cam, orbit, "top", target, distance);
    } else {
      applyPerspNavigation();
      if (!restorePerspPose()) {
        const target = orbit.target.clone();
        const preset = getViewportState().viewPreset;
        applyViewPreset(
          cam,
          orbit,
          preset === "top" ? "persp" : preset,
          target,
          28,
        );
      }
    }
  };
  setActiveCamera(getViewportState().ortho);
  const ro = new ResizeObserver(updateFrustums);
  ro.observe(host);

  const onFrame = new Set<() => void>();
  let disposed = false;
  let frame = 0;
  const tick = () => {
    if (disposed || !renderer) {
      return;
    }
    orbit.update();
    for (const cb of onFrame) {
      try {
        cb();
      } catch {
        /* a failing label updater must not kill the loop */
      }
    }
    renderer.render(scene, activeCamera);
    frame = window.requestAnimationFrame(tick);
  };
  tick();

  const world: ViewportWorld = {
    host,
    labelLayer,
    scene,
    perspCam,
    orthoCam,
    orbit,
    gizmo,
    markRoot,
    entityRoot,
    mapRoot,
    sceneMesh: null,
    projectionPool: null,
    cameraProjection: null,
    entityNodes: new Map(),
    onFrame,
    getActiveCamera: () => activeCamera,
    setActiveCamera: (useOrtho: boolean) => {
      if (!disposed) {
        setActiveCamera(useOrtho);
      }
    },
    setTheme: (next: ViewportTheme) => {
      if (theme === next || disposed) {
        return;
      }
      theme = next;
      applyThemeColors();
      rebuildGrid();
    },
    setGridVisible: (visible: boolean) => {
      if (grid) {
        grid.visible = visible;
      }
    },
    dispose: () => {
      disposed = true;
      unsubscribeLighting();
      window.cancelAnimationFrame(frame);
      ro.disconnect();
      onFrame.clear();
      gizmo.detach();
      gizmo.dispose();
      orbit.dispose();
      renderer?.dispose();
      renderer?.domElement.remove();
      renderer = null;
    },
  };
  return world;
}
