// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import {
  AmbientLight,
  BoxGeometry,
  BufferGeometry,
  Color,
  ConeGeometry,
  CylinderGeometry,
  DirectionalLight,
  DoubleSide,
  EdgesGeometry,
  GridHelper,
  Group,
  Line,
  LineBasicMaterial,
  LineLoop,
  LineSegments,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  PerspectiveCamera,
  PlaneGeometry,
  Scene,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { Button } from "../components/Button";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { FormSection } from "../components/FormSection";
import { SelectField } from "../components/SelectField";
import { TextField } from "../components/TextField";
import type { RestError } from "../lib/rest";
import {
  applyDefaultPose,
  createAssetRecord,
  defaultAsset3D,
  footprintOf,
  getAssetRecord,
  glbBasename,
  readDefaultPose,
  saveAssetRecord,
  SHIFT_TYPE_LABELS,
  simulateTrackerPath,
  type Asset3DRecord,
} from "./AssetLibrary";
import "./MarkEditor.css";

type Props = {
  /** Asset3D uid, or null to create a new class. */
  assetId: string | null;
  authToken: string;
  onClose: () => void;
  onSaved: () => void;
};

type GizmoMode = "translate" | "rotate" | "scale";

function fmt(n: number): string {
  return Number.isFinite(n) ? String(Math.round(n * 1000) / 1000) : "0";
}

/**
 * Mark editor (Phase 1.3): an isolated stage showing the class GLB with its
 * default pose applied — exactly the runtime composition
 * tracker transform × default pose × GLB. The pose frame carries a
 * move/rotate/scale gizmo with live-synced numeric fields; footprint
 * overlays show size/buffer; "Simulate tracker" drives the tracker frame
 * around a path with velocity heading (honoring rotation_from_velocity).
 *
 * Mount: <MarkEditor assetId={uid} authToken={token}
 *           onClose={…} onSaved={…} />
 */
export function MarkEditor({ assetId, authToken, onClose, onSaved }: Props) {
  const [rec, setRec] = useState<Asset3DRecord | null>(null);
  const [gizmoMode, setGizmoMode] = useState<GizmoMode>("translate");
  const [simulate, setSimulate] = useState(false);
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [clearModel, setClearModel] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmClose, setConfirmClose] = useState(false);
  const [webglFailed, setWebglFailed] = useState(false);

  const hostRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<{
    poseGroup: Group;
    trackerGroup: Group;
    gizmo: TransformControls;
    orbit: OrbitControls;
    setGizmoMode: (m: GizmoMode) => void;
    dragging: () => boolean;
    syncRecord: (r: Asset3DRecord) => void;
    setSimulate: (on: boolean) => void;
    dispose: () => void;
  } | null>(null);

  const recRef = useRef(rec);
  recRef.current = rec;

  /* ---------------- record load ---------------- */
  useEffect(() => {
    let cancelled = false;
    setError(null);
    setDirty(false);
    setModelFile(null);
    setClearModel(false);
    if (assetId == null) {
      setRec({ ...defaultAsset3D(), name: "" });
      return;
    }
    setBusy(true);
    getAssetRecord(authToken, assetId)
      .then((r) => {
        if (!cancelled) {
          setRec(r);
        }
      })
      .catch((e: RestError) => {
        if (!cancelled) {
          setError(e.message || "Failed to load asset");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setBusy(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [assetId, authToken]);

  const markDirty = useCallback(() => setDirty(true), []);

  /* ---------------- stage (isolated three.js scene) ---------------- */
  useEffect(() => {
    const host = hostRef.current;
    if (!host) {
      return;
    }
    let renderer: WebGLRenderer | null = null;
    try {
      renderer = new WebGLRenderer({ antialias: true });
    } catch {
      setWebglFailed(true);
      return;
    }

    const dark = document.documentElement.dataset.theme !== "light";
    const scene = new Scene();
    scene.background = new Color(dark ? 0x14171b : 0xe8eaed);

    const camera = new PerspectiveCamera(
      45,
      Math.max(host.clientWidth, 1) / Math.max(host.clientHeight, 1),
      0.1,
      500,
    );
    camera.up.set(0, 0, 1);
    camera.position.set(7.5, -9.5, 6);
    camera.lookAt(0, 0, 1);

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(host.clientWidth, host.clientHeight, false);
    host.replaceChildren(renderer.domElement);

    scene.add(new AmbientLight(0xffffff, 0.75));
    const key = new DirectionalLight(0xffffff, 1.1);
    key.position.set(5, -6, 10);
    scene.add(key);

    const grid = new GridHelper(30, 30, dark ? 0x3a3f47 : 0x9aa0a8, dark ? 0x262b31 : 0xc4c9d0);
    grid.rotation.x = Math.PI / 2; // XZ -> XY (Z-up floor)
    scene.add(grid);

    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.08;
    orbit.maxPolarAngle = Math.PI * 0.495;
    orbit.target.set(0, 0, 1);

    // trackerGroup × poseGroup × GLB — the runtime composition.
    const trackerGroup = new Group();
    trackerGroup.name = "tracker";
    const poseGroup = new Group();
    poseGroup.name = "default-pose";
    trackerGroup.add(poseGroup);
    scene.add(trackerGroup);

    // White arrow marking tracker forward (+X, the heading reference).
    const arrowMat = new MeshBasicMaterial({ color: 0xffffff });
    const shaft = new Mesh(new CylinderGeometry(0.05, 0.05, 0.9, 8), arrowMat);
    shaft.rotation.z = -Math.PI / 2; // Y-axis -> +X
    shaft.position.set(0.45, 0, 0.12);
    const head = new Mesh(new ConeGeometry(0.14, 0.35, 12), arrowMat);
    head.rotation.z = -Math.PI / 2;
    head.position.set(1.05, 0, 0.12);
    trackerGroup.add(shaft, head);

    // Footprint overlays (rebuilt when the record changes).
    const footprintGroup = new Group();
    trackerGroup.add(footprintGroup);
    const rebuildFootprints = () => {
      const r = recRef.current;
      footprintGroup.traverse((o) => {
        const mesh = o as Mesh | Line | LineLoop | LineSegments;
        (mesh as Mesh).geometry?.dispose?.();
        const mat = (mesh as Mesh).material as
          | { dispose?: () => void }
          | undefined;
        mat?.dispose?.();
      });
      footprintGroup.clear();
      if (!r) {
        return;
      }
      const fp = footprintOf(r);
      const color = new Color(r.markColor || "#888888");
      const inner = new Mesh(
        new PlaneGeometry(fp.inner.x, fp.inner.y),
        new MeshBasicMaterial({
          color,
          transparent: true,
          opacity: 0.22,
          side: DoubleSide,
        }),
      );
      inner.position.z = 0.02;
      const mkOutline = (w: number, h: number, opacity: number) => {
        const g = new BufferGeometry().setFromPoints([
          new Vector3(-w / 2, -h / 2, 0),
          new Vector3(w / 2, -h / 2, 0),
          new Vector3(w / 2, h / 2, 0),
          new Vector3(-w / 2, h / 2, 0),
        ]);
        const loop = new LineLoop(
          g,
          new LineBasicMaterial({ color, transparent: true, opacity }),
        );
        loop.position.z = 0.03;
        return loop;
      };
      const volume = new LineSegments(
        new EdgesGeometry(new BoxGeometry(fp.inner.x, fp.inner.y, fp.height)),
        new LineBasicMaterial({ color, transparent: true, opacity: 0.35 }),
      );
      volume.position.z = fp.height / 2;
      footprintGroup.add(
        inner,
        mkOutline(fp.inner.x, fp.inner.y, 0.9),
        mkOutline(fp.outer.x, fp.outer.y, 0.55),
        volume,
      );
    };
    rebuildFootprints();

    // GLB / placeholder model inside the pose frame.
    const loader = new GLTFLoader();
    let modelToken = 0;
    let lastModelUrl: string | null | undefined;
    let currentModel: Group | null = null;
    let placeholder: Mesh | null = null;
    const clearModel = () => {
      if (currentModel) {
        poseGroup.remove(currentModel);
        currentModel.traverse((o) => {
          const mesh = o as Mesh;
          mesh.geometry?.dispose?.();
          const mat = mesh.material as { dispose?: () => void } | undefined;
          if (Array.isArray(mat)) {
            mat.forEach((m) => m.dispose?.());
          } else {
            mat?.dispose?.();
          }
        });
        currentModel = null;
      }
      placeholder = null;
    };
    const ensurePlaceholder = () => {
      if (placeholder || currentModel) {
        return;
      }
      // Unit box, scaled to the class footprint — no GLB configured.
      placeholder = new Mesh(
        new BoxGeometry(1, 1, 1),
        new MeshStandardMaterial({
          roughness: 0.6,
          transparent: true,
          opacity: 0.85,
        }),
      );
      const holder = new Group();
      holder.add(placeholder);
      currentModel = holder;
      poseGroup.add(holder);
    };
    const setModelUrl = (url: string | null) => {
      if (url === lastModelUrl) {
        return;
      }
      lastModelUrl = url;
      const token = ++modelToken;
      clearModel();
      if (!url) {
        ensurePlaceholder();
        return;
      }
      const group = new Group();
      loader
        .loadAsync(url)
        .then((gltf) => {
          if (token !== modelToken) {
            return;
          }
          group.add(gltf.scene);
          currentModel = group;
          poseGroup.add(group);
        })
        .catch(() => {
          if (token !== modelToken) {
            return;
          }
          // Unreachable/broken GLB: fall back to the placeholder.
          lastModelUrl = null;
          ensurePlaceholder();
        });
    };

    // Gizmo on the pose frame.
    const gizmo = new TransformControls(camera, renderer.domElement);
    gizmo.setMode("translate");
    gizmo.setSize(0.9);
    scene.add(gizmo.getHelper());
    gizmo.attach(poseGroup);
    let dragging = false;
    const onDraggingChanged = (ev: { value: unknown }) => {
      dragging = Boolean(ev.value);
      orbit.enabled = !dragging;
      if (!dragging) {
        // Bake the finished drag into the record (uniform scale).
        const r = recRef.current;
        if (r) {
          const baked = readDefaultPose(poseGroup);
          setRec({ ...r, rotation: baked.rotation, translation: baked.translation, scale: baked.scale });
          setDirty(true);
        }
      }
    };
    const onObjectChange = () => {
      // Asset3D scale is uniform: mirror the X handle onto Y/Z live so the
      // gizmo never shows a non-uniform state the model cannot store.
      if (gizmo.getMode() === "scale") {
        const s = poseGroup.scale.x;
        poseGroup.scale.set(s, s, s);
      }
    };
    gizmo.addEventListener("dragging-changed", onDraggingChanged);
    gizmo.addEventListener("objectChange", onObjectChange);
    const setGizmoMode = (m: GizmoMode) => {
      gizmo.setMode(m);
      const scaleOnly = m === "scale";
      gizmo.showX = true;
      gizmo.showY = !scaleOnly;
      gizmo.showZ = !scaleOnly;
    };

    // Simulation state (driven from React).
    let simOn = false;
    const simState = { t: 0, last: performance.now() };

    const ro = new ResizeObserver(() => {
      const w = Math.max(host.clientWidth, 1);
      const h = Math.max(host.clientHeight, 1);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer?.setSize(w, h, false);
    });
    ro.observe(host);

    let disposed = false;
    let frame = 0;
    const tick = () => {
      if (disposed || !renderer) {
        return;
      }
      const now = performance.now();
      const dt = Math.min((now - simState.last) / 1000, 0.1);
      simState.last = now;
      if (simOn) {
        simState.t += dt;
        const s = simulateTrackerPath(simState.t);
        trackerGroup.position.set(s.x, s.y, 0);
        const r = recRef.current;
        trackerGroup.rotation.z =
          r?.rotationFromVelocity ? s.heading : 0;
      } else {
        trackerGroup.position.set(0, 0, 0);
        trackerGroup.rotation.z = 0;
      }
      orbit.update();
      renderer.render(scene, camera);
      frame = window.requestAnimationFrame(tick);
    };
    tick();

    stageRef.current = {
      poseGroup,
      trackerGroup,
      gizmo,
      orbit,
      setGizmoMode,
      dragging: () => dragging,
      syncRecord: (r: Asset3DRecord) => {
        rebuildFootprints();
        setModelUrl(r.modelUrl);
        if (placeholder) {
          const fp = footprintOf(r);
          placeholder.scale.set(fp.inner.x, fp.inner.y, fp.height);
          placeholder.position.z = fp.height / 2;
          (placeholder.material as MeshStandardMaterial).color.set(
            r.markColor || "#888888",
          );
        }
      },
      setSimulate: (on: boolean) => {
        simOn = on;
        simState.t = 0;
      },
      dispose: () => {
        disposed = true;
        window.cancelAnimationFrame(frame);
        ro.disconnect();
        gizmo.removeEventListener("dragging-changed", onDraggingChanged);
        gizmo.removeEventListener("objectChange", onObjectChange);
        gizmo.detach();
        gizmo.dispose();
        orbit.dispose();
        clearModel();
        scene.traverse((o) => {
          const mesh = o as Mesh | Line | LineLoop | LineSegments;
          (mesh as Mesh).geometry?.dispose?.();
          const mat = (mesh as Mesh).material as
            | { dispose?: () => void }
            | undefined;
          mat?.dispose?.();
        });
        renderer?.dispose();
        renderer = null;
      },
    };

    return () => {
      stageRef.current?.dispose();
      stageRef.current = null;
      host.replaceChildren();
    };
    // Mount once: the stage is imperative; record changes flow via effects.
  }, []);

  /* Sync record -> stage (pose, model, footprints). Skipped mid-drag so the
     gizmo never fights the fields. */
  const poseKey = rec
    ? [...rec.rotation, ...rec.translation, rec.scale].join(",")
    : "";
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || !rec || stage.dragging()) {
      return;
    }
    applyDefaultPose(stage.poseGroup, rec);
  }, [rec, poseKey]);

  const modelKey = rec ? `${rec.modelUrl}` : "";
  useEffect(() => {
    if (rec) {
      stageRef.current?.syncRecord(rec);
    }
  }, [rec, modelKey]);

  useEffect(() => {
    stageRef.current?.setGizmoMode(gizmoMode);
  }, [gizmoMode]);

  useEffect(() => {
    stageRef.current?.setSimulate(simulate);
  }, [simulate]);

  /* ---------------- keyboard ---------------- */
  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if (ev.metaKey || ev.ctrlKey || ev.altKey) {
        return;
      }
      const t = ev.target as HTMLElement | null;
      const typing =
        t &&
        (t.tagName === "INPUT" ||
          t.tagName === "TEXTAREA" ||
          t.tagName === "SELECT" ||
          t.isContentEditable);
      if (ev.key === "Escape") {
        requestClose();
        return;
      }
      if (typing) {
        return;
      }
      const k = ev.key.toLowerCase();
      if (k === "w") {
        setGizmoMode("translate");
      } else if (k === "e") {
        setGizmoMode("rotate");
      } else if (k === "r") {
        setGizmoMode("scale");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  /* ---------------- field helpers ---------------- */
  const set = useCallback(
    <K extends keyof Asset3DRecord>(key: K, value: Asset3DRecord[K]) => {
      setRec((prev) => (prev ? { ...prev, [key]: value } : prev));
      markDirty();
    },
    [markDirty],
  );

  const setTriple = useCallback(
    (
      key: "rotation" | "translation" | "geometricCenter" | "centerOfMass",
      i: number,
      raw: string,
    ) => {
      const v = Number(raw);
      setRec((prev) => {
        if (!prev) {
          return prev;
        }
        const next = [...prev[key]] as [number, number, number];
        next[i] = Number.isFinite(v) ? v : 0;
        return { ...prev, [key]: next };
      });
      markDirty();
    },
    [markDirty],
  );

  const setNum = useCallback(
    (
      key:
        | "xSize"
        | "ySize"
        | "zSize"
        | "xBuffer"
        | "yBuffer"
        | "zBuffer"
        | "scale"
        | "trackingRadius"
        | "mass"
        | "ttl"
        | "linearDamping"
        | "angularDamping"
        | "restitution",
      raw: string,
    ) => {
      const v = Number(raw);
      set(key, (Number.isFinite(v) ? v : 0) as never);
    },
    [set],
  );

  /* ---------------- save / close ---------------- */
  const requestClose = useCallback(() => {
    if (dirty) {
      setConfirmClose(true);
      return;
    }
    onClose();
  }, [dirty, onClose]);

  const save = useCallback(async () => {
    const r = recRef.current;
    if (!r || busy) {
      return;
    }
    if (!r.name.trim()) {
      setError("Class name is required.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const opts = { modelFile, clearModel };
      if (assetId == null) {
        await createAssetRecord(authToken, r, { modelFile });
      } else {
        await saveAssetRecord(authToken, assetId, r, opts);
      }
      setDirty(false);
      onSaved();
      onClose();
    } catch (e) {
      setError((e as RestError).message || "Save failed");
    } finally {
      setBusy(false);
    }
  }, [assetId, authToken, busy, modelFile, clearModel, onClose, onSaved]);

  const title = useMemo(
    () => (assetId == null ? "New class" : `Mark editor — ${rec?.name || "…"}`),
    [assetId, rec?.name],
  );

  const footprint = rec ? footprintOf(rec) : null;

  return createPortal(
    <div className="ss-mark-editor" role="dialog" aria-modal="true" aria-label={title}>
      <header className="ss-mark-editor-header">
        <div className="ss-mark-editor-title">{title}</div>
        <div className="ss-mark-editor-tools">
          <div
            className="ss-mark-editor-segmented"
            role="group"
            aria-label="Gizmo mode"
          >
            {(
              [
                ["translate", "Move", "W"],
                ["rotate", "Rotate", "E"],
                ["scale", "Scale", "R"],
              ] as [GizmoMode, string, string][]
            ).map(([m, label, key]) => (
              <button
                key={m}
                type="button"
                className={`ss-mark-editor-seg${gizmoMode === m ? " is-active" : ""}`}
                onClick={() => setGizmoMode(m)}
                title={`${label} (${key})`}
              >
                {label}
              </button>
            ))}
          </div>
          <label className="ss-mark-editor-sim">
            <input
              type="checkbox"
              checked={simulate}
              onChange={(ev) => setSimulate(ev.target.checked)}
            />
            Simulate tracker
          </label>
          {dirty ? <span className="ss-mark-editor-dirty">Unsaved</span> : null}
        </div>
        <div className="ss-mark-editor-actions">
          <Button variant="secondary" onClick={requestClose} disabled={busy}>
            Cancel
          </Button>
          <Button variant="primary" onClick={save} disabled={busy || !dirty}>
            {busy ? "Saving…" : "Save"}
          </Button>
          <button
            type="button"
            className="ss-mark-editor-close"
            aria-label="Close mark editor"
            onClick={requestClose}
          >
            ×
          </button>
        </div>
      </header>

      {simulate ? (
        <div className="ss-mark-editor-simbar">
          Driving the tracker around a 4&nbsp;m circle — white arrow marks
          tracker forward{rec?.rotationFromVelocity ? "; heading follows velocity" : "; heading locked (rotation from velocity is off)"}.
        </div>
      ) : null}

      <div className="ss-mark-editor-main">
        <div className="ss-mark-editor-stage" ref={hostRef}>
          {webglFailed ? (
            <p className="ss-mark-editor-status">
              WebGL is unavailable — the 3D stage cannot render.
            </p>
          ) : null}
        </div>

        <aside className="ss-mark-editor-props" aria-label="Class properties">
          {error ? <p className="ss-mark-editor-error">{error}</p> : null}
          {!rec ? (
            <p className="ss-mark-editor-status">Loading…</p>
          ) : (
            <form
              onSubmit={(ev) => {
                ev.preventDefault();
                save();
              }}
            >
              <FormSection title="Model" description="Class name and GLB file.">
                <TextField
                  id="ss-lib-name"
                  label="Class name"
                  value={rec.name}
                  onChange={(ev) => set("name", ev.target.value)}
                  required
                  disabled={busy}
                />
                <div className="ss-text-field">
                  <label className="ss-text-field-label" htmlFor="ss-lib-glb">
                    GLB model
                  </label>
                  <div className="ss-text-field-control">
                    {rec.modelUrl && !modelFile && !clearModel ? (
                      <p className="ss-file-current">
                        Current: {glbBasename(rec.modelUrl)}{" "}
                        <button
                          type="button"
                          className="ss-file-clear"
                          disabled={busy}
                          onClick={() => {
                            setClearModel(true);
                            setModelFile(null);
                            markDirty();
                          }}
                        >
                          Clear
                        </button>
                      </p>
                    ) : null}
                    {clearModel && !modelFile ? (
                      <p className="ss-file-current">
                        Model will be removed on save.{" "}
                        <button
                          type="button"
                          className="ss-file-clear"
                          disabled={busy}
                          onClick={() => {
                            setClearModel(false);
                            markDirty();
                          }}
                        >
                          Undo
                        </button>
                      </p>
                    ) : null}
                    {modelFile ? (
                      <p className="ss-file-current">New file: {modelFile.name}</p>
                    ) : null}
                    <input
                      id="ss-lib-glb"
                      type="file"
                      accept=".glb,model/gltf-binary"
                      disabled={busy}
                      onChange={(ev) => {
                        const file = ev.target.files?.[0] || null;
                        setModelFile(file);
                        if (file) {
                          setClearModel(false);
                        }
                        markDirty();
                      }}
                    />
                  </div>
                </div>
              </FormSection>

              <FormSection
                title="Default pose"
                description="Gizmo (W/E/R) or type exact values — they stay in sync."
              >
                <div className="ss-mark-editor-triple">
                  {(["X", "Y", "Z"] as const).map((axis, i) => (
                    <TextField
                      key={axis}
                      id={`ss-lib-rx-${axis}`}
                      label={`Rotation ${axis} (°)`}
                      type="number"
                      step="any"
                      value={fmt(rec.rotation[i])}
                      onChange={(ev) => setTriple("rotation", i, ev.target.value)}
                      disabled={busy}
                    />
                  ))}
                </div>
                <div className="ss-mark-editor-triple">
                  {(["X", "Y", "Z"] as const).map((axis, i) => (
                    <TextField
                      key={axis}
                      id={`ss-lib-tx-${axis}`}
                      label={`Translation ${axis} (m)`}
                      type="number"
                      step="any"
                      value={fmt(rec.translation[i])}
                      onChange={(ev) => setTriple("translation", i, ev.target.value)}
                      disabled={busy}
                    />
                  ))}
                </div>
                <TextField
                  id="ss-lib-scale"
                  label="Scale (uniform)"
                  type="number"
                  step="any"
                  min="0"
                  value={fmt(rec.scale)}
                  onChange={(ev) => setNum("scale", ev.target.value)}
                  disabled={busy}
                />
              </FormSection>

              <FormSection
                title="Mark"
                description="How the tracker draws this class."
              >
                <div className="ss-text-field">
                  <label className="ss-text-field-label" htmlFor="ss-lib-color">
                    Mark color
                  </label>
                  <div className="ss-text-field-control ss-lib-color-row">
                    <input
                      id="ss-lib-color"
                      type="color"
                      value={/^#[0-9a-f]{6}$/i.test(rec.markColor) ? rec.markColor : "#888888"}
                      onChange={(ev) => set("markColor", ev.target.value)}
                      disabled={busy}
                    />
                    <output htmlFor="ss-lib-color">
                      {rec.markColor.toUpperCase()}
                    </output>
                  </div>
                </div>
                <div className="ss-mark-editor-triple">
                  <TextField
                    id="ss-lib-xs"
                    label="X size (m)"
                    type="number"
                    step="any"
                    min="0"
                    value={fmt(rec.xSize)}
                    onChange={(ev) => setNum("xSize", ev.target.value)}
                    disabled={busy}
                  />
                  <TextField
                    id="ss-lib-ys"
                    label="Y size (m)"
                    type="number"
                    step="any"
                    min="0"
                    value={fmt(rec.ySize)}
                    onChange={(ev) => setNum("ySize", ev.target.value)}
                    disabled={busy}
                  />
                  <TextField
                    id="ss-lib-zs"
                    label="Z size (m)"
                    type="number"
                    step="any"
                    min="0"
                    value={fmt(rec.zSize)}
                    onChange={(ev) => setNum("zSize", ev.target.value)}
                    disabled={busy}
                  />
                </div>
                <div className="ss-mark-editor-triple">
                  <TextField
                    id="ss-lib-xb"
                    label="X buffer (m)"
                    type="number"
                    step="any"
                    min="0"
                    value={fmt(rec.xBuffer)}
                    onChange={(ev) => setNum("xBuffer", ev.target.value)}
                    disabled={busy}
                  />
                  <TextField
                    id="ss-lib-yb"
                    label="Y buffer (m)"
                    type="number"
                    step="any"
                    min="0"
                    value={fmt(rec.yBuffer)}
                    onChange={(ev) => setNum("yBuffer", ev.target.value)}
                    disabled={busy}
                  />
                  <TextField
                    id="ss-lib-zb"
                    label="Z buffer (m)"
                    type="number"
                    step="any"
                    min="0"
                    value={fmt(rec.zBuffer)}
                    onChange={(ev) => setNum("zBuffer", ev.target.value)}
                    disabled={busy}
                  />
                </div>
                {footprint ? (
                  <p className="ss-mark-editor-footnote">
                    Footprint {fmt(footprint.inner.x)}×{fmt(footprint.inner.y)} m
                    {" → "}buffered {fmt(footprint.outer.x)}×{fmt(footprint.outer.y)} m
                  </p>
                ) : null}
                <TextField
                  id="ss-lib-trackr"
                  label="Tracking radius (m)"
                  type="number"
                  step="any"
                  min="0"
                  value={fmt(rec.trackingRadius)}
                  onChange={(ev) => setNum("trackingRadius", ev.target.value)}
                  disabled={busy}
                />
                <SelectField
                  id="ss-lib-shift"
                  label="Shift type"
                  value={String(rec.shiftType)}
                  onChange={(ev) => set("shiftType", Number(ev.target.value) || 1)}
                  disabled={busy}
                >
                  {Object.entries(SHIFT_TYPE_LABELS).map(([v, label]) => (
                    <option key={v} value={v}>
                      {label}
                    </option>
                  ))}
                </SelectField>
                <label className="ss-check-row">
                  <input
                    type="checkbox"
                    checked={rec.projectToMap}
                    disabled={busy}
                    onChange={(ev) => set("projectToMap", ev.target.checked)}
                  />
                  Project to map
                </label>
                <label className="ss-check-row">
                  <input
                    type="checkbox"
                    checked={rec.rotationFromVelocity}
                    disabled={busy}
                    onChange={(ev) => set("rotationFromVelocity", ev.target.checked)}
                  />
                  Rotation from velocity
                </label>
              </FormSection>

              <FormSection
                title="Physics"
                description="Mass, damping, friction, and track lifetime."
                collapsible
                defaultOpen={false}
              >
                <div className="ss-mark-editor-triple">
                  {(["X", "Y", "Z"] as const).map((axis, i) => (
                    <TextField
                      key={axis}
                      id={`ss-lib-geo-${axis}`}
                      label={`Geometric center ${axis}`}
                      type="number"
                      step="any"
                      value={fmt(rec.geometricCenter[i])}
                      onChange={(ev) => setTriple("geometricCenter", i, ev.target.value)}
                      disabled={busy}
                    />
                  ))}
                </div>
                <TextField
                  id="ss-lib-mass"
                  label="Mass (kg)"
                  type="number"
                  step="any"
                  min="0"
                  value={fmt(rec.mass)}
                  onChange={(ev) => setNum("mass", ev.target.value)}
                  disabled={busy}
                />
                <div className="ss-mark-editor-triple">
                  {(["X", "Y", "Z"] as const).map((axis, i) => (
                    <TextField
                      key={axis}
                      id={`ss-lib-com-${axis}`}
                      label={`Center of mass ${axis}`}
                      type="number"
                      step="any"
                      value={fmt(rec.centerOfMass[i])}
                      onChange={(ev) => setTriple("centerOfMass", i, ev.target.value)}
                      disabled={busy}
                    />
                  ))}
                </div>
                <label className="ss-check-row">
                  <input
                    type="checkbox"
                    checked={rec.isStatic}
                    disabled={busy}
                    onChange={(ev) => set("isStatic", ev.target.checked)}
                  />
                  Static object
                </label>
                <TextField
                  id="ss-lib-ttl"
                  label="TTL (s, 0 = infinite)"
                  type="number"
                  step="any"
                  min="0"
                  value={fmt(rec.ttl)}
                  onChange={(ev) => setNum("ttl", ev.target.value)}
                  disabled={busy}
                />
                <TextField
                  id="ss-lib-lindamp"
                  label="Linear damping"
                  type="number"
                  step="any"
                  min="0"
                  max="1"
                  value={fmt(rec.linearDamping)}
                  onChange={(ev) => setNum("linearDamping", ev.target.value)}
                  disabled={busy}
                />
                <TextField
                  id="ss-lib-angdamp"
                  label="Angular damping"
                  type="number"
                  step="any"
                  min="0"
                  max="1"
                  value={fmt(rec.angularDamping)}
                  onChange={(ev) => setNum("angularDamping", ev.target.value)}
                  disabled={busy}
                />
                <TextField
                  id="ss-lib-rest"
                  label="Restitution"
                  type="number"
                  step="any"
                  min="0"
                  max="1"
                  value={fmt(rec.restitution)}
                  onChange={(ev) => setNum("restitution", ev.target.value)}
                  disabled={busy}
                />
                <div className="ss-mark-editor-double">
                  <TextField
                    id="ss-lib-frics"
                    label="Static friction"
                    type="number"
                    step="any"
                    min="0"
                    value={fmt(rec.friction[0])}
                    onChange={(ev) => {
                      const v = Number(ev.target.value);
                      set("friction", [
                        Number.isFinite(v) ? v : 0,
                        recRef.current?.friction[1] ?? 0.4,
                      ]);
                    }}
                    disabled={busy}
                  />
                  <TextField
                    id="ss-lib-fricd"
                    label="Dynamic friction"
                    type="number"
                    step="any"
                    min="0"
                    value={fmt(rec.friction[1])}
                    onChange={(ev) => {
                      const v = Number(ev.target.value);
                      set("friction", [
                        recRef.current?.friction[0] ?? 0.5,
                        Number.isFinite(v) ? v : 0,
                      ]);
                    }}
                    disabled={busy}
                  />
                </div>
              </FormSection>
            </form>
          )}
        </aside>
      </div>

      <ConfirmDialog
        open={confirmClose}
        title="Leave without saving?"
        confirmLabel="Leave"
        danger
        onConfirm={() => {
          setConfirmClose(false);
          onClose();
        }}
        onCancel={() => setConfirmClose(false)}
      >
        <p>You have unsaved changes to this class. Leave without saving?</p>
      </ConfirmDialog>
    </div>,
    document.body,
  );
}
