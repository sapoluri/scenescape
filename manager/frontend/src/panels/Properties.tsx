// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type Dispatch,
  type SetStateAction,
} from "react";
import { createPortal } from "react-dom";
import { useAppToast } from "../components/ToastProvider";
import { useChildStatusStore } from "../mqtt/childStatusStore";
import {
  fovFromFy,
  fyFromFov,
  opticsPersistKey,
  parseCameraOptics,
} from "../lib/cameraOptics";
import {
  entityPersistKey,
  persistCameraEntity,
  persistMapEntity,
  persistRegionEntity,
  persistSensorEntity,
  persistTripwireEntity,
  remapGeometryEntityIds,
} from "../lib/entityPersist";
import { api, type RestError } from "../lib/rest";
import {
  cameraCardFromRest,
  upsertCameraCard,
} from "../lib/sceneEntityCards";
import type { SceneCameraBootstrap, SceneChildBootstrap } from "../scene/types";
import { enterCameraCalibrate } from "../viewport/calibrate";
import { useCalibrateStore } from "../viewport/calibrate/calibrateStore";
import type { CameraOptics } from "../viewport/calibrate/types";
import { restPoseToRigEuler } from "../viewport/entities";
import {
  writeRegionToModel,
  writeTripwireToModel,
} from "../viewport/geometrySync";
import {
  markEntityPersistClean,
  usePersistStore,
} from "../viewport/persistStore";
import { getViewportState, useViewportStore } from "../viewport/store";
import type {
  CameraEntity,
  ChildSceneEntity,
  MapEntity,
  MarkEntity,
  RegionEntity,
  SensorAreaMode,
  SensorEntity,
  TripwireEntity,
  ViewportEntity,
} from "../viewport/types";
import "./Properties.css";

function fmt(n: number, digits = 2): string {
  return Number.isFinite(n) ? n.toFixed(digits) : "—";
}

function polygonArea(points: [number, number][]): number {
  let a = 0;
  for (let i = 0; i < points.length; i++) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % points.length];
    a += x1 * y2 - x2 * y1;
  }
  return Math.abs(a / 2);
}

function polylineLength(points: [number, number][]): number {
  let l = 0;
  for (let i = 1; i < points.length; i++) {
    l += Math.hypot(
      points[i][0] - points[i - 1][0],
      points[i][1] - points[i - 1][1],
    );
  }
  return l;
}

/** Round to a fixed number of fractional digits (avoids float noise in inputs). */
function roundDigits(n: number, digits: number): number {
  if (!Number.isFinite(n)) {
    return n;
  }
  const f = 10 ** digits;
  return Math.round(n * f) / f;
}

function NumField({
  label,
  value,
  step = 0.1,
  digits,
  onCommit,
}: {
  label: string;
  value: number;
  step?: number;
  /** When set, display and commit are rounded to this many fractional digits. */
  digits?: number;
  onCommit: (v: number) => void;
}) {
  const [text, setText] = useState<string | null>(null);
  const display =
    digits !== undefined && Number.isFinite(value)
      ? value.toFixed(digits)
      : String(value);
  const inputStep = digits !== undefined ? 10 ** -digits : step;
  return (
    <label className="ss-prop-field">
      <span className="ss-prop-label">{label}</span>
      <input
        type="number"
        step={inputStep}
        className="ss-prop-input"
        value={text ?? display}
        onChange={(ev) => setText(ev.target.value)}
        onBlur={() => {
          if (text !== null) {
            const v = Number(text);
            if (Number.isFinite(v)) {
              onCommit(digits !== undefined ? roundDigits(v, digits) : v);
            }
            setText(null);
          }
        }}
        onKeyDown={(ev) => {
          if (ev.key === "Enter") {
            (ev.target as HTMLInputElement).blur();
          }
        }}
      />
    </label>
  );
}

function ReadField({ label, value }: { label: string; value: string }) {
  return (
    <div className="ss-prop-field">
      <span className="ss-prop-label">{label}</span>
      <span className="ss-prop-value">{value}</span>
    </div>
  );
}

function NameRow({
  entity,
  onRenamed,
}: {
  entity: ViewportEntity;
  onRenamed?: (name: string) => void;
}) {
  const [text, setText] = useState<string | null>(null);
  const editFocus = useViewportStore((s) => s.editFocus);
  const inputRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (editFocus < 1) {
      return;
    }
    if (getViewportState().selectedId !== entity.id) {
      return;
    }
    // The placement click focuses the canvas after this effect. Focus the
    // name once that click sequence has finished.
    const timer = window.setTimeout(() => {
      if (getViewportState().selectedId !== entity.id) {
        return;
      }
      const input = inputRef.current;
      if (!input) {
        return;
      }
      input.focus();
      input.select();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [editFocus, entity.id]);
  return (
    <label className="ss-prop-field ss-prop-name">
      <span className="ss-prop-label">Name</span>
      <input
        ref={inputRef}
        className="ss-prop-input"
        value={text ?? entity.name}
        onChange={(ev) => setText(ev.target.value)}
        onBlur={() => {
          if (text !== null && text.trim()) {
            const name = text.trim();
            getViewportState().updateEntity(entity.id, { name });
            onRenamed?.(name);
          }
          setText(null);
        }}
        onKeyDown={(ev) => {
          if (ev.key === "Enter") {
            (ev.target as HTMLInputElement).blur();
          }
        }}
      />
    </label>
  );
}

/** Preset swatches for region / tripwire / sensor color (no OS picker). */
const PROP_COLOR_PRESETS = [
  "#30d158",
  "#ff9f0a",
  "#0a84ff",
  "#bf5af2",
  "#ff453a",
  "#64d2ff",
  "#ffd60a",
  "#ac8e68",
  "#8e8e93",
  "#f5f5f7",
] as const;

function normalizeHexColor(raw: string): string | null {
  const t = raw.trim();
  const m = /^#?([0-9a-fA-F]{6})$/.exec(t);
  return m ? `#${m[1].toLowerCase()}` : null;
}

function ColorRow({
  entity,
  onColor,
}: {
  entity: ViewportEntity & { color: string };
  onColor?: (color: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [hexDraft, setHexDraft] = useState(entity.color);
  const [popoverStyle, setPopoverStyle] = useState<CSSProperties>({});
  const rootRef = useRef<HTMLDivElement>(null);
  const swatchRef = useRef<HTMLButtonElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setHexDraft(entity.color);
  }, [entity.color]);

  useLayoutEffect(() => {
    if (!open) {
      return;
    }
    const place = () => {
      const swatch = swatchRef.current;
      if (!swatch) {
        return;
      }
      const r = swatch.getBoundingClientRect();
      const width = 196;
      const height = popoverRef.current?.offsetHeight || 168;
      const pad = 8;
      let top = r.bottom + 6;
      if (top + height > window.innerHeight - pad) {
        top = Math.max(pad, r.top - height - 6);
      }
      // Prefer flush with the swatch’s right edge; clamp into the viewport.
      let right = window.innerWidth - r.right;
      right = Math.min(right, window.innerWidth - width - pad);
      right = Math.max(pad, right);
      setPopoverStyle({ top, right, width });
    };
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open]);

  useEffect(() => {
    if (!open) {
      return;
    }
    const onPointer = (ev: PointerEvent) => {
      const t = ev.target as Node;
      if (rootRef.current?.contains(t) || popoverRef.current?.contains(t)) {
        return;
      }
      setOpen(false);
    };
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === "Escape") {
        setOpen(false);
      }
    };
    window.addEventListener("pointerdown", onPointer, true);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("pointerdown", onPointer, true);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const apply = (color: string) => {
    getViewportState().updateEntity(entity.id, { color });
    onColor?.(color);
    setHexDraft(color);
  };

  const current = normalizeHexColor(entity.color) || entity.color;

  const popover =
    open && typeof document !== "undefined"
      ? createPortal(
          <div
            ref={popoverRef}
            className="ss-prop-color-popover"
            role="dialog"
            aria-label="Choose color"
            style={popoverStyle}
          >
            <div className="ss-prop-color-grid" role="listbox">
              {PROP_COLOR_PRESETS.map((c) => {
                const selected = current.toLowerCase() === c;
                return (
                  <button
                    key={c}
                    type="button"
                    role="option"
                    aria-selected={selected}
                    className={`ss-prop-color-chip${selected ? " is-selected" : ""}`}
                    style={{ backgroundColor: c }}
                    title={c}
                    aria-label={c}
                    onClick={() => {
                      apply(c);
                      setOpen(false);
                    }}
                  />
                );
              })}
            </div>
            <label className="ss-prop-color-hex">
              <span className="ss-prop-color-hex-label">Hex</span>
              <input
                type="text"
                className="ss-prop-input ss-prop-color-hex-input"
                value={hexDraft}
                spellCheck={false}
                aria-label="Hex color"
                onChange={(ev) => setHexDraft(ev.target.value)}
                onBlur={() => {
                  const next = normalizeHexColor(hexDraft);
                  if (next) {
                    apply(next);
                  } else {
                    setHexDraft(entity.color);
                  }
                }}
                onKeyDown={(ev) => {
                  if (ev.key === "Enter") {
                    (ev.target as HTMLInputElement).blur();
                    setOpen(false);
                  }
                }}
              />
            </label>
          </div>,
          document.body,
        )
      : null;

  return (
    <div className="ss-prop-field ss-prop-color-field" ref={rootRef}>
      <span className="ss-prop-label">Color</span>
      <div className="ss-prop-color-control">
        <button
          ref={swatchRef}
          type="button"
          className="ss-prop-color"
          style={{ backgroundColor: current }}
          title={current}
          aria-label={`Color ${current}`}
          aria-expanded={open}
          aria-haspopup="dialog"
          onClick={() => setOpen((v) => !v)}
        />
        {popover}
      </div>
    </div>
  );
}

function SaveBar({
  dirty,
  busy,
  savedFlash,
  onSave,
  canPersist = true,
  disabledTitle,
}: {
  dirty: boolean;
  busy: boolean;
  savedFlash: boolean;
  onSave: () => void;
  canPersist?: boolean;
  disabledTitle?: string;
}) {
  const enabled = dirty && canPersist && !busy;
  return (
    <button
      type="button"
      className={`ss-prop-save${dirty && canPersist ? " is-dirty" : ""}`}
      disabled={!enabled}
      title={
        !canPersist
          ? disabledTitle || "Cannot save this item"
          : dirty
            ? "Save changes to the database"
            : "No unsaved changes"
      }
      onClick={onSave}
    >
      <i className="bi bi-check-lg" aria-hidden="true" />
      <span>{busy ? "Saving…" : savedFlash ? "Saved" : "Save"}</span>
    </button>
  );
}

function useEntityDirty(entity: ViewportEntity): boolean {
  const baseline = usePersistStore((s) => s.baselines[entity.id]);
  return baseline === undefined || baseline !== entityPersistKey(entity);
}

function RegionProps({ e }: { e: RegionEntity }) {
  const toast = useAppToast();
  const dirty = useEntityDirty(e);
  const [busy, setBusy] = useState(false);
  const [savedFlash, setSavedFlash] = useState(false);

  const patch = (p: Partial<RegionEntity>) => {
    getViewportState().updateEntity(e.id, p);
    writeRegionToModel({ ...e, ...p } as RegionEntity);
  };

  const onSave = async () => {
    if (!dirty || busy) {
      return;
    }
    setBusy(true);
    try {
      const cur = getViewportState().entities[e.id];
      if (!cur || cur.type !== "region") {
        return;
      }
      await persistRegionEntity(cur);
      setSavedFlash(true);
      window.setTimeout(() => setSavedFlash(false), 1500);
      toast.show("Region saved", "ok");
    } catch (err) {
      toast.show((err as Error).message || "Save failed", "bad");
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <NameRow
        entity={e}
        onRenamed={(name) => writeRegionToModel({ ...e, name })}
      />
      <ColorRow entity={e} />
      <ReadField label="Area" value={`${fmt(polygonArea(e.points))} m²`} />
      <ReadField label="Corners" value={String(e.points.length)} />
      <NumField
        label="Height (m)"
        value={e.height}
        onCommit={(v) => patch({ height: Math.max(0.1, v) })}
      />
      <NumField
        label="Buffer (m)"
        value={e.bufferSize}
        onCommit={(v) => patch({ bufferSize: Math.max(0, v) })}
      />
      <label className="ss-prop-check">
        <input
          type="checkbox"
          checked={e.volumetric}
          onChange={(ev) => patch({ volumetric: ev.target.checked })}
        />
        <span>Volumetric</span>
      </label>
      <SaveBar
        dirty={dirty}
        busy={busy}
        savedFlash={savedFlash}
        onSave={() => void onSave()}
      />
      <p className="ss-prop-hint">
        Drag corners in the viewport to reshape. Save writes this region; use
        toolbar Save all for bulk edits.
      </p>
    </>
  );
}

function TripwireProps({ e }: { e: TripwireEntity }) {
  const toast = useAppToast();
  const dirty = useEntityDirty(e);
  const [busy, setBusy] = useState(false);
  const [savedFlash, setSavedFlash] = useState(false);

  const onSave = async () => {
    if (!dirty || busy) {
      return;
    }
    setBusy(true);
    try {
      const cur = getViewportState().entities[e.id];
      if (!cur || cur.type !== "tripwire") {
        return;
      }
      await persistTripwireEntity(cur);
      setSavedFlash(true);
      window.setTimeout(() => setSavedFlash(false), 1500);
      toast.show("Tripwire saved", "ok");
    } catch (err) {
      toast.show((err as Error).message || "Save failed", "bad");
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <NameRow
        entity={e}
        onRenamed={(name) => writeTripwireToModel({ ...e, name })}
      />
      <ColorRow entity={e} />
      <ReadField label="Length" value={`${fmt(polylineLength(e.points))} m`} />
      <ReadField label="Points" value={String(e.points.length)} />
      <ReadField
        label="Point order"
        value={`${fmt(e.a[0])}, ${fmt(e.a[1])} → ${fmt(e.b[0])}, ${fmt(e.b[1])}`}
      />
      <SaveBar
        dirty={dirty}
        busy={busy}
        savedFlash={savedFlash}
        onSave={() => void onSave()}
      />
      <p className="ss-prop-hint">
        The arrow marks the +1 crossing side. Save writes this tripwire; use
        toolbar Save all for bulk edits.
      </p>
    </>
  );
}

function CameraProps({
  e,
  sensorId,
  sceneId,
  calibrateHref,
  authToken,
  onCalibrate,
  onCamerasChange,
}: {
  e: CameraEntity;
  sensorId?: string;
  sceneId?: string;
  calibrateHref?: string;
  authToken?: string;
  onCalibrate?: () => void;
  onCamerasChange?: Dispatch<SetStateAction<SceneCameraBootstrap[]>>;
}) {
  const toast = useAppToast();
  const isDraft = e.id.startsWith("draft-");
  const projecting = useViewportStore((s) => s.projectCameraIds.includes(e.id));
  // Wizard toolbar owns Auto Calibrate + Save while this camera is calibrating.
  const calibrating = useCalibrateStore((s) => s.active && s.cameraId === e.id);
  const calOptics = useCalibrateStore((s) =>
    s.active && s.cameraId === e.id ? s.optics : null,
  );
  const poseDirty = useEntityDirty(e);
  const [optics, setOptics] = useState<CameraOptics | null>(null);
  const [opticsBaseline, setOpticsBaseline] = useState<string | null>(null);
  const opticsDirty =
    optics != null &&
    opticsBaseline != null &&
    opticsPersistKey(optics) !== opticsBaseline;
  const dirty = poseDirty || opticsDirty;
  const [saveBusy, setSaveBusy] = useState(false);
  const [savedFlash, setSavedFlash] = useState(false);

  useEffect(() => {
    if (!authToken) {
      setOptics(null);
      setOpticsBaseline(null);
      return;
    }
    // Draft cameras have no server row yet — seed default optics locally.
    if (!sensorId || isDraft) {
      const o = parseCameraOptics({}, e.name);
      setOptics(o);
      setOpticsBaseline(opticsPersistKey(o));
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const row = (await api.getCamera(authToken, sensorId)) ?? {};
        if (cancelled) {
          return;
        }
        const o = parseCameraOptics(row, e.name);
        setOptics(o);
        setOpticsBaseline(opticsPersistKey(o));
      } catch {
        if (!cancelled) {
          const o = parseCameraOptics({}, e.name);
          setOptics(o);
          setOpticsBaseline(opticsPersistKey(o));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [sensorId, authToken, e.name, e.id, isDraft]);

  // Live fx/fy (and friends) from in-viewport calibrate (≥6 pairs).
  useEffect(() => {
    if (!calibrating || !calOptics) {
      return;
    }
    setOptics(calOptics);
  }, [calibrating, calOptics]);

  const patch = (p: Partial<CameraEntity>) =>
    getViewportState().updateEntity(e.id, p);
  const setPos = (i: number, v: number) => {
    const position: [number, number, number] = [...e.position];
    position[i] = roundDigits(v, 3);
    patch({ position, posed: true });
  };
  const setRot = (i: number, v: number) => {
    const rotation: [number, number, number] = [...e.rotation];
    rotation[i] = roundDigits(v, 2);
    // Rig ↔ REST is the same Rx(π) involution; keep restRotation in sync
    // so projection / save stay consistent with a manual pose edit.
    patch({
      rotation,
      restRotation: restPoseToRigEuler(rotation),
      posed: true,
    });
  };

  const patchOptics = (p: Partial<CameraOptics>) => {
    setOptics((prev) => {
      if (!prev) {
        return prev;
      }
      const next = { ...prev, ...p };
      if (calibrating) {
        useCalibrateStore.getState().setOptics(next);
      }
      return next;
    });
  };

  const onFovCommit = (v: number) => {
    const fov = roundDigits(Math.min(120, Math.max(10, v)), 2);
    patch({ fov });
    if (optics) {
      patchOptics({ fy: roundDigits(fyFromFov(fov, optics.height), 2) });
    }
  };

  const onSave = async () => {
    if (!dirty || saveBusy || !authToken || !sceneId) {
      return;
    }
    setSaveBusy(true);
    try {
      const cur = getViewportState().entities[e.id];
      if (!cur || cur.type !== "camera") {
        return;
      }
      const restRotation = restPoseToRigEuler(cur.rotation);
      let persistId = e.id;
      let persistSensorId = sensorId;

      if (isDraft || !persistSensorId) {
        const base =
          cur.name.trim().replace(/\s+/g, "_") ||
          `camera_${Date.now().toString(36)}`;
        let sensor_id = base;
        const existing = new Set(
          Object.values(getViewportState().entities)
            .filter((x) => x.type === "camera" && x.id !== e.id)
            .map((x) => x.id),
        );
        let n = 2;
        while (existing.has(sensor_id)) {
          sensor_id = `${base}_${n}`;
          n += 1;
        }
        const payload: Record<string, unknown> = {
          name: cur.name,
          sensor_id,
          scene: sceneId,
          transform_type: "euler",
          translation: cur.position,
          rotation: restRotation,
          scale: [1, 1, 1],
        };
        if (optics) {
          payload.intrinsics = {
            fx: optics.fx,
            fy: optics.fy,
            cx: optics.cx,
            cy: optics.cy,
          };
          payload.distortion = {
            k1: optics.k1,
            k2: optics.k2,
            p1: optics.p1,
            p2: optics.p2,
            k3: optics.k3,
          };
          payload.resolution = {
            width: optics.width,
            height: optics.height,
          };
        }
        const created = await api.createCamera(authToken, payload);
        const card = cameraCardFromRest(created);
        if (!card) {
          throw new Error("Camera create returned no id");
        }
        remapGeometryEntityIds({ [e.id]: card.id });
        onCamerasChange?.((prev) => upsertCameraCard(prev, card));
        persistId = card.id;
        persistSensorId = card.sensorId;
      } else {
        await persistCameraEntity(
          authToken,
          sceneId,
          persistSensorId,
          cur,
          optics ?? undefined,
        );
      }

      getViewportState().updateEntity(persistId, {
        restRotation,
        posed: true,
      });
      markEntityPersistClean(persistId);
      if (optics) {
        setOpticsBaseline(opticsPersistKey(optics));
      }
      setSavedFlash(true);
      window.setTimeout(() => setSavedFlash(false), 1500);
      toast.show("Camera saved", "ok");
    } catch (err) {
      toast.show((err as RestError).message || "Save failed", "bad");
    } finally {
      setSaveBusy(false);
    }
  };

  const projectOn = calibrating || projecting;
  const projectDisabled = calibrating || !e.posed;

  return (
    <>
      <NameRow entity={e} />
      <span className="ss-prop-section">Position</span>
      <div className="ss-prop-grid3">
        <NumField
          label="X"
          value={e.position[0]}
          digits={3}
          onCommit={(v) => setPos(0, v)}
        />
        <NumField
          label="Y"
          value={e.position[1]}
          digits={3}
          onCommit={(v) => setPos(1, v)}
        />
        <NumField
          label="Z"
          value={e.position[2]}
          digits={3}
          onCommit={(v) => setPos(2, v)}
        />
      </div>
      <span className="ss-prop-section">Rotation (°)</span>
      <div className="ss-prop-grid3">
        <NumField
          label="X"
          value={e.rotation[0]}
          digits={2}
          onCommit={(v) => setRot(0, v)}
        />
        <NumField
          label="Y"
          value={e.rotation[1]}
          digits={2}
          onCommit={(v) => setRot(1, v)}
        />
        <NumField
          label="Z"
          value={e.rotation[2]}
          digits={2}
          onCommit={(v) => setRot(2, v)}
        />
      </div>
      <NumField
        label="FOV (°)"
        value={e.fov}
        digits={2}
        onCommit={onFovCommit}
      />
      {optics ? (
        <>
          <span className="ss-prop-section">Intrinsics (px)</span>
          <div className="ss-prop-grid2">
            <NumField
              label="fx"
              value={optics.fx}
              digits={2}
              onCommit={(v) => patchOptics({ fx: v })}
            />
            <NumField
              label="fy"
              value={optics.fy}
              digits={2}
              onCommit={(v) => {
                const fy = roundDigits(v, 2);
                patchOptics({ fy });
                patch({
                  fov: roundDigits(
                    Math.min(120, Math.max(10, fovFromFy(fy, optics.height))),
                    2,
                  ),
                });
              }}
            />
            <NumField
              label="cx"
              value={optics.cx}
              digits={2}
              onCommit={(v) => patchOptics({ cx: v })}
            />
            <NumField
              label="cy"
              value={optics.cy}
              digits={2}
              onCommit={(v) => patchOptics({ cy: v })}
            />
          </div>
          <span className="ss-prop-section">Distortion</span>
          <div className="ss-prop-grid3">
            <NumField
              label="k1"
              value={optics.k1}
              digits={4}
              onCommit={(v) => patchOptics({ k1: v })}
            />
            <NumField
              label="k2"
              value={optics.k2}
              digits={4}
              onCommit={(v) => patchOptics({ k2: v })}
            />
            <NumField
              label="k3"
              value={optics.k3}
              digits={4}
              onCommit={(v) => patchOptics({ k3: v })}
            />
          </div>
          <div className="ss-prop-grid3">
            <NumField
              label="p1"
              value={optics.p1}
              digits={4}
              onCommit={(v) => patchOptics({ p1: v })}
            />
            <NumField
              label="p2"
              value={optics.p2}
              digits={4}
              onCommit={(v) => patchOptics({ p2: v })}
            />
          </div>
        </>
      ) : null}
      <ReadField label="Pose" value={e.posed ? "Calibrated" : "Unset"} />
      <label
        className={`ss-prop-switch${projectDisabled ? " is-disabled" : ""}`}
        title={
          calibrating
            ? "Projection stays on while calibrating"
            : e.posed
              ? "Project the live camera frame onto the scene map"
              : "Calibrate this camera first — projection needs a pose"
        }
      >
        <span className="ss-prop-label">Project frame</span>
        <span
          className={`ss-prop-switch-control${projectOn ? " is-on" : ""}`}
        >
          <input
            type="checkbox"
            className="ss-prop-switch-input"
            role="switch"
            checked={projectOn}
            disabled={projectDisabled}
            aria-checked={projectOn}
            onChange={(ev) =>
              getViewportState().setProjectCameraEnabled(
                e.id,
                ev.target.checked,
              )
            }
          />
          <span className="ss-prop-switch-track" aria-hidden="true">
            <span className="ss-prop-switch-thumb" />
          </span>
        </span>
      </label>
      {!calibrating ? (
        <div className="ss-prop-actions">
          <SaveBar
            dirty={dirty}
            busy={saveBusy}
            savedFlash={savedFlash}
            canPersist={Boolean(authToken && sceneId && (sensorId || isDraft))}
            onSave={() => void onSave()}
            disabledTitle="Sign in and open a scene to save this camera"
          />
          {onCalibrate || calibrateHref ? (
            onCalibrate ? (
              <button
                type="button"
                className="ss-prop-calibrate"
                onClick={onCalibrate}
              >
                <i className="bi bi-bullseye" aria-hidden="true" />
                <span>Calibrate</span>
              </button>
            ) : (
              <a href={calibrateHref} className="ss-prop-calibrate">
                <i className="bi bi-bullseye" aria-hidden="true" />
                <span>Calibrate</span>
              </a>
            )
          ) : null}
        </div>
      ) : null}
      <p className="ss-prop-hint">
        {calibrating
          ? "Use the toolbar to Auto Calibrate or Save point correspondences. Edit pose and optics here; they apply after Save."
          : isDraft
            ? "Save creates this camera on the server with the current pose. Frames are not required. Delete from the Outliner."
            : "Save writes pose, FOV/fy, intrinsics, and distortion. Calibrate opens the point-pair wizard (Auto Calibrate is on the wizard toolbar). Delete from the Outliner."}
      </p>
    </>
  );
}

function SensorProps({
  e,
  sensorId,
  sceneId,
  authToken,
}: {
  e: SensorEntity;
  sensorId?: string;
  sceneId?: string;
  authToken?: string;
}) {
  const toast = useAppToast();
  const dirty = useEntityDirty(e);
  const [busy, setBusy] = useState(false);
  const [savedFlash, setSavedFlash] = useState(false);
  const patch = (p: Partial<SensorEntity>) =>
    getViewportState().updateEntity(e.id, p);
  const setPos = (i: number, v: number) => {
    const position: [number, number, number] = [...e.position];
    position[i] = roundDigits(v, 3);
    patch({ position });
  };

  const onSave = async () => {
    if (!dirty || busy || !sensorId || !authToken || !sceneId) {
      return;
    }
    setBusy(true);
    try {
      const cur = getViewportState().entities[e.id];
      if (!cur || cur.type !== "sensor") {
        return;
      }
      await persistSensorEntity(authToken, sceneId, sensorId, cur);
      markEntityPersistClean(e.id);
      setSavedFlash(true);
      window.setTimeout(() => setSavedFlash(false), 1500);
      toast.show("Sensor saved", "ok");
    } catch (err) {
      toast.show((err as RestError).message || "Save failed", "bad");
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <NameRow entity={e} />
      <ColorRow entity={e} />
      <div className="ss-prop-grid3">
        <NumField
          label="X"
          value={e.position[0]}
          digits={3}
          onCommit={(v) => setPos(0, v)}
        />
        <NumField
          label="Y"
          value={e.position[1]}
          digits={3}
          onCommit={(v) => setPos(1, v)}
        />
        <NumField
          label="Z"
          value={e.position[2]}
          digits={3}
          onCommit={(v) => setPos(2, v)}
        />
      </div>
      <label className="ss-prop-field">
        <span className="ss-prop-label">Area</span>
        <select
          className="ss-prop-input ss-prop-select"
          value={e.areaMode}
          disabled={busy}
          onChange={(ev) => {
            const areaMode = ev.target.value as SensorAreaMode;
            // Switching into polygon clears vertices so floor-click drawing
            // starts immediately in the 3D viewport.
            if (areaMode === "poly" && e.areaMode !== "poly") {
              patch({ areaMode, points: [] });
            } else if (areaMode !== "poly") {
              patch({ areaMode, points: [] });
            } else {
              patch({ areaMode });
            }
          }}
          aria-label="Sensor coverage area"
        >
          <option value="scene">Full scene</option>
          <option value="circle">Circle</option>
          <option value="poly">Polygon</option>
        </select>
      </label>
      {e.areaMode === "circle" ? (
        <NumField
          label="Radius (m)"
          value={e.radius}
          onCommit={(v) => patch({ radius: Math.max(0.5, v) })}
        />
      ) : null}
      {e.areaMode === "poly" ? (
        <div className="ss-prop-actions">
          <ReadField
            label="Vertices"
            value={
              e.points.length >= 3
                ? String(e.points.length)
                : e.points.length === 0
                  ? "Click floor to draw"
                  : `${e.points.length} (need 3+)`
            }
          />
          {e.points.length >= 3 ? (
            <button
              type="button"
              className="ss-prop-calibrate"
              disabled={busy}
              onClick={() => patch({ points: [] })}
            >
              <i className="bi bi-pencil" aria-hidden="true" />
              <span>Redraw polygon</span>
            </button>
          ) : null}
        </div>
      ) : null}
      <SaveBar
        dirty={dirty}
        busy={busy}
        savedFlash={savedFlash}
        canPersist={Boolean(sensorId && authToken && sceneId)}
        onSave={() => void onSave()}
        disabledTitle="Sensor is not linked to the server yet"
      />
      <p className="ss-prop-hint">
        {e.areaMode === "scene"
          ? "Full scene coverage. Save writes name and area to the database."
          : e.areaMode === "circle"
            ? "Circle uses X/Y as center. Save writes area and radius to the database."
            : e.points.length >= 3
              ? "Polygon is ready. Save writes vertices and center to the database."
              : "Click the floor to place vertices. Click the first point (or Enter / double-click) to close."}
      </p>
    </>
  );
}

function MarkProps({ e }: { e: MarkEntity }) {
  return (
    <>
      <ReadField label="Class" value={e.className} />
      <ReadField
        label="Position"
        value={`${fmt(e.position[0])}, ${fmt(e.position[1])}, ${fmt(e.position[2])}`}
      />
      <ReadField
        label="Heading"
        value={
          Number.isFinite(e.heading)
            ? `${fmt((e.heading * 180) / Math.PI, 1)}°`
            : "—"
        }
      />
      <ReadField label="Speed" value={`${fmt(e.speed)} m/frame`} />
      <p className="ss-prop-hint">
        Live tracked object — values update from the tracker feed. Mark shape
        and color come from the Object Library (B).
      </p>
    </>
  );
}

function ChildProps({
  e,
  boot,
  isSuperuser,
}: {
  e: ChildSceneEntity;
  boot?: SceneChildBootstrap;
  isSuperuser: boolean;
}) {
  const remoteId = e.remoteChildId || boot?.remoteChildId;
  const statusEntry = useChildStatusStore((s) =>
    remoteId ? s.byRemoteId[remoteId] : undefined,
  );
  const detailUrl = e.detailUrl || boot?.detailUrl;
  const editHref = e.editHref || boot?.editHref;
  const childType = e.childType || boot?.childType || "local";
  const transformSource = e.transformSource || "—";
  const statusLabel =
    childType !== "remote"
      ? "Local"
      : statusEntry?.status === "connected"
        ? "Connected"
        : statusEntry?.status === "disconnected"
          ? "Disconnected"
          : statusEntry?.status === "error"
            ? statusEntry.message || "Error"
            : "Unknown";

  return (
    <>
      <NameRow entity={e} />
      <ReadField
        label="Type"
        value={childType === "remote" ? "Remote" : "Local"}
      />
      <ReadField label="Status" value={statusLabel} />
      <ReadField
        label="Transform"
        value={typeof transformSource === "string" ? transformSource : "—"}
      />
      <ReadField
        label="Position"
        value={`${fmt(e.position[0])} , ${fmt(e.position[1])} , ${fmt(e.position[2])}`}
      />
      <ReadField
        label="Rotation"
        value={`${fmt(e.rotation[0])}° , ${fmt(e.rotation[1])}° , ${fmt(e.rotation[2])}°`}
      />
      <div className="ss-prop-actions">
        {detailUrl ? (
          <a href={detailUrl} className="ss-prop-calibrate">
            <i className="bi bi-box-arrow-up-right" aria-hidden="true" />
            <span>Open child scene</span>
          </a>
        ) : null}
        {isSuperuser && editHref ? (
          <a href={editHref} className="ss-prop-calibrate">
            <i className="bi bi-pencil" aria-hidden="true" />
            <span>Edit link</span>
          </a>
        ) : null}
      </div>
      <p className="ss-prop-hint">
        Pose editing stays in the child placement workspace. Use Edit link to
        change transform or connection settings. Unlink from the Outliner.
      </p>
    </>
  );
}

function MapProps({
  e,
  sceneId,
  authToken,
}: {
  e: MapEntity;
  sceneId?: string;
  authToken?: string;
}) {
  const toast = useAppToast();
  const dirty = useEntityDirty(e);
  const [busy, setBusy] = useState(false);
  const [savedFlash, setSavedFlash] = useState(false);

  const patch = (p: Partial<MapEntity>) =>
    getViewportState().updateEntity(e.id, p);

  const setPos = (i: number, v: number) => {
    const position: [number, number, number] = [...e.position];
    position[i] = roundDigits(v, 3);
    patch({ position });
  };
  const setRot = (i: number, v: number) => {
    const rotation: [number, number, number] = [...e.rotation];
    rotation[i] = roundDigits(v, 2);
    patch({ rotation });
  };
  const setScale = (i: number, v: number) => {
    const scale: [number, number, number] = [...e.scale];
    scale[i] = roundDigits(Math.max(1e-6, v), 4);
    patch({ scale });
  };

  const onSave = async () => {
    if (!dirty || busy || !authToken || !sceneId) {
      return;
    }
    setBusy(true);
    try {
      const cur = getViewportState().entities[e.id];
      if (!cur || cur.type !== "map") {
        return;
      }
      await persistMapEntity(authToken, sceneId, cur);
      markEntityPersistClean(e.id);
      setSavedFlash(true);
      window.setTimeout(() => setSavedFlash(false), 1500);
      toast.show("Scene map pose saved", "ok");
    } catch (err) {
      toast.show((err as RestError).message || "Save failed", "bad");
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <NameRow entity={e} />
      <span className="ss-prop-section">Position</span>
      <div className="ss-prop-grid3">
        <NumField
          label="X"
          value={e.position[0]}
          digits={3}
          onCommit={(v) => setPos(0, v)}
        />
        <NumField
          label="Y"
          value={e.position[1]}
          digits={3}
          onCommit={(v) => setPos(1, v)}
        />
        <NumField
          label="Z"
          value={e.position[2]}
          digits={3}
          onCommit={(v) => setPos(2, v)}
        />
      </div>
      <span className="ss-prop-section">Rotation (°)</span>
      <div className="ss-prop-grid3">
        <NumField
          label="X"
          value={e.rotation[0]}
          digits={2}
          onCommit={(v) => setRot(0, v)}
        />
        <NumField
          label="Y"
          value={e.rotation[1]}
          digits={2}
          onCommit={(v) => setRot(1, v)}
        />
        <NumField
          label="Z"
          value={e.rotation[2]}
          digits={2}
          onCommit={(v) => setRot(2, v)}
        />
      </div>
      <span className="ss-prop-section">Scale</span>
      <div className="ss-prop-grid3">
        <NumField
          label="X"
          value={e.scale[0]}
          digits={4}
          onCommit={(v) => setScale(0, v)}
        />
        <NumField
          label="Y"
          value={e.scale[1]}
          digits={4}
          onCommit={(v) => setScale(1, v)}
        />
        <NumField
          label="Z"
          value={e.scale[2]}
          digits={4}
          onCommit={(v) => setScale(2, v)}
        />
      </div>
      <div className="ss-prop-actions">
        <SaveBar
          dirty={dirty}
          busy={busy}
          savedFlash={savedFlash}
          canPersist={Boolean(authToken && sceneId)}
          onSave={() => void onSave()}
          disabledTitle="Sign in and open a scene to save the map pose"
        />
      </div>
      <p className="ss-prop-hint">
        Use Move / Rotate / Scale on the left toolbar (or W / E / R) to transform
        the uploaded GLB. Save writes mesh pose to the scene.
      </p>
    </>
  );
}

function EmptyProps({ sceneName }: { sceneName: string }) {
  const entities = useViewportStore((s) => s.entities);
  const counts = {
    map: 0,
    region: 0,
    tripwire: 0,
    camera: 0,
    sensor: 0,
    mark: 0,
    child: 0,
  };
  for (const e of Object.values(entities)) {
    if (e.type in counts) {
      counts[e.type as keyof typeof counts] += 1;
    }
  }
  return (
    <>
      <ReadField label="Scene" value={sceneName} />
      <ReadField label="Scene map" value={counts.map ? "GLB" : "—"} />
      <ReadField label="Regions" value={String(counts.region)} />
      <ReadField label="Tripwires" value={String(counts.tripwire)} />
      <ReadField label="Cameras" value={String(counts.camera)} />
      <ReadField label="Sensors" value={String(counts.sensor)} />
      <ReadField label="Child scenes" value={String(counts.child)} />
      <ReadField label="Tracked now" value={String(counts.mark)} />
      <p className="ss-prop-hint">
        Select anything in the viewport or Outliner to inspect it. Press{" "}
        <kbd>G</kbd> to draw a region, <kbd>T</kbd> for a tripwire.
      </p>
    </>
  );
}

/**
 * Properties: Blender-style inspector for the current selection.
 * Numeric fields write through the viewport store so the 3D view updates
 * live. Nothing selected → scene summary.
 */
export function Properties({
  sceneName,
  sceneId,
  cameras = [],
  sensors = [],
  children = [],
  onCamerasChange,
  authToken = "",
  isSuperuser = false,
}: {
  sceneName: string;
  sceneId?: string;
  cameras?: SceneCameraBootstrap[];
  sensors?: {
    id: string;
    sensorId: string;
    name: string;
    calibrateHref: string;
    deleteUrl?: string | null;
  }[];
  children?: SceneChildBootstrap[];
  onCamerasChange?: Dispatch<SetStateAction<SceneCameraBootstrap[]>>;
  authToken?: string;
  isSuperuser?: boolean;
}) {
  const selectedId = useViewportStore((s) => s.selectedId);
  const entities = useViewportStore((s) => s.entities);
  const e = selectedId ? entities[selectedId] : undefined;
  const camBoot =
    e?.type === "camera" ? cameras.find((c) => c.id === e.id) : undefined;
  const senBoot =
    e?.type === "sensor" ? sensors.find((s) => s.id === e.id) : undefined;
  const childBoot =
    e?.type === "child"
      ? children.find(
          (c) => c.id === e.id || c.restUid === e.id || c.restUid === e.restUid,
        )
      : undefined;

  return (
    <div className="ss-props">
      {!e && <EmptyProps sceneName={sceneName} />}
      {e?.type === "map" && (
        <MapProps e={e} sceneId={sceneId} authToken={authToken} />
      )}
      {e?.type === "region" && <RegionProps e={e} />}
      {e?.type === "tripwire" && <TripwireProps e={e} />}
      {e?.type === "camera" && (
        <CameraProps
          e={e}
          sensorId={camBoot?.sensorId}
          sceneId={sceneId}
          calibrateHref={camBoot?.calibrateHref}
          authToken={authToken}
          onCamerasChange={onCamerasChange}
          onCalibrate={
            camBoot
              ? () =>
                  enterCameraCalibrate({
                    cameraId: camBoot.id,
                    sensorId: camBoot.sensorId,
                    cameraName: camBoot.name,
                  })
              : undefined
          }
        />
      )}
      {e?.type === "sensor" && (
        <SensorProps
          e={e}
          sensorId={senBoot?.sensorId}
          sceneId={sceneId}
          authToken={authToken}
        />
      )}
      {e?.type === "mark" && <MarkProps e={e} />}
      {e?.type === "child" && (
        <ChildProps e={e} boot={childBoot} isSuperuser={isSuperuser} />
      )}
    </div>
  );
}
