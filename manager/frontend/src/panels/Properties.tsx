// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { useAppToast } from "../components/ToastProvider";
import {
  frameToBase64,
  humanAutocalibMessage,
  resolveAutocalibAvailability,
  startCameraAutocalibration,
} from "../lib/autocalibration";
import {
  fovFromFy,
  fyFromFov,
  opticsPersistKey,
  parseCameraOptics,
} from "../lib/cameraOptics";
import {
  entityPersistKey,
  persistCameraEntity,
  persistRegionEntity,
  persistSensorEntity,
  persistTripwireEntity,
} from "../lib/entityPersist";
import { api, type RestError } from "../lib/rest";
import { restDeleteTarget } from "../lib/restDelete";
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
  return (
    <label className="ss-prop-field ss-prop-name">
      <span className="ss-prop-label">Name</span>
      <input
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

function ColorRow({
  entity,
  onColor,
}: {
  entity: ViewportEntity & { color: string };
  onColor?: (color: string) => void;
}) {
  return (
    <label className="ss-prop-field">
      <span className="ss-prop-label">Color</span>
      <input
        type="color"
        className="ss-prop-color"
        value={entity.color}
        onChange={(ev) => {
          getViewportState().updateEntity(entity.id, {
            color: ev.target.value,
          });
          onColor?.(ev.target.value);
        }}
      />
    </label>
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
        label="Direction"
        value={`${fmt(e.a[0])}, ${fmt(e.a[1])} → ${fmt(e.b[0])}, ${fmt(e.b[1])}`}
      />
      <SaveBar
        dirty={dirty}
        busy={busy}
        savedFlash={savedFlash}
        onSave={() => void onSave()}
      />
      <p className="ss-prop-hint">
        Crossings are counted A → B. Save writes this tripwire; use toolbar
        Save all for bulk edits.
      </p>
    </>
  );
}

function frameSrcForSensor(sensorId: string): string | null {
  const imgs = document.querySelectorAll<HTMLImageElement>(
    `img[data-ss-card-sensor="${CSS.escape(sensorId)}"]`,
  );
  for (const img of imgs) {
    const src = img.currentSrc || img.getAttribute("src") || "";
    if (!src || src.includes("offline.png")) {
      continue;
    }
    if (img.naturalWidth > 0 || src.startsWith("data:image")) {
      return src;
    }
  }
  return null;
}

function CameraProps({
  e,
  sensorId,
  sceneId,
  calibrateHref,
  deleteUrl,
  authToken,
  sceneHomeHref,
  onCalibrate,
}: {
  e: CameraEntity;
  sensorId?: string;
  sceneId?: string;
  calibrateHref?: string;
  deleteUrl?: string | null;
  authToken?: string;
  sceneHomeHref?: string;
  onCalibrate?: () => void;
}) {
  const toast = useAppToast();
  const projectCameraId = useViewportStore((s) => s.projectCameraId);
  const projectOpacity = useViewportStore((s) => s.projectOpacity);
  const projecting = projectCameraId === e.id;
  // Wizard toolbar owns Auto Calibrate + Save while this camera is calibrating.
  const calibrating = useCalibrateStore(
    (s) => s.active && s.cameraId === e.id,
  );
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
  const [autoBusy, setAutoBusy] = useState(false);
  const [autoReady, setAutoReady] = useState(false);
  const [autoTitle, setAutoTitle] = useState(
    "Checking auto-calibration service…",
  );
  const [confirmDelete, setConfirmDelete] = useState(false);

  useEffect(() => {
    if (!sensorId || !authToken) {
      setOptics(null);
      setOpticsBaseline(null);
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
  }, [sensorId, authToken, e.name, e.id]);

  // Live fx/fy (and friends) from in-viewport calibrate (≥6 pairs).
  useEffect(() => {
    if (!calibrating || !calOptics) {
      return;
    }
    setOptics(calOptics);
  }, [calibrating, calOptics]);
  const [deleteBusy, setDeleteBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!sceneId) {
        setAutoReady(false);
        setAutoTitle("Auto Calibrate system not available for this scene");
        return;
      }
      const avail = await resolveAutocalibAvailability(sceneId, authToken);
      if (cancelled) {
        return;
      }
      setAutoReady(avail.ready);
      setAutoTitle(avail.title);
    })();
    return () => {
      cancelled = true;
    };
  }, [sensorId, sceneId, authToken]);

  const patch = (p: Partial<CameraEntity>) =>
    getViewportState().updateEntity(e.id, p);
  const setPos = (i: number, v: number) => {
    const position: [number, number, number] = [...e.position];
    position[i] = roundDigits(v, 3);
    patch({ position });
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
    if (!dirty || saveBusy || !sensorId || !authToken || !sceneId) {
      return;
    }
    setSaveBusy(true);
    try {
      const cur = getViewportState().entities[e.id];
      if (!cur || cur.type !== "camera") {
        return;
      }
      await persistCameraEntity(
        authToken,
        sceneId,
        sensorId,
        cur,
        optics ?? undefined,
      );
      const restRotation = restPoseToRigEuler(cur.rotation);
      getViewportState().updateEntity(e.id, {
        restRotation,
        posed: true,
      });
      markEntityPersistClean(e.id);
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

  const onAutoCalibrate = async () => {
    if (!sensorId || !authToken || autoBusy) {
      return;
    }
    const src = frameSrcForSensor(sensorId);
    if (!src) {
      toast.show("Enable Live View and wait for a camera frame first", "bad");
      return;
    }
    setAutoBusy(true);
    try {
      const row = (await api.getCamera(authToken, sensorId)) ?? {};
      const intr = (row.intrinsics as Record<string, number>) || {};
      const fx = Number(intr.fx) || 905;
      const fy = Number(intr.fy) || fx;
      const cx = Number(intr.cx) || 640;
      const cy = Number(intr.cy) || 360;
      const image = await frameToBase64(src);
      const result = await startCameraAutocalibration(sensorId, image, [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
      ]);
      if (result.status === "error") {
        toast.show(
          humanAutocalibMessage(result, "Auto calibrate failed"),
          "bad",
        );
      } else {
        toast.show("Auto calibrate started", "ok");
      }
    } catch (err) {
      toast.show((err as Error).message || "Auto calibrate failed", "bad");
    } finally {
      setAutoBusy(false);
    }
  };

  const onDelete = async () => {
    if (!deleteUrl || !authToken) {
      return;
    }
    setDeleteBusy(true);
    try {
      const path = deleteUrl.startsWith("http")
        ? new URL(deleteUrl).pathname
        : deleteUrl;
      const m = path.match(/\/cam\/delete\/([^/]+)\/?$/);
      if (!m) {
        throw new Error("Unrecognized camera delete URL");
      }
      await restDeleteTarget(authToken, {
        kind: "camera",
        uid: decodeURIComponent(m[1]),
      });
      getViewportState().removeEntity(e.id);
      getViewportState().setProjectCamera(null);
      toast.show(`Deleted ${e.name}`, "ok");
      setConfirmDelete(false);
      if (sceneHomeHref) {
        window.location.assign(sceneHomeHref);
      } else {
        window.location.reload();
      }
    } catch (err) {
      toast.show((err as RestError).message || "Delete failed", "bad");
      setDeleteBusy(false);
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
      <NumField
        label="FOV (°)"
        value={e.fov}
        digits={2}
        onCommit={onFovCommit}
      />
      {optics ? (
        <>
          <span className="ss-prop-section">Intrinsics (px)</span>
          <div className="ss-prop-grid3">
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
          </div>
          <NumField
            label="cy"
            value={optics.cy}
            digits={2}
            onCommit={(v) => patchOptics({ cy: v })}
          />
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
        className={`ss-prop-check${e.posed || calibrating ? "" : " is-disabled"}`}
        title={
          calibrating
            ? "Projection stays on while calibrating"
            : e.posed
              ? "Project the live camera frame onto the scene map"
              : "Calibrate this camera first — projection needs a pose"
        }
      >
        <input
          type="checkbox"
          checked={calibrating || projecting}
          disabled={calibrating || !e.posed}
          onChange={(ev) =>
            getViewportState().setProjectCamera(
              ev.target.checked ? e.id : null,
            )
          }
        />
        <span>Project frame</span>
      </label>
      {/* While calibrating, opacity lives on the wizard toolbar only. */}
      {!calibrating && e.posed && projecting ? (
        <label className="ss-prop-field">
          <span className="ss-prop-label">Opacity</span>
          <input
            type="range"
            className="ss-prop-range"
            min={0}
            max={100}
            step={1}
            value={Math.round(projectOpacity * 100)}
            onChange={(ev) =>
              getViewportState().setProjectOpacity(
                Number(ev.target.value) / 100,
              )
            }
          />
          <span className="ss-prop-value">{Math.round(projectOpacity * 100)}%</span>
        </label>
      ) : null}
      {!calibrating ? (
        <SaveBar
          dirty={dirty}
          busy={saveBusy}
          savedFlash={savedFlash}
          canPersist={Boolean(sensorId && authToken && sceneId)}
          onSave={() => void onSave()}
          disabledTitle="Camera is not linked to the server yet"
        />
      ) : null}
      <div className="ss-prop-actions">
        {!calibrating && (onCalibrate || calibrateHref)
          ? onCalibrate ? (
              <button
                type="button"
                className="ss-prop-calibrate"
                onClick={onCalibrate}
              >
                <i className="bi bi-crosshair" aria-hidden="true" />
                <span>Calibrate</span>
              </button>
            ) : (
              <a href={calibrateHref} className="ss-prop-calibrate">
                <i className="bi bi-crosshair" aria-hidden="true" />
                <span>Calibrate</span>
              </a>
            )
          : null}
        {!calibrating && sensorId && authToken ? (
          <button
            type="button"
            className="ss-prop-calibrate"
            disabled={!autoReady || autoBusy}
            title={autoTitle}
            onClick={() => void onAutoCalibrate()}
          >
            <i className="bi bi-lightning" aria-hidden="true" />
            <span>{autoBusy ? "Starting…" : "Auto Calibrate"}</span>
          </button>
        ) : null}
        {deleteUrl && authToken ? (
          <button
            type="button"
            className="ss-prop-danger"
            onClick={() => setConfirmDelete(true)}
          >
            <i className="bi bi-trash" aria-hidden="true" />
            <span>Delete</span>
          </button>
        ) : null}
      </div>
      <p className="ss-prop-hint">
        {calibrating
          ? "Use the toolbar to Auto Calibrate or Save point correspondences. Edit pose and optics here; they apply after Save."
          : "Save writes pose, FOV/fy, intrinsics, and distortion. FOV is a convenience view of fy; fx/fy drive solve and projection. Calibrate opens the point-pair split; Auto Calibrate needs the service and a live frame."}
      </p>
      <ConfirmDialog
        open={confirmDelete}
        title="Delete camera?"
        confirmLabel="Delete"
        danger
        busy={deleteBusy}
        onConfirm={() => void onDelete()}
        onCancel={() => {
          if (!deleteBusy) {
            setConfirmDelete(false);
          }
        }}
      >
        <p>
          Are you sure you want to delete <strong>{e.name}</strong>?
        </p>
        <p>This action cannot be undone.</p>
      </ConfirmDialog>
    </>
  );
}

function SensorProps({
  e,
  sensorId,
  sceneId,
  calibrateHref,
  authToken,
}: {
  e: SensorEntity;
  sensorId?: string;
  sceneId?: string;
  calibrateHref?: string;
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
          onChange={(ev) =>
            patch({ areaMode: ev.target.value as SensorAreaMode })
          }
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
      {e.areaMode === "poly" && calibrateHref ? (
        <a href={calibrateHref} className="ss-prop-calibrate">
          <i className="bi bi-pencil" aria-hidden="true" />
          <span>Edit polygon on map</span>
        </a>
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
            : "Draw polygon vertices on the map first, then Save name/area here."}
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

function EmptyProps({ sceneName }: { sceneName: string }) {
  const entities = useViewportStore((s) => s.entities);
  const counts = { region: 0, tripwire: 0, camera: 0, sensor: 0, mark: 0 };
  for (const e of Object.values(entities)) {
    if (e.type in counts) {
      counts[e.type as keyof typeof counts] += 1;
    }
  }
  return (
    <>
      <ReadField label="Scene" value={sceneName} />
      <ReadField label="Regions" value={String(counts.region)} />
      <ReadField label="Tripwires" value={String(counts.tripwire)} />
      <ReadField label="Cameras" value={String(counts.camera)} />
      <ReadField label="Sensors" value={String(counts.sensor)} />
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
  authToken = "",
  isSuperuser = false,
}: {
  sceneName: string;
  sceneId?: string;
  cameras?: {
    id: string;
    sensorId: string;
    name: string;
    calibrateHref: string;
    deleteUrl?: string | null;
  }[];
  sensors?: {
    id: string;
    sensorId: string;
    name: string;
    calibrateHref: string;
    deleteUrl?: string | null;
  }[];
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

  return (
    <div className="ss-props">
      {!e && <EmptyProps sceneName={sceneName} />}
      {e?.type === "region" && <RegionProps e={e} />}
      {e?.type === "tripwire" && <TripwireProps e={e} />}
      {e?.type === "camera" && (
        <CameraProps
          e={e}
          sensorId={camBoot?.sensorId}
          sceneId={sceneId}
          calibrateHref={camBoot?.calibrateHref}
          deleteUrl={isSuperuser ? camBoot?.deleteUrl : null}
          authToken={authToken}
          sceneHomeHref={sceneId ? `/${sceneId}/` : undefined}
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
          calibrateHref={senBoot?.calibrateHref}
          authToken={authToken}
        />
      )}
      {e?.type === "mark" && <MarkProps e={e} />}
      {e?.type === "child" && (
        <>
          <NameRow entity={e} />
          <p className="ss-prop-hint">Linked child scene.</p>
        </>
      )}
    </div>
  );
}
