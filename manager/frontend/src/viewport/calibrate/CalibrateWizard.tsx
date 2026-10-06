// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";
import { api, type RestError } from "../../lib/rest";
import { useAppToast } from "../../components/ToastProvider";
import { restPoseToRigEuler } from "../entities";
import { getViewportState } from "../store";
import { useCalibrateStore } from "./calibrateStore";
import { pairsToTransforms, TRANSFORM_TYPE_POINT } from "./types";
import "./CalibrateWizard.css";

type Props = {
  sceneId: string;
  authToken: string;
  cameras: { id: string; sensorId: string; name: string }[];
};

/**
 * Tool-header wizard for Phase 2.3: camera context, pair count, undo/reset,
 * dirty-gated Save, and Exit.
 */
export function CalibrateWizard({
  sceneId,
  authToken,
  cameras,
}: Props): React.JSX.Element {
  const toast = useAppToast();
  const cameraId = useCalibrateStore((s) => s.cameraId);
  const sensorId = useCalibrateStore((s) => s.sensorId);
  const cameraName = useCalibrateStore((s) => s.cameraName);
  const pairs = useCalibrateStore((s) => s.pairs);
  const pending = useCalibrateStore((s) => s.pending);
  const dirty = useCalibrateStore((s) => s.dirty);
  const optics = useCalibrateStore((s) => s.optics);
  const priorPose = useCalibrateStore((s) => s.priorPose);
  const exit = useCalibrateStore((s) => s.exit);
  const undoPair = useCalibrateStore((s) => s.undoPair);
  const resetPairs = useCalibrateStore((s) => s.resetPairs);
  const markClean = useCalibrateStore((s) => s.markClean);
  const enter = useCalibrateStore((s) => s.enter);
  const [busy, setBusy] = useState(false);
  const [savedFlash, setSavedFlash] = useState(false);

  const canSave = pairs.length >= 4 && dirty && !busy && Boolean(optics);

  const restorePrior = () => {
    if (!cameraId || !priorPose) {
      return;
    }
    getViewportState().updateEntity(cameraId, {
      position: priorPose.position,
      rotation: priorPose.rotation,
      fov: priorPose.fov,
    });
  };

  const onExit = () => {
    if (dirty && !window.confirm("Discard unsaved calibration changes?")) {
      return;
    }
    restorePrior();
    exit();
  };

  const onSave = async () => {
    if (!canSave || !sensorId || !optics) {
      return;
    }
    setBusy(true);
    try {
      await api.updateCamera(authToken, sensorId, {
        name: optics.name || cameraName,
        sensor_id: sensorId,
        scene: sceneId,
        intrinsics: {
          fx: optics.fx,
          fy: optics.fy,
          cx: optics.cx,
          cy: optics.cy,
        },
        distortion: {
          k1: optics.k1,
          k2: optics.k2,
          p1: optics.p1,
          p2: optics.p2,
          k3: optics.k3,
        },
        resolution: { width: optics.width, height: optics.height },
        transform_type: TRANSFORM_TYPE_POINT,
        transforms: pairsToTransforms(pairs),
      });
      toast.show("Camera calibration saved", "ok");
      markClean();
      setSavedFlash(true);
      window.setTimeout(() => setSavedFlash(false), 1500);
      // Refresh pose from server (authoritative).
      try {
        const row = (await api.getCamera(authToken, sensorId)) ?? {};
        const t = row.translation as number[] | undefined;
        const r = row.rotation as number[] | undefined;
        if (cameraId && Array.isArray(t) && t.length >= 3) {
          const restRot: [number, number, number] =
            Array.isArray(r) && r.length >= 3
              ? [Number(r[0]), Number(r[1]), Number(r[2])]
              : [0, 0, 0];
          getViewportState().updateEntity(cameraId, {
            position: [Number(t[0]), Number(t[1]), Number(t[2])],
            rotation: restPoseToRigEuler(restRot),
          });
        }
      } catch {
        /* keep live preview pose */
      }
    } catch (err) {
      toast.show((err as RestError).message || "Save failed", "bad");
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === "Escape") {
        onExit();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- exit once
  }, []);

  const stepLabel =
    pairs.length < 4
      ? `Step 2 · Pick points (${pairs.length}/4+)`
      : "Step 3–4 · Frustum updated · Save when ready";

  return (
    <div className="ss-cal-wizard" role="toolbar" aria-label="Calibration wizard">
      <span className="ss-cal-wizard-badge">Calibrate</span>
      <label className="ss-cal-wizard-cam">
        <span className="ss-viewport-bar-label">Camera</span>
        <select
          value={cameraId || ""}
          onChange={(ev) => {
            const id = ev.target.value;
            const cam = cameras.find((c) => c.id === id);
            if (!cam) {
              return;
            }
            if (dirty && !window.confirm("Switch camera and discard points?")) {
              return;
            }
            restorePrior();
            enter({
              cameraId: cam.id,
              sensorId: cam.sensorId,
              cameraName: cam.name,
            });
          }}
          aria-label="Select camera to calibrate"
        >
          {cameras.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </select>
      </label>
      <span className="ss-cal-wizard-step" title={stepLabel}>
        {stepLabel}
        {pending === "scene" ? " · waiting for 3D click" : ""}
      </span>
      <span className="ss-viewport-bar-sep" aria-hidden="true" />
      <button
        type="button"
        className="ss-tool-btn"
        disabled={pairs.length === 0 || busy}
        onClick={undoPair}
        title="Undo last pair"
      >
        Undo
      </button>
      <button
        type="button"
        className="ss-tool-btn"
        disabled={pairs.length === 0 || busy}
        onClick={resetPairs}
        title="Clear all pairs"
      >
        Reset
      </button>
      <button
        type="button"
        className={`ss-tool-btn ss-tool-save${dirty && pairs.length >= 4 ? " is-dirty" : ""}`}
        disabled={!canSave}
        onClick={onSave}
        title={
          pairs.length < 4
            ? "Need at least 4 point pairs"
            : "Save calibration (PUT camera)"
        }
      >
        <i className="bi bi-check-lg" aria-hidden="true" />
        <span className="ss-tool-label">
          {busy ? "Saving…" : savedFlash ? "Saved" : "Save"}
        </span>
      </button>
      <button
        type="button"
        className="ss-tool-btn ss-cal-wizard-exit"
        onClick={onExit}
        title="Exit calibrate (Esc)"
      >
        Exit
      </button>
    </div>
  );
}
