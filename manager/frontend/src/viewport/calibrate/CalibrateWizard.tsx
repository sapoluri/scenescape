// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";
import { useAppToast } from "../../components/ToastProvider";
import {
  frameToBase64,
  pointsFromAutocalibResult,
  resolveAutocalibAvailability,
  startCameraAutocalibration,
  waitForCameraAutocalibration,
} from "../../lib/autocalibration";
import { api, type RestError } from "../../lib/rest";
import { restPoseToRigEuler } from "../entities";
import { getViewportState, useViewportStore } from "../store";
import { getCalibrateState, useCalibrateStore } from "./calibrateStore";
import {
  pairsToTransforms,
  TRANSFORM_TYPE_POINT,
  zipCalibratePairs,
} from "./types";
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
  const camPoints = useCalibrateStore((s) => s.camPoints);
  const mapPoints = useCalibrateStore((s) => s.mapPoints);
  const dirty = useCalibrateStore((s) => s.dirty);
  const optics = useCalibrateStore((s) => s.optics);
  const priorPose = useCalibrateStore((s) => s.priorPose);
  const exit = useCalibrateStore((s) => s.exit);
  const undoPoint = useCalibrateStore((s) => s.undoPoint);
  const resetPoints = useCalibrateStore((s) => s.resetPoints);
  const markClean = useCalibrateStore((s) => s.markClean);
  const enter = useCalibrateStore((s) => s.enter);
  const pairCount =
    camPoints.length === mapPoints.length ? camPoints.length : 0;
  const hasPoints = camPoints.length > 0 || mapPoints.length > 0;
  const [busy, setBusy] = useState(false);
  const [savedFlash, setSavedFlash] = useState(false);
  const [autoBusy, setAutoBusy] = useState(false);
  const [autoReady, setAutoReady] = useState(false);
  const [autoTitle, setAutoTitle] = useState(
    "Checking auto-calibration service…",
  );
  const feedImageUrl = useCalibrateStore((s) => s.feedImageUrl);
  const updatingPose = useCalibrateStore((s) => s.updatingPose);
  const projectOpacity = useViewportStore((s) => s.projectOpacity);

  const canSave = pairCount >= 4 && dirty && !busy && Boolean(optics);

  useEffect(() => {
    let cancelled = false;
    (async () => {
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
  }, [sceneId, sensorId, authToken]);

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

  const onAutoCalibrate = async () => {
    if (!sensorId || !optics || autoBusy) {
      return;
    }
    if (!feedImageUrl) {
      toast.show("Wait for a calibration frame first", "bad");
      return;
    }
    setAutoBusy(true);
    try {
      const image = await frameToBase64(feedImageUrl);
      const started = await startCameraAutocalibration(sensorId, image, [
        [optics.fx, 0, optics.cx],
        [0, optics.fy, optics.cy],
        [0, 0, 1],
      ]);
      if (started.status === "error") {
        toast.show(started.message || "Auto calibrate failed", "bad");
        return;
      }
      toast.show("Detecting AprilTags…", "ok");
      const result = await waitForCameraAutocalibration(sensorId);
      if (result.status !== "success") {
        toast.show(
          result.message ||
            "Auto calibrate failed. Check that tags are visible in the frame.",
          "bad",
        );
        return;
      }
      const pts = pointsFromAutocalibResult(result);
      if (!pts) {
        toast.show("Auto calibrate returned no point correspondences", "bad");
        return;
      }
      getCalibrateState().replacePoints(pts.camPoints, pts.mapPoints);
      toast.show(
        `Auto calibrate found ${pts.camPoints.length} tag correspondences`,
        "ok",
      );
    } catch (err) {
      toast.show((err as Error).message || "Auto calibrate failed", "bad");
    } finally {
      setAutoBusy(false);
    }
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
        transforms: pairsToTransforms(
          zipCalibratePairs(camPoints, mapPoints),
        ),
      });
      toast.show("Camera calibration saved", "ok");
      markClean();
      setSavedFlash(true);
      window.setTimeout(() => setSavedFlash(false), 1500);
      // Refresh pose from server (authoritative). Skip identity — that is
      // the empty-transforms fallback (looks like Rx=180 after REST→rig).
      try {
        const row = (await api.getCamera(authToken, sensorId)) ?? {};
        const t = row.translation as number[] | undefined;
        const r = row.rotation as number[] | undefined;
        if (cameraId && Array.isArray(t) && t.length >= 3) {
          const restRot: [number, number, number] =
            Array.isArray(r) && r.length >= 3
              ? [Number(r[0]), Number(r[1]), Number(r[2])]
              : [0, 0, 0];
          const identity =
            Math.hypot(Number(t[0]), Number(t[1]), Number(t[2])) < 1e-9 &&
            Math.hypot(restRot[0], restRot[1], restRot[2]) < 1e-9;
          if (!identity) {
            getViewportState().updateEntity(cameraId, {
              position: [Number(t[0]), Number(t[1]), Number(t[2])],
              rotation: restPoseToRigEuler(restRot),
              restRotation: restRot,
              posed: true,
            });
          }
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

  const stepLabel = updatingPose
    ? pairCount >= 6
      ? "Updating pose + fx/fy…"
      : "Updating pose…"
    : pairCount < 4
      ? camPoints.length !== mapPoints.length
        ? `Step 2 · Pick points (cam ${camPoints.length} / map ${mapPoints.length})`
        : `Step 2 · Pick points (${pairCount}/4+)`
      : pairCount >= 6
        ? "Step 3–4 · Pose + fx/fy updated · Save when ready"
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
      </span>
      <label
        className="ss-cal-wizard-opacity"
        title="Map projection opacity"
      >
        <span className="ss-viewport-bar-label">Overlay</span>
        <input
          type="range"
          min={0}
          max={100}
          step={1}
          value={Math.round(projectOpacity * 100)}
          onChange={(ev) =>
            getViewportState().setProjectOpacity(
              Number(ev.target.value) / 100,
            )
          }
          aria-label="Projection opacity"
        />
        <span className="ss-cal-wizard-opacity-val">
          {Math.round(projectOpacity * 100)}%
        </span>
      </label>
      <span className="ss-viewport-bar-sep" aria-hidden="true" />
      <button
        type="button"
        className="ss-tool-btn"
        disabled={!hasPoints || busy}
        onClick={undoPoint}
        title="Undo last point"
      >
        Undo
      </button>
      <button
        type="button"
        className="ss-tool-btn"
        disabled={!hasPoints || busy}
        onClick={resetPoints}
        title="Clear all points"
      >
        Reset
      </button>
      <button
        type="button"
        className="ss-tool-btn"
        id="auto-autocalibration"
        disabled={!autoReady || autoBusy || !feedImageUrl || busy}
        title={autoBusy ? "Waiting for AprilTag calibration…" : autoTitle}
        onClick={() => void onAutoCalibrate()}
      >
        <i className="bi bi-lightning" aria-hidden="true" />
        <span className="ss-tool-label">
          {autoBusy ? "Calibrating…" : "Auto Calibrate"}
        </span>
      </button>
      <button
        type="button"
        className={`ss-tool-btn ss-tool-save${dirty && pairCount >= 4 ? " is-dirty" : ""}`}
        disabled={!canSave}
        onClick={onSave}
        title={
          camPoints.length !== mapPoints.length
            ? "Need an equal number of points in each view"
            : pairCount < 4
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
