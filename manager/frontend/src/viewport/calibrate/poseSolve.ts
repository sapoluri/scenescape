// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Euler, Matrix4, Quaternion, Vector3 } from "three";
import type { CalibratePair, CameraOptics } from "./types";

type CvMat = {
  data64F: Float64Array;
  delete: () => void;
};

type CvModule = {
  Mat: new () => CvMat;
  matFromArray: (
    rows: number,
    cols: number,
    type: number,
    array: number[],
  ) => CvMat;
  solvePnP: (
    objectPoints: CvMat,
    imagePoints: CvMat,
    cameraMatrix: CvMat,
    distCoeffs: CvMat,
    rvec: CvMat,
    tvec: CvMat,
    useExtrinsicGuess: boolean,
    flags: number,
  ) => boolean;
  Rodrigues: (src: CvMat, dst: CvMat) => void;
  CV_64F: number;
  SOLVEPNP_ITERATIVE: number;
  getBuildInformation?: () => string;
  onRuntimeInitialized?: () => void;
};

declare global {
  interface Window {
    cv?: CvModule;
  }
}

export type SolvedPose = {
  /** REST-style translation (meters, Z-up). */
  position: [number, number, number];
  /** REST-style XYZ euler degrees (before Rx(π) frustum remapping). */
  rotation: [number, number, number];
  fov: number;
};

type WorkerResultMsg = {
  type: "result";
  requestId: number;
  ok: boolean;
  R?: number[];
  t?: number[];
  error?: string;
};

type WorkerOutMsg =
  | { type: "ready" }
  | { type: "progress"; stage?: string }
  | { type: "error"; error: string }
  | WorkerResultMsg;

type Pending = {
  resolve: (pose: SolvedPose) => void;
  reject: (err: Error) => void;
  optics: CameraOptics;
  timer: number;
};

let cvReady: Promise<CvModule> | null = null;
let worker: Worker | null = null;
let workerFailed = false;
let workerBoot: Promise<Worker> | null = null;
let nextRequestId = 0;
let latestRequestId = 0;
let loggedWorkerFallback = false;
const pending = new Map<number, Pending>();

/**
 * Worker may take a while to importScripts the ~10MB OpenCV build; that is
 * fine because it does not freeze the UI. Progress messages reset this timer.
 * Never fall back to main-thread solvePnP — that freezes the page.
 */
const WORKER_READY_TIMEOUT_MS = 90000;
const SOLVE_TIMEOUT_MS = 15000;

function workerScriptUrl(): string {
  // BASE_URL is a path ("/static/ui/"); URL() needs an absolute base.
  const base = import.meta.env.BASE_URL || "/static/ui/";
  const path = `${base.endsWith("/") ? base : `${base}/`}calibrate-pose-worker.js`;
  return new URL(`${path}?v=5`, window.location.origin).toString();
}

function attachWorkerDispatcher(w: Worker): void {
  w.onmessage = (ev: MessageEvent<WorkerOutMsg>) => {
    const msg = ev.data;
    if (!msg || typeof msg !== "object") {
      return;
    }
    if (msg.type === "ready" || msg.type === "progress") {
      return;
    }
    if (msg.type === "error") {
      const err = new Error(msg.error || "OpenCV worker failed");
      for (const [id, p] of pending) {
        window.clearTimeout(p.timer);
        p.reject(err);
        pending.delete(id);
      }
      return;
    }
    if (msg.type !== "result") {
      return;
    }
    const p = pending.get(msg.requestId);
    if (!p) {
      return;
    }
    pending.delete(msg.requestId);
    window.clearTimeout(p.timer);
    if (!msg.ok || !msg.R || !msg.t) {
      p.reject(new Error(msg.error || "OpenCV worker solvePnP returned null"));
      return;
    }
    const pose = poseFromRt(msg.R, msg.t, p.optics);
    if (!pose) {
      p.reject(new Error("OpenCV worker pose conversion failed"));
      return;
    }
    p.resolve(pose);
  };
}

function ensureWorker(): Promise<Worker> {
  if (workerFailed) {
    return Promise.reject(new Error("OpenCV worker unavailable"));
  }
  if (worker) {
    return Promise.resolve(worker);
  }
  if (workerBoot) {
    return workerBoot;
  }
  workerBoot = new Promise((resolve, reject) => {
    let settled = false;
    let readyTimer = 0;
    let w: Worker;
    try {
      w = new Worker(workerScriptUrl());
    } catch (err) {
      workerFailed = true;
      workerBoot = null;
      reject(err);
      return;
    }
    const fail = (err: Error) => {
      if (settled) {
        return;
      }
      settled = true;
      window.clearTimeout(readyTimer);
      workerFailed = true;
      worker = null;
      workerBoot = null;
      try {
        w.terminate();
      } catch {
        /* ignore */
      }
      reject(err);
    };
    const armReadyTimer = () => {
      window.clearTimeout(readyTimer);
      readyTimer = window.setTimeout(() => {
        fail(new Error("OpenCV worker ready timed out"));
      }, WORKER_READY_TIMEOUT_MS);
    };
    const ok = () => {
      if (settled) {
        return;
      }
      settled = true;
      window.clearTimeout(readyTimer);
      attachWorkerDispatcher(w);
      worker = w;
      resolve(w);
    };
    w.onmessage = (ev: MessageEvent<WorkerOutMsg>) => {
      const msg = ev.data;
      // importScripts can take tens of seconds; progress keeps us alive.
      if (msg?.type === "progress") {
        armReadyTimer();
        return;
      }
      if (msg?.type === "ready") {
        ok();
        return;
      }
      if (msg?.type === "error") {
        fail(new Error(msg.error || "OpenCV worker failed"));
      }
    };
    w.onerror = () => {
      fail(new Error("OpenCV worker script error"));
    };
    armReadyTimer();
    w.postMessage({ type: "preload" });
  });
  return workerBoot;
}

/** Prefetch OpenCV in the worker only (never on the main thread). */
export function preloadOpenCv(): void {
  void ensureWorker().catch(() => {
    /* solve will surface the error */
  });
}

function loadOpenCvMain(): Promise<CvModule> {
  if (cvReady) {
    return cvReady;
  }
  cvReady = new Promise((resolve, reject) => {
    const finish = (cv: CvModule) => {
      if (cv.getBuildInformation?.()) {
        resolve(cv);
        return;
      }
      cv.onRuntimeInitialized = () => resolve(cv);
    };
    if (window.cv?.Mat) {
      finish(window.cv);
      return;
    }
    const existing = document.querySelector<HTMLScriptElement>(
      'script[data-ss-opencv="1"]',
    );
    if (existing) {
      const poll = window.setInterval(() => {
        if (window.cv?.Mat) {
          window.clearInterval(poll);
          finish(window.cv);
        }
      }, 50);
      window.setTimeout(() => {
        window.clearInterval(poll);
        reject(new Error("OpenCV.js load timed out"));
      }, 30000);
      return;
    }
    const script = document.createElement("script");
    script.src = "/static/assets/opencv.js";
    script.async = true;
    script.dataset.ssOpencv = "1";
    script.onload = () => {
      if (!window.cv) {
        reject(new Error("OpenCV.js loaded without cv global"));
        return;
      }
      finish(window.cv);
    };
    script.onerror = () => reject(new Error("Failed to load OpenCV.js"));
    document.head.appendChild(script);
  });
  return cvReady;
}

/**
 * OpenCV→OpenGL flip + invert. Same convention as REST translation/rotation.
 */
function poseFromRt(
  R: ArrayLike<number>,
  t: ArrayLike<number>,
  optics: CameraOptics,
): SolvedPose | null {
  if (R.length < 9 || t.length < 3) {
    return null;
  }
  const T = new Matrix4().set(
    R[0],
    R[1],
    R[2],
    t[0],
    -R[3],
    -R[4],
    -R[5],
    -t[1],
    -R[6],
    -R[7],
    -R[8],
    -t[2],
    0,
    0,
    0,
    1,
  );
  T.invert();
  const position = new Vector3().setFromMatrixPosition(T);
  const q = new Quaternion().setFromRotationMatrix(T);
  const e = new Euler().setFromQuaternion(q, "XYZ");
  const r2d = (r: number) => (r * 180) / Math.PI;
  if (
    !Number.isFinite(position.x) ||
    !Number.isFinite(position.y) ||
    !Number.isFinite(position.z)
  ) {
    return null;
  }
  const fov =
    optics.fy > 0
      ? r2d(2 * Math.atan(optics.height / (2 * optics.fy)))
      : 60;
  return {
    position: [position.x, position.y, position.z],
    rotation: [r2d(e.x), r2d(e.y), r2d(e.z)],
    fov,
  };
}

function solvePnPOnce(
  cv: CvModule,
  pairs: CalibratePair[],
  optics: CameraOptics,
): SolvedPose | null {
  const imagePts = pairs.map((p) => p.cam);
  const objectPts = pairs.map((p) => p.map);
  const imagePointsMat = cv.matFromArray(
    imagePts.length,
    2,
    cv.CV_64F,
    imagePts.flat(),
  );
  const objectPointsMat = cv.matFromArray(
    objectPts.length,
    3,
    cv.CV_64F,
    objectPts.flat(),
  );
  const cameraMatrixMat = cv.matFromArray(3, 3, cv.CV_64F, [
    optics.fx,
    0,
    optics.cx,
    0,
    optics.fy,
    optics.cy,
    0,
    0,
    1,
  ]);
  const distCoeffsMat = cv.matFromArray(1, 5, cv.CV_64F, [
    optics.k1,
    optics.k2,
    optics.p1,
    optics.p2,
    optics.k3,
  ]);
  const rvec = new cv.Mat();
  const tvec = new cv.Mat();
  const R = new cv.Mat();
  try {
    // Always ITERATIVE. SQPNP in opencv.js can freeze the main thread on
    // near-planar floor correspondences (the calibrate default).
    const ok = cv.solvePnP(
      objectPointsMat,
      imagePointsMat,
      cameraMatrixMat,
      distCoeffsMat,
      rvec,
      tvec,
      false,
      cv.SOLVEPNP_ITERATIVE,
    );
    if (!ok) {
      return null;
    }
    cv.Rodrigues(rvec, R);
    return poseFromRt(R.data64F, tvec.data64F, optics);
  } finally {
    imagePointsMat.delete();
    objectPointsMat.delete();
    cameraMatrixMat.delete();
    distCoeffsMat.delete();
    rvec.delete();
    tvec.delete();
    R.delete();
  }
}

function solveInWorker(
  pairs: CalibratePair[],
  optics: CameraOptics,
  requestId: number,
): Promise<SolvedPose> {
  return ensureWorker().then(
    (w) =>
      new Promise((resolve, reject) => {
        const timer = window.setTimeout(() => {
          pending.delete(requestId);
          reject(new Error("OpenCV worker solve timed out"));
        }, SOLVE_TIMEOUT_MS);
        pending.set(requestId, { resolve, reject, optics, timer });
        try {
          w.postMessage({
            type: "solve",
            requestId,
            pairs,
            optics,
          });
        } catch (err) {
          window.clearTimeout(timer);
          pending.delete(requestId);
          reject(err instanceof Error ? err : new Error(String(err)));
        }
      }),
  );
}

function yieldToUi(): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, 0);
  });
}

async function solveOnMainThread(
  pairs: CalibratePair[],
  optics: CameraOptics,
): Promise<SolvedPose | null> {
  await yieldToUi();
  const cv = await loadOpenCvMain();
  await yieldToUi();
  return solvePnPOnce(cv, pairs, optics);
}

/**
 * Mirror legacy cameracalibrate.js solvePnP + OpenCV→OpenGL flip + invert.
 * Always solves in the Web Worker. Waiting for the worker is async and does
 * not freeze the UI; main-thread OpenCV is intentionally not used.
 * Stale results (superseded requestId) are dropped.
 */
export async function solveCameraPose(
  pairs: CalibratePair[],
  optics: CameraOptics,
): Promise<SolvedPose | null> {
  if (pairs.length < 4) {
    return null;
  }
  if (!(optics.fx > 0) || !(optics.fy > 0)) {
    console.warn("Calibrate solve skipped: optics not ready");
    return null;
  }
  const requestId = ++nextRequestId;
  latestRequestId = requestId;
  try {
    await ensureWorker();
    if (requestId !== latestRequestId) {
      return null;
    }
    const pose = await solveInWorker(pairs, optics, requestId);
    if (requestId !== latestRequestId) {
      return null;
    }
    return pose;
  } catch (err) {
    if (requestId !== latestRequestId) {
      return null;
    }
    if (!loggedWorkerFallback) {
      loggedWorkerFallback = true;
      console.warn("Calibrate OpenCV worker solve failed", err);
    }
    return null;
  }
}
