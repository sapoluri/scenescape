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

let cvReady: Promise<CvModule> | null = null;

/** Prefetch OpenCV.js while the user is still picking points (idle). */
export function preloadOpenCv(): void {
  void loadOpenCv().catch(() => {
    /* first solve will surface the error */
  });
}

function loadOpenCv(): Promise<CvModule> {
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

export type SolvedPose = {
  /** REST-style translation (meters, Z-up). */
  position: [number, number, number];
  /** REST-style XYZ euler degrees (before Rx(π) frustum remapping). */
  rotation: [number, number, number];
  fov: number;
};

/**
 * Mirror legacy cameracalibrate.js solvePnP + OpenCV→OpenGL flip + invert.
 * Returns pose in the same convention as REST `translation` / `rotation`.
 */
function yieldToUi(): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, 0);
  });
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
    // OpenCV → OpenGL: negate rows 2 and 3, then invert for camera pose.
    const T = new Matrix4().set(
      R.data64F[0],
      R.data64F[1],
      R.data64F[2],
      tvec.data64F[0],
      -R.data64F[3],
      -R.data64F[4],
      -R.data64F[5],
      -tvec.data64F[1],
      -R.data64F[6],
      -R.data64F[7],
      -R.data64F[8],
      -tvec.data64F[2],
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

/**
 * Mirror legacy cameracalibrate.js solvePnP + OpenCV→OpenGL flip + invert.
 * Returns pose in the same convention as REST `translation` / `rotation`.
 */
export async function solveCameraPose(
  pairs: CalibratePair[],
  optics: CameraOptics,
): Promise<SolvedPose | null> {
  if (pairs.length < 4) {
    return null;
  }
  // Let the UI paint the 4th marker before pulling in OpenCV / solving.
  await yieldToUi();
  const cv = await loadOpenCv();
  await yieldToUi();
  return solvePnPOnce(cv, pairs, optics);
}
