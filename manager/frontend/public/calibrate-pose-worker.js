// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Classic worker: load OpenCV via importScripts and run solvePnP off the
 * main thread. Returns raw R (3x3 row-major) + t; main thread finishes the
 * OpenCV→OpenGL pose conversion with three.js.
 */
/* global importScripts, cv */

let cvReady = null;

function loadOpenCv() {
  if (cvReady) {
    return cvReady;
  }
  cvReady = new Promise((resolve, reject) => {
    try {
      if (typeof cv === "undefined" || !cv.Mat) {
        importScripts("/static/assets/opencv.js");
      }
    } catch (err) {
      reject(err);
      return;
    }
    const finish = (mod) => {
      if (mod.getBuildInformation && mod.getBuildInformation()) {
        resolve(mod);
        return;
      }
      mod.onRuntimeInitialized = () => resolve(mod);
    };
    if (typeof cv !== "undefined" && cv.Mat) {
      finish(cv);
      return;
    }
    let tries = 0;
    const poll = setInterval(() => {
      tries += 1;
      if (typeof cv !== "undefined" && cv.Mat) {
        clearInterval(poll);
        finish(cv);
      } else if (tries > 600) {
        clearInterval(poll);
        reject(new Error("OpenCV.js worker load timed out"));
      }
    }, 50);
  });
  return cvReady;
}

function solveRt(mod, pairs, optics) {
  const imagePts = pairs.map((p) => p.cam);
  const objectPts = pairs.map((p) => p.map);
  const imagePointsMat = mod.matFromArray(
    imagePts.length,
    2,
    mod.CV_64F,
    imagePts.flat(),
  );
  const objectPointsMat = mod.matFromArray(
    objectPts.length,
    3,
    mod.CV_64F,
    objectPts.flat(),
  );
  const cameraMatrixMat = mod.matFromArray(3, 3, mod.CV_64F, [
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
  const distCoeffsMat = mod.matFromArray(1, 5, mod.CV_64F, [
    optics.k1,
    optics.k2,
    optics.p1,
    optics.p2,
    optics.k3,
  ]);
  const rvec = new mod.Mat();
  const tvec = new mod.Mat();
  const R = new mod.Mat();
  try {
    const ok = mod.solvePnP(
      objectPointsMat,
      imagePointsMat,
      cameraMatrixMat,
      distCoeffsMat,
      rvec,
      tvec,
      false,
      mod.SOLVEPNP_ITERATIVE,
    );
    if (!ok) {
      return null;
    }
    mod.Rodrigues(rvec, R);
    return {
      R: Array.from(R.data64F),
      t: Array.from(tvec.data64F),
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

self.onmessage = (ev) => {
  const msg = ev.data || {};
  if (msg.type === "preload") {
    loadOpenCv()
      .then(() => {
        self.postMessage({ type: "ready" });
      })
      .catch((err) => {
        self.postMessage({
          type: "error",
          error: String(err && err.message ? err.message : err),
        });
      });
    return;
  }
  if (msg.type !== "solve") {
    return;
  }
  const requestId = msg.requestId;
  loadOpenCv()
    .then((mod) => {
      const rt = solveRt(mod, msg.pairs || [], msg.optics || {});
      if (!rt) {
        self.postMessage({ type: "result", requestId, ok: false });
        return;
      }
      self.postMessage({
        type: "result",
        requestId,
        ok: true,
        R: rt.R,
        t: rt.t,
      });
    })
    .catch((err) => {
      self.postMessage({
        type: "result",
        requestId,
        ok: false,
        error: String(err && err.message ? err.message : err),
      });
    });
};
