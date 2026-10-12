// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Classic worker: load OpenCV via importScripts and run solvePnP off the
 * main thread. Returns raw R (3x3) + t; main thread finishes the
 * OpenCV→OpenGL pose conversion with three.js.
 *
 * Never run this OpenCV path on the main thread — import + solvePnP freeze UI.
 */
/* global importScripts, cv */

let cvReady = false;

function matValues(mat) {
  const n = mat.rows * mat.cols;
  const out = new Array(n);
  if (mat.data64F && mat.data64F.length >= n) {
    for (let i = 0; i < n; i++) {
      out[i] = mat.data64F[i];
    }
    return out;
  }
  if (mat.data32F && mat.data32F.length >= n) {
    for (let i = 0; i < n; i++) {
      out[i] = mat.data32F[i];
    }
    return out;
  }
  throw new Error("OpenCV Mat has no readable float data");
}

function solveRt(pairs, optics) {
  if (!pairs.length || !optics || !(optics.fx > 0) || !(optics.fy > 0)) {
    throw new Error("solvePnP missing pairs or optics");
  }
  const mod = cv;
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
    optics.k1 || 0,
    optics.k2 || 0,
    optics.p1 || 0,
    optics.p2 || 0,
    optics.k3 || 0,
  ]);
  const rvec = new mod.Mat();
  const tvec = new mod.Mat();
  const R = new mod.Mat();
  try {
    const okPnP = mod.solvePnP(
      objectPointsMat,
      imagePointsMat,
      cameraMatrixMat,
      distCoeffsMat,
      rvec,
      tvec,
      false,
      mod.SOLVEPNP_ITERATIVE,
    );
    if (!okPnP) {
      return null;
    }
    mod.Rodrigues(rvec, R);
    return {
      R: matValues(R),
      t: matValues(tvec),
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

self.addEventListener("message", (ev) => {
  const msg = ev.data || {};
  if (msg.type === "ping") {
    self.postMessage({ type: "pong", t: msg.t });
    return;
  }
  if (msg.type === "preload") {
    try {
      self.postMessage({ type: "progress", stage: "importScripts-start" });
      if (typeof cv === "undefined") {
        importScripts(
          new URL("/static/assets/opencv.js", self.location.origin).toString(),
        );
      }
      self.postMessage({ type: "progress", stage: "importScripts-done" });
    } catch (err) {
      self.postMessage({
        type: "error",
        error: String(err && err.message ? err.message : err),
      });
      return;
    }
    const prev = cv.onRuntimeInitialized;
    cv.onRuntimeInitialized = () => {
      if (typeof prev === "function") {
        prev();
      }
      setTimeout(() => {
        cvReady = true;
        self.postMessage({ type: "progress", stage: "wasm-ready" });
        self.postMessage({ type: "ready" });
      }, 0);
    };
    return;
  }
  if (msg.type !== "solve") {
    return;
  }
  const requestId = msg.requestId;
  self.postMessage({ type: "progress", stage: "solve-start", requestId });
  if (!cvReady || typeof cv === "undefined" || !cv.solvePnP) {
    self.postMessage({
      type: "result",
      requestId,
      ok: false,
      error: "OpenCV not ready",
    });
    return;
  }
  try {
    self.postMessage({
      type: "progress",
      stage: "solvePnP-enter",
      requestId,
    });
    const rt = solveRt(msg.pairs || [], msg.optics || {});
    self.postMessage({
      type: "progress",
      stage: "solvePnP-exit",
      requestId,
    });
    if (!rt) {
      self.postMessage({
        type: "result",
        requestId,
        ok: false,
        error: "solvePnP returned false",
      });
      return;
    }
    self.postMessage({
      type: "result",
      requestId,
      ok: true,
      R: rt.R,
      t: rt.t,
    });
  } catch (err) {
    self.postMessage({
      type: "result",
      requestId,
      ok: false,
      error: String(err && err.message ? err.message : err),
    });
  }
});
