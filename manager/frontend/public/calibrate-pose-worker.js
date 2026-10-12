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

let cvReady = null;

function loadOpenCv() {
  if (cvReady) {
    return cvReady;
  }
  cvReady = new Promise((resolve, reject) => {
    let settled = false;
    const done = (mod) => {
      if (settled) {
        return;
      }
      settled = true;
      resolve(mod);
    };
    const fail = (err) => {
      if (settled) {
        return;
      }
      settled = true;
      reject(err);
    };
    try {
      self.postMessage({ type: "progress", stage: "importScripts-start" });
      if (typeof cv === "undefined" || !cv.Mat) {
        importScripts("/static/assets/opencv.js");
      }
      self.postMessage({ type: "progress", stage: "importScripts-done" });
    } catch (err) {
      fail(err);
      return;
    }
    const smoke = (mod) => {
      const m = mod.matFromArray(1, 1, mod.CV_64F, [1]);
      m.delete();
      if (typeof mod.solvePnP !== "function") {
        throw new Error("OpenCV solvePnP missing after init");
      }
    };
    const finish = (mod) => {
      const accept = () => {
        try {
          smoke(mod);
          self.postMessage({ type: "progress", stage: "wasm-ready" });
          done(mod);
        } catch (err) {
          fail(err);
        }
      };
      if (mod.solvePnP && mod.getBuildInformation) {
        try {
          if (mod.getBuildInformation()) {
            accept();
            return;
          }
        } catch {
          /* wait for runtime init */
        }
      }
      const prev = mod.onRuntimeInitialized;
      mod.onRuntimeInitialized = () => {
        if (typeof prev === "function") {
          prev();
        }
        accept();
      };
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
        fail(new Error("OpenCV.js worker load timed out"));
      }
    }, 50);
  });
  return cvReady;
}

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

function solveRt(mod, pairs, optics) {
  if (!pairs.length || !optics || !(optics.fx > 0) || !(optics.fy > 0)) {
    throw new Error("solvePnP missing pairs or optics");
  }
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
      try {
        const rt = solveRt(mod, msg.pairs || [], msg.optics || {});
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
