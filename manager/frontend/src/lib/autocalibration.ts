// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { api } from "./rest";

const AUTOCALIB_PROXY_BASE = "/api/v1/autocalibration";

export type AutocalibStatus = {
  status?: string;
  message?: string;
  strategy?: string;
};

export type AutocalibAvailability = {
  /** True only when this scene supports in-viewport auto calibrate (AprilTag). */
  ready: boolean;
  title: string;
  /** Scene `camera_calibration` value, or null if unknown. */
  strategy: string | null;
};

/**
 * Gate the Auto Calibrate control by scene strategy + service health.
 * Manual scenes must stay disabled; Markerless is 3D-only; AprilTag needs
 * a healthy autocalibration service (legacy calib_strategy check).
 */
export async function resolveAutocalibAvailability(
  sceneId: string,
  authToken?: string | null,
): Promise<AutocalibAvailability> {
  let strategy: string | null = null;
  if (authToken && sceneId) {
    try {
      const scene = (await api.getScene(authToken, sceneId)) ?? {};
      const raw = scene.camera_calibration;
      strategy =
        typeof raw === "string" && raw.trim() ? raw.trim() : "Manual";
    } catch {
      strategy = null;
    }
  }

  if (strategy === "Manual" || strategy == null) {
    return {
      ready: false,
      title:
        strategy === "Manual"
          ? "This scene uses Manual calibration — Auto Calibrate is not available"
          : "Could not load scene calibration type",
      strategy: strategy ?? null,
    };
  }

  if (strategy === "Markerless") {
    return {
      ready: false,
      title: "Go to 3D view for Markerless auto camera calibration.",
      strategy,
    };
  }

  const status = await getAutocalibrationStatus();
  if (status?.status === "running" || status?.status === "success") {
    return {
      ready: true,
      title: "Click to calibrate the camera automatically",
      strategy,
    };
  }
  return {
    ready: false,
    title:
      status?.message ||
      "Auto Calibrate system not available for this scene",
    strategy,
  };
}

/** Result payload from GET …/cameras/{id}/calibration (and socket push). */
export type AutocalibPoseResult = {
  status?: string;
  message?: string;
  calibration_points_2d?: number[][];
  calibration_points_3d?: number[][];
  quaternion?: number[];
  translation?: number[];
};

type IoSocket = {
  connected: boolean;
  emit: (event: string, data: unknown) => void;
  on: (event: string, cb: (data: unknown) => void) => void;
  off: (event: string, cb: (data: unknown) => void) => void;
  disconnect?: () => void;
};

type IoFactory = (
  opts: { path: string; transports: string[] },
) => IoSocket;

declare global {
  interface Window {
    io?: IoFactory;
  }
}

/** Strip a data:/blob: frame to raw base64 for the autocalibration POST. */
export async function frameToBase64(imageUrl: string): Promise<string> {
  if (imageUrl.startsWith("data:image/")) {
    const comma = imageUrl.indexOf(",");
    return comma >= 0 ? imageUrl.slice(comma + 1) : imageUrl;
  }
  const resp = await fetch(imageUrl);
  const blob = await resp.blob();
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("Failed to read frame"));
    reader.readAsDataURL(blob);
  });
  const comma = dataUrl.indexOf(",");
  return comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl;
}

export async function getAutocalibrationStatus(): Promise<AutocalibStatus | null> {
  try {
    const resp = await fetch(`${AUTOCALIB_PROXY_BASE}/status`, {
      method: "GET",
      headers: { Accept: "application/json" },
      credentials: "same-origin",
    });
    if (!resp.ok) {
      return null;
    }
    return (await resp.json()) as AutocalibStatus;
  } catch {
    return null;
  }
}

/**
 * Start marker-based auto calibration for one camera (legacy
 * startCameraCalibration from calibration.js).
 */
export async function startCameraAutocalibration(
  cameraUid: string,
  imageBase64: string,
  intrinsics: number[][],
): Promise<{ status?: string; message?: string }> {
  const resp = await fetch(
    `${AUTOCALIB_PROXY_BASE}/cameras/${encodeURIComponent(cameraUid)}/calibration`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({
        image: imageBase64,
        intrinsics,
      }),
    },
  );
  if (!resp.ok) {
    let message = `HTTP ${resp.status}`;
    try {
      const body = (await resp.json()) as { message?: string };
      if (body.message) {
        message = body.message;
      }
    } catch {
      /* keep status text */
    }
    return { status: "error", message };
  }
  return (await resp.json()) as { status?: string; message?: string };
}

export async function getCameraAutocalibration(
  cameraUid: string,
): Promise<AutocalibPoseResult> {
  const resp = await fetch(
    `${AUTOCALIB_PROXY_BASE}/cameras/${encodeURIComponent(cameraUid)}/calibration`,
    {
      method: "GET",
      headers: { Accept: "application/json" },
      credentials: "same-origin",
    },
  );
  if (!resp.ok) {
    let message = `HTTP ${resp.status}`;
    try {
      const body = (await resp.json()) as { message?: string };
      if (body.message) {
        message = body.message;
      }
    } catch {
      /* keep status text */
    }
    return { status: "error", message };
  }
  return (await resp.json()) as AutocalibPoseResult;
}

function isTerminalStatus(status: string | undefined): boolean {
  return status === "success" || status === "error";
}

/**
 * Wait for AprilTag / marker auto-calibrate to finish. Prefers the legacy
 * socket.io push (register_camera → calibration_result); falls back to
 * polling GET …/calibration so results still apply without a socket.
 */
export async function waitForCameraAutocalibration(
  cameraUid: string,
  options?: { timeoutMs?: number; intervalMs?: number },
): Promise<AutocalibPoseResult> {
  const timeoutMs = options?.timeoutMs ?? 90_000;
  const intervalMs = options?.intervalMs ?? 500;
  const deadline = Date.now() + timeoutMs;

  let socket: IoSocket | null = null;
  const pushed: { result: AutocalibPoseResult | null } = { result: null };
  let onResult: ((data: unknown) => void) | null = null;

  try {
    if (typeof window !== "undefined" && typeof window.io === "function") {
      socket = window.io({
        path: "/api/v1/autocalibration/socket.io",
        transports: ["websocket"],
      });
      onResult = (data: unknown) => {
        const note = data as {
          camera_id?: string;
          result?: AutocalibPoseResult;
        };
        if (
          note?.result &&
          (!note.camera_id || note.camera_id === cameraUid)
        ) {
          pushed.result = note.result;
        }
      };
      socket.on("calibration_result", onResult);
      const register = () => {
        socket?.emit("register_camera", { camera_id: cameraUid });
      };
      if (socket.connected) {
        register();
      } else {
        socket.on("connect", register);
      }
    }
  } catch {
    socket = null;
  }

  try {
    while (Date.now() < deadline) {
      const fromSocket = pushed.result;
      if (fromSocket && isTerminalStatus(fromSocket.status)) {
        return fromSocket;
      }
      const result = await getCameraAutocalibration(cameraUid);
      if (isTerminalStatus(result.status)) {
        return result;
      }
      await new Promise((r) => window.setTimeout(r, intervalMs));
    }
    return { status: "error", message: "Auto calibrate timed out" };
  } finally {
    if (socket && onResult) {
      try {
        socket.off("calibration_result", onResult);
      } catch {
        /* ignore */
      }
    }
    try {
      socket?.disconnect?.();
    } catch {
      /* ignore */
    }
  }
}

/** Parse 2d/3d correspondence arrays from an autocalib success payload. */
export function pointsFromAutocalibResult(result: AutocalibPoseResult): {
  camPoints: [number, number][];
  mapPoints: [number, number, number][];
} | null {
  const pts2 = result.calibration_points_2d;
  const pts3 = result.calibration_points_3d;
  if (!Array.isArray(pts2) || !Array.isArray(pts3) || pts2.length === 0) {
    return null;
  }
  const n = Math.min(pts2.length, pts3.length);
  const camPoints: [number, number][] = [];
  const mapPoints: [number, number, number][] = [];
  for (let i = 0; i < n; i++) {
    const c = pts2[i];
    const m = pts3[i];
    if (
      !Array.isArray(c) ||
      c.length < 2 ||
      !Array.isArray(m) ||
      m.length < 2
    ) {
      continue;
    }
    const cx = Number(c[0]);
    const cy = Number(c[1]);
    const mx = Number(m[0]);
    const my = Number(m[1]);
    const mz = m.length >= 3 ? Number(m[2]) : 0;
    if (
      ![cx, cy, mx, my, mz].every((v) => Number.isFinite(v))
    ) {
      continue;
    }
    camPoints.push([cx, cy]);
    mapPoints.push([mx, my, mz]);
  }
  if (camPoints.length === 0 || camPoints.length !== mapPoints.length) {
    return null;
  }
  return { camPoints, mapPoints };
}
