// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const AUTOCALIB_PROXY_BASE = "/api/v1/autocalibration";

export type AutocalibStatus = {
  status?: string;
  message?: string;
  strategy?: string;
};

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
