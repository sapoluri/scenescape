// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const NON_BITMAP_EXT = /\.(glb|gltf|ply|zip)(\?|#|$)/i;

function firstBitmapUrl(
  ...candidates: Array<string | null | undefined>
): string | null {
  for (const raw of candidates) {
    if (!raw || typeof raw !== "string") {
      continue;
    }
    const url = raw.trim();
    if (!url || NON_BITMAP_EXT.test(url)) {
      continue;
    }
    return url;
  }
  return null;
}

/**
 * URL for a 2D scene preview bitmap (gallery / map host).
 * Prefer the generated ortho thumbnail; fall back to image maps only
 * (never .glb / .ply / .zip — those are not displayable as `<img>`).
 */
export function sceneMapBitmapUrl(scene: {
  thumbnailUrl?: string | null;
  mapUrl?: string | null;
  thumbnail?: string | null;
  map?: string | null;
}): string | null {
  return firstBitmapUrl(
    scene.thumbnailUrl,
    scene.thumbnail,
    scene.mapUrl,
    scene.map,
  );
}
