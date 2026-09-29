// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { readBootstrapJson } from "./bootstrap";

const BOOTSTRAP_TOKEN_IDS = [
  "ss-auth-bootstrap",
  "ss-scene-detail-bootstrap",
  "ss-scenes-home-bootstrap",
  "ss-list-sheets-bootstrap",
] as const;

/**
 * DRF Token for portable `/api/v1` calls. Prefers the page-wide
 * `ss-auth-bootstrap` from base.html, then island bootstraps, then legacy
 * `#auth-token`.
 */
export function readAuthToken(): string {
  for (const id of BOOTSTRAP_TOKEN_IDS) {
    if (id === "ss-auth-bootstrap") {
      const el = document.getElementById(id);
      if (!el?.textContent) {
        continue;
      }
      try {
        const parsed = JSON.parse(el.textContent);
        if (typeof parsed === "string" && parsed) {
          return parsed;
        }
        if (
          parsed &&
          typeof parsed === "object" &&
          typeof (parsed as { authToken?: string }).authToken === "string"
        ) {
          return (parsed as { authToken: string }).authToken;
        }
      } catch {
        /* ignore */
      }
      continue;
    }
    const boot = readBootstrapJson<{ authToken?: string }>(id);
    if (boot?.authToken) {
      return boot.authToken;
    }
  }
  const legacy = document.getElementById(
    "auth-token",
  ) as HTMLInputElement | null;
  return legacy?.value?.trim() || "";
}
