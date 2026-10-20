// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { endMqttClient } from "../mqtt/client";

/**
 * Portable session helpers for Manager islands.
 *
 * Token REST (`lib/rest.ts`) is the long-term auth path. Session + CSRF is
 * transitional for DeleteViews, mesh generation, and model-directory until
 * those move behind Token APIs. Feature code should call these helpers rather
 * than reading `csrfmiddlewaretoken` or `csrftoken` directly.
 *
 * See docs/design/manager-ui-backend-contract.md §5.2.
 */

export function getCsrfToken(): string {
  const input = document.querySelector(
    'input[name="csrfmiddlewaretoken"]',
  ) as HTMLInputElement | null;
  if (input?.value) {
    return input.value;
  }
  const match = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : "";
}

let loggingOut = false;

export function handleExpiredSession(response: Response, token: string): void {
  if (response.status !== 401 || !token || loggingOut ||
      window.location.pathname.startsWith("/sign_in/") ||
      window.location.pathname.startsWith("/sign_out/")) {
    return;
  }
  loggingOut = true;
  endMqttClient(window.ssMqttClient);
  window.ssMqttClient = undefined;
  window.location.replace("/sign_out/");
}

export function watchSessionExpiry(token: string): () => void {
  if (!token) return () => {};
  let active = true;
  let pending = false;
  let lastCheck = -Infinity;
  const check = async () => {
    if (!active || loggingOut || pending || document.visibilityState === "hidden" ||
        Date.now() - lastCheck < 1000) return;
    pending = true;
    lastCheck = Date.now();
    try {
      const response = await fetch("/api/v1/session", {
        credentials: "same-origin",
        cache: "no-store",
      });
      if (active) handleExpiredSession(response, token);
    } catch {
      return;
    } finally {
      pending = false;
    }
  };
  window.addEventListener("focus", check);
  document.addEventListener("visibilitychange", check);
  void check();
  return () => {
    active = false;
    window.removeEventListener("focus", check);
    document.removeEventListener("visibilitychange", check);
  };
}

/** Headers for Token-authenticated `/api/v1` calls. */
export function tokenAuthHeaders(token: string): HeadersInit {
  return token ? { Authorization: `Token ${token}` } : {};
}

/** CSRF header for session-authenticated mutating requests. */
export function csrfHeaders(extra?: HeadersInit): HeadersInit {
  const csrf = getCsrfToken();
  return {
    ...(csrf ? { "X-CSRFToken": csrf } : {}),
    ...extra,
  };
}

function methodNeedsCsrf(method: string): boolean {
  const m = method.toUpperCase();
  return m !== "GET" && m !== "HEAD" && m !== "OPTIONS" && m !== "TRACE";
}

/**
 * `fetch` with same-origin credentials and CSRF on mutating methods.
 * Prefer `lib/rest.ts` for Token CRUD.
 */
export async function sessionFetch(
  input: RequestInfo | URL,
  init: RequestInit = {},
): Promise<Response> {
  const method = init.method || "GET";
  const headers = new Headers(init.headers || {});
  if (methodNeedsCsrf(method) && !headers.has("X-CSRFToken")) {
    const csrf = getCsrfToken();
    if (csrf) {
      headers.set("X-CSRFToken", csrf);
    }
  }
  return fetch(input, {
    ...init,
    credentials: init.credentials ?? "same-origin",
    headers,
  });
}
