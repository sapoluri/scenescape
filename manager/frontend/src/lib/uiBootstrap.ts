// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { readBootstrapJson } from "./bootstrap";
import { readAuthToken } from "./authToken";

export type UiBootstrapPage =
  | "chrome"
  | "scenes"
  | "scene"
  | "cameras"
  | "sensors"
  | "assets"
  | "models"
  | "list-sheets";

/**
 * Prefer embedded `json_script`, else GET /api/v1/ui-bootstrap/ (session or Token).
 * Enables static shells without Django page templates.
 */
export async function loadUiBootstrap<T>(
  elementId: string,
  page: UiBootstrapPage,
  entityId?: string,
): Promise<T | null> {
  const embedded = readBootstrapJson<T>(elementId);
  if (embedded) {
    return embedded;
  }
  const params = new URLSearchParams({ page });
  if (entityId) {
    params.set("id", entityId);
  }
  const headers: HeadersInit = {};
  const token = readAuthToken();
  if (token) {
    headers.Authorization = `Token ${token}`;
  }
  const res = await fetch(`/api/v1/ui-bootstrap/?${params.toString()}`, {
    credentials: "same-origin",
    headers,
  });
  if (res.status === 401) {
    const next = encodeURIComponent(
      `${window.location.pathname}${window.location.search}`,
    );
    window.location.href = `/sign_in/?next=${next}`;
    return null;
  }
  if (!res.ok) {
    console.error(`ui-bootstrap failed: ${res.status} ${page}`);
    return null;
  }
  return (await res.json()) as T;
}
