// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Replay provider registry (Phase 2.6). The UI selects a provider by id —
 * deployment config, not code changes. Today only the stub exists; the
 * Rerun provider registers itself once @rerun-io/web-viewer-react lands.
 */

import { StubProvider } from "./StubProvider";
import { RerunProvider } from "./RerunProvider";
import type { ReplayProvider } from "./types";

export type { RecordingMeta, ReplayProvider } from "./types";
export { StubProvider } from "./StubProvider";
export { RerunProvider } from "./RerunProvider";

const providers = new Map<string, ReplayProvider>();

export function registerReplayProvider(provider: ReplayProvider): void {
  providers.set(provider.id, provider);
}

export function getReplayProvider(id: string): ReplayProvider | undefined {
  return providers.get(id);
}

/** Provider id from deployment config; defaults to the Rerun reference backend. */
export function resolveReplayProvider(): ReplayProvider {
  const configured =
    typeof window !== "undefined"
      ? (window as unknown as { ssReplayProvider?: string }).ssReplayProvider
      : undefined;
  const want = configured || "rerun";
  const found = providers.get(want);
  if (found) {
    return found;
  }
  let stub = providers.get("stub");
  if (!stub) {
    stub = new StubProvider();
    providers.set("stub", stub);
  }
  return stub;
}

// The stub is always available; Rerun is the reference backend (2.6).
registerReplayProvider(new StubProvider());
registerReplayProvider(new RerunProvider());
