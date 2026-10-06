// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Replay provider registry (Phase 2.6). The UI selects a provider by id —
 * deployment config, not code changes. Today only the stub exists; the
 * Rerun provider registers itself once @rerun-io/web-viewer-react lands.
 */

import { StubProvider } from "./StubProvider";
import type { ReplayProvider } from "./types";

export type { RecordingMeta, ReplayProvider } from "./types";
export { StubProvider } from "./StubProvider";

const providers = new Map<string, ReplayProvider>();

export function registerReplayProvider(provider: ReplayProvider): void {
  providers.set(provider.id, provider);
}

export function getReplayProvider(id: string): ReplayProvider | undefined {
  return providers.get(id);
}

/** Provider id from deployment config; falls back to the stub. */
export function resolveReplayProvider(): ReplayProvider {
  const configured =
    typeof window !== "undefined"
      ? (window as unknown as { ssReplayProvider?: string }).ssReplayProvider
      : undefined;
  if (configured) {
    const found = providers.get(configured);
    if (found) {
      return found;
    }
  }
  let stub = providers.get("stub");
  if (!stub) {
    stub = new StubProvider();
    providers.set("stub", stub);
  }
  return stub;
}

// The stub is always available.
registerReplayProvider(new StubProvider());
