// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";
import { loadUiBootstrap } from "../lib/uiBootstrap";
import type { ChromeBootstrap } from "./types";

let cached: ChromeBootstrap | null | undefined;

/**
 * Shared chrome bootstrap (urls, version, staff). Cached after the first
 * load so the 3D menu bar and AppChrome can both read it cheaply.
 */
export function useChromeBootstrap(): ChromeBootstrap | null {
  const [boot, setBoot] = useState<ChromeBootstrap | null>(
    cached === undefined ? null : cached,
  );

  useEffect(() => {
    if (cached !== undefined) {
      setBoot(cached);
      return;
    }
    let cancelled = false;
    void loadUiBootstrap<ChromeBootstrap>("ss-chrome-bootstrap", "chrome").then(
      (payload) => {
        cached = payload;
        if (!cancelled) {
          setBoot(payload);
        }
      },
    );
    return () => {
      cancelled = true;
    };
  }, []);

  return boot;
}
