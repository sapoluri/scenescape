// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { CREATION_TOOLS, TOOLS, TRANSFORM_TOOLS, toolDef } from "./tools";
import type { ToolId } from "./types";

describe("TOOLS", () => {
  it("has the Phase 1.2 tool set with unique ids and shortcuts", () => {
    const ids = TOOLS.map((t) => t.id);
    expect(new Set(ids).size).toBe(ids.length);
    const shortcuts = TOOLS.map((t) => t.shortcut.toLowerCase());
    expect(new Set(shortcuts).size).toBe(shortcuts.length);
    const expected: ToolId[] = [
      "select",
      "move",
      "rotate",
      "scale",
      "region",
      "tripwire",
      "camera",
      "sensor",
      "measure",
      "live",
    ];
    for (const id of expected) {
      expect(ids).toContain(id);
    }
  });

  it("toolDef resolves and throws on unknown ids", () => {
    expect(toolDef("move").label).toBe("Move");
    expect(() => toolDef("nope" as ToolId)).toThrow();
  });

  it("transform/creation tool lists are subsets of the tool set", () => {
    const ids = new Set(TOOLS.map((t) => t.id));
    for (const id of [...TRANSFORM_TOOLS, ...CREATION_TOOLS]) {
      expect(ids.has(id)).toBe(true);
    }
  });
});
