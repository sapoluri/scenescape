// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import {
  CREATION_TOOLS,
  TOOLS,
  TRANSFORM_TOOLS,
  entityCanScale,
  toolDef,
} from "./tools";
import type { ToolId } from "./types";

describe("TOOLS", () => {
  it("has the Phase 1.2 tool set with unique ids and shortcuts", () => {
    const ids = TOOLS.map((t) => t.id);
    expect(new Set(ids).size).toBe(ids.length);
    const shortcuts = TOOLS.map((t) => t.shortcut.toLowerCase());
    expect(new Set(shortcuts).size).toBe(shortcuts.length);
    const expected: ToolId[] = [
      "live",
      "move",
      "rotate",
      "scale",
      "region",
      "tripwire",
      "camera",
      "sensor",
      "measure",
    ];
    for (const id of expected) {
      expect(ids).toContain(id);
    }
  });

  it("toolDef resolves and throws on unknown ids", () => {
    expect(toolDef("move").label).toBe("Move");
    expect(toolDef("tripwire").icon).toBe("bi-arrow-bar-right");
    expect(() => toolDef("nope" as ToolId)).toThrow();
  });

  it("transform/creation tool lists are subsets of the tool set", () => {
    const ids = new Set(TOOLS.map((t) => t.id));
    for (const id of [...TRANSFORM_TOOLS, ...CREATION_TOOLS]) {
      expect(ids.has(id)).toBe(true);
    }
  });

  it("allows scale only for regions and tripwires", () => {
    expect(entityCanScale({ type: "region" })).toBe(true);
    expect(entityCanScale({ type: "tripwire" })).toBe(true);
    expect(entityCanScale({ type: "camera" })).toBe(false);
    expect(entityCanScale({ type: "sensor" })).toBe(false);
    expect(entityCanScale({ type: "mark" })).toBe(false);
    expect(entityCanScale({ type: "child" })).toBe(false);
    expect(entityCanScale(null)).toBe(false);
    expect(entityCanScale(undefined)).toBe(false);
  });
});
