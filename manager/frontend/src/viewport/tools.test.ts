// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from "vitest";
import {
  attachPlacementGuidance,
  CREATION_TOOLS,
  TOOLS,
  TRANSFORM_TOOLS,
  entityCanScale,
  isToolEnabledInScene,
  toolDef,
} from "./tools";
import type { ToolId } from "./types";

describe("TOOLS", () => {
  it.each(CREATION_TOOLS)("prompts and enables the placement cursor for %s", (tool) => {
    const host = { setAttribute: vi.fn(), removeAttribute: vi.fn() };
    const notify = vi.fn();
    const cleanup = attachPlacementGuidance(host, tool, notify);
    expect(host.setAttribute).toHaveBeenCalledWith("data-placement", tool);
    const persistent = tool === "region" || tool === "tripwire";
    expect(notify).toHaveBeenCalledExactlyOnceWith(
      persistent
        ? expect.stringMatching(
            /^Continue clicking the scene floor to add .* points\. Press Enter to finish\.$/,
          )
        : expect.stringMatching(/^Click the scene floor to /),
      persistent,
    );
    expect(notify.mock.calls[0][0]).toContain(tool === "measure" ? "measuring" : tool);
    cleanup();
    expect(host.removeAttribute).toHaveBeenCalledWith("data-placement");
  });

  it.each<ToolId>(["live", "move", "rotate", "scale"])(
    "clears placement without a prompt for %s",
    (tool) => {
      const host = { setAttribute: vi.fn(), removeAttribute: vi.fn() };
      const notify = vi.fn();
      attachPlacementGuidance(host, tool, notify)();
      expect(host.setAttribute).not.toHaveBeenCalled();
      expect(host.removeAttribute).toHaveBeenCalledWith("data-placement");
      expect(notify).not.toHaveBeenCalled();
    },
  );

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
    expect(toolDef("rotate").icon).toBe("bi-arrow-repeat");
    expect(toolDef("camera").icon).toBe("bi-camera-video");
    expect(toolDef("region").icon).toBe("bi-bounding-box-circles");
    expect(toolDef("tripwire").icon).toBe("bi-arrow-bar-right");
    expect(() => toolDef("nope" as ToolId)).toThrow();
  });

  it("disables scene-editing tools when no scene is open", () => {
    expect(isToolEnabledInScene("live", false)).toBe(true);
    expect(isToolEnabledInScene("move", false)).toBe(true);
    expect(isToolEnabledInScene("rotate", false)).toBe(true);
    expect(isToolEnabledInScene("region", false)).toBe(false);
    expect(isToolEnabledInScene("tripwire", false)).toBe(false);
    expect(isToolEnabledInScene("camera", false)).toBe(false);
    expect(isToolEnabledInScene("sensor", false)).toBe(false);
    expect(isToolEnabledInScene("measure", false)).toBe(false);
    expect(isToolEnabledInScene("region", true)).toBe(true);
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
