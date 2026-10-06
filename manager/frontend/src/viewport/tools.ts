// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ToolId } from "./types";

/** Phase 1.2 tool system: definitions for the viewport tool header. */
export interface ToolDef {
  id: ToolId;
  label: string;
  /** Bootstrap icon class (the app already ships bootstrap-icons). */
  icon: string;
  shortcut: string;
  hint: string;
}

export const TOOLS: ToolDef[] = [
  {
    id: "live",
    label: "Live",
    icon: "bi-play-circle",
    shortcut: "L",
    hint: "Navigate + click to select (default)",
  },
  {
    id: "select",
    label: "Select",
    icon: "bi-cursor",
    shortcut: "Q",
    hint: "Click to select (orbit off while held)",
  },
  {
    id: "move",
    label: "Move",
    icon: "bi-arrows-move",
    shortcut: "W",
    hint: "Drag the gizmo to move the selection",
  },
  {
    id: "rotate",
    label: "Rotate",
    icon: "bi-arrow-clockwise",
    shortcut: "E",
    hint: "Drag the gizmo to rotate the selection",
  },
  {
    id: "scale",
    label: "Scale",
    icon: "bi-aspect-ratio",
    shortcut: "R",
    hint: "Drag the gizmo to scale the selection",
  },
  {
    id: "region",
    label: "Region",
    icon: "bi-bounding-box",
    shortcut: "G",
    hint: "Click floor points, double-click/Enter to finish (Esc cancels)",
  },
  {
    id: "tripwire",
    label: "Tripwire",
    icon: "bi-slash-lg",
    shortcut: "T",
    hint: "Click 2+ floor points, double-click/Enter to finish (Esc cancels)",
  },
  {
    id: "camera",
    label: "Camera",
    icon: "bi-camera",
    shortcut: "C",
    hint: "Click the floor to place a camera marker",
  },
  {
    id: "sensor",
    label: "Sensor",
    icon: "bi-broadcast",
    shortcut: "S",
    hint: "Click the floor to place a sensor marker",
  },
  {
    id: "measure",
    label: "Measure",
    icon: "bi-rulers",
    shortcut: "M",
    hint: "Click two floor points to measure distance (Esc clears)",
  },
];

/** Tools that attach the TransformControls gizmo to the selection. */
export const TRANSFORM_TOOLS: ToolId[] = ["move", "rotate", "scale"];

/** Tools with an in-progress floor-plane creation gesture. */
export const CREATION_TOOLS: ToolId[] = [
  "region",
  "tripwire",
  "camera",
  "sensor",
  "measure",
];

export function toolDef(id: ToolId): ToolDef {
  const def = TOOLS.find((t) => t.id === id);
  if (!def) {
    throw new Error(`unknown tool: ${id}`);
  }
  return def;
}
