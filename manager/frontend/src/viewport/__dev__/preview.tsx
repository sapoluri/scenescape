// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Dev-only preview harness for the 3D viewport (not a build entry).
 * Served by `vite dev` via viewport-preview.html. Seeds synthetic entities
 * and synthetic live marks so the viewport + tools can be exercised without
 * the Django backend.
 */
import { StrictMode, useEffect } from "react";
import { createRoot } from "react-dom/client";
import { SceneViewport } from "../SceneViewport";
import { getViewportState } from "../store";
import type {
  CameraEntity,
  MarkEntity,
  RegionEntity,
  SensorEntity,
  TripwireEntity,
} from "../types";
import "../viewport.css";
import "../../tokens/tokens.css";

function seedSyntheticEntities(): void {
  const s = getViewportState();
  const region: RegionEntity = {
    id: "region-dock",
    type: "region",
    name: "Loading dock",
    visible: true,
    points: [
      [-6, -4],
      [2, -4],
      [2, 3],
      [-6, 3],
    ],
    min: [-6, -4],
    max: [2, 3],
    height: 2.5,
    color: "#30d158",
    volumetric: true,
    bufferSize: 0,
  };
  const flat: RegionEntity = {
    id: "region-yard",
    type: "region",
    name: "Yard",
    visible: true,
    points: [
      [4, -5],
      [11, -5],
      [11, 2],
      [4, 2],
    ],
    min: [4, -5],
    max: [11, 2],
    height: 1,
    color: "#ffd60a",
    volumetric: false,
    bufferSize: 0,
  };
  const trip: TripwireEntity = {
    id: "trip-gate",
    type: "tripwire",
    name: "Gate line",
    visible: true,
    points: [
      [-8, 6],
      [-2, 6],
      [3, 9],
    ],
    a: [-8, 6],
    b: [3, 9],
    color: "#ff9f0a",
  };
  const camera: CameraEntity = {
    id: "cam-1",
    type: "camera",
    name: "Overhead cam",
    visible: true,
    position: [-10, -8, 4],
    target: [0, 0, 0],
    rotation: [-35, 0, 45],
    fov: 60,
    color: "#0a84ff",
  };
  const sensor: SensorEntity = {
    id: "sensor-1",
    type: "sensor",
    name: "Lidar north",
    visible: true,
    position: [8, 8, 0],
    radius: 6,
    color: "#bf5af2",
  };
  s.upsertEntities([region, flat, trip, camera, sensor]);
}

function emitSyntheticMarks(): void {
  // Drive the frozen live-mark event the viewport consumes.
  const marks: MarkEntity[] = [
    {
      id: "mark-1",
      type: "mark",
      name: "person",
      visible: true,
      className: "person",
      color: "#ff453a",
      position: [-3, 0, 0],
      heading: 0.6,
      speed: 1.2,
    },
    {
      id: "mark-2",
      type: "mark",
      name: "forklift",
      visible: true,
      className: "forklift",
      color: "#0a84ff",
      position: [6, -2, 0],
      heading: 2.4,
      speed: 0.8,
    },
  ];
  window.dispatchEvent(
    new CustomEvent("ss-scene-objects", { detail: { objects: marks } }),
  );
}

function PreviewApp() {
  useEffect(() => {
    seedSyntheticEntities();
    emitSyntheticMarks();
    const t = window.setInterval(emitSyntheticMarks, 4000);
    return () => window.clearInterval(t);
  }, []);
  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      <SceneViewport
        sceneId="preview"
        assetMarkColors={{ person: "#ff453a", forklift: "#0a84ff" }}
        cameras={[]}
        sensors={[]}
        authToken=""
      />
    </div>
  );
}

const root = document.getElementById("preview-root");
if (root) {
  createRoot(root).render(
    <StrictMode>
      <PreviewApp />
    </StrictMode>,
  );
}
