// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { enterCameraCalibrate } from "../viewport/calibrate";
import { getViewportState, useViewportStore } from "../viewport/store";
import type {
  CameraEntity,
  MarkEntity,
  RegionEntity,
  SensorEntity,
  TripwireEntity,
  ViewportEntity,
} from "../viewport/types";
import "./Properties.css";

function fmt(n: number, digits = 2): string {
  return Number.isFinite(n) ? n.toFixed(digits) : "—";
}

function polygonArea(points: [number, number][]): number {
  let a = 0;
  for (let i = 0; i < points.length; i++) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % points.length];
    a += x1 * y2 - x2 * y1;
  }
  return Math.abs(a / 2);
}

function polylineLength(points: [number, number][]): number {
  let l = 0;
  for (let i = 1; i < points.length; i++) {
    l += Math.hypot(
      points[i][0] - points[i - 1][0],
      points[i][1] - points[i - 1][1],
    );
  }
  return l;
}

function NumField({
  label,
  value,
  step = 0.1,
  onCommit,
}: {
  label: string;
  value: number;
  step?: number;
  onCommit: (v: number) => void;
}) {
  const [text, setText] = useState<string | null>(null);
  return (
    <label className="ss-prop-field">
      <span className="ss-prop-label">{label}</span>
      <input
        type="number"
        step={step}
        className="ss-prop-input"
        value={text ?? String(value)}
        onChange={(ev) => setText(ev.target.value)}
        onBlur={() => {
          if (text !== null) {
            const v = Number(text);
            if (Number.isFinite(v)) {
              onCommit(v);
            }
            setText(null);
          }
        }}
        onKeyDown={(ev) => {
          if (ev.key === "Enter") {
            (ev.target as HTMLInputElement).blur();
          }
        }}
      />
    </label>
  );
}

function ReadField({ label, value }: { label: string; value: string }) {
  return (
    <div className="ss-prop-field">
      <span className="ss-prop-label">{label}</span>
      <span className="ss-prop-value">{value}</span>
    </div>
  );
}

function NameRow({ entity }: { entity: ViewportEntity }) {
  const [text, setText] = useState<string | null>(null);
  return (
    <label className="ss-prop-field ss-prop-name">
      <span className="ss-prop-label">Name</span>
      <input
        className="ss-prop-input"
        value={text ?? entity.name}
        onChange={(ev) => setText(ev.target.value)}
        onBlur={() => {
          if (text !== null && text.trim()) {
            getViewportState().updateEntity(entity.id, { name: text.trim() });
          }
          setText(null);
        }}
        onKeyDown={(ev) => {
          if (ev.key === "Enter") {
            (ev.target as HTMLInputElement).blur();
          }
        }}
      />
    </label>
  );
}

function ColorRow({ entity }: { entity: ViewportEntity & { color: string } }) {
  return (
    <label className="ss-prop-field">
      <span className="ss-prop-label">Color</span>
      <input
        type="color"
        className="ss-prop-color"
        value={entity.color}
        onChange={(ev) =>
          getViewportState().updateEntity(entity.id, { color: ev.target.value })
        }
      />
    </label>
  );
}

function RegionProps({ e }: { e: RegionEntity }) {
  const patch = (p: Partial<RegionEntity>) =>
    getViewportState().updateEntity(e.id, p);
  return (
    <>
      <NameRow entity={e} />
      <ColorRow entity={e} />
      <ReadField label="Area" value={`${fmt(polygonArea(e.points))} m²`} />
      <ReadField label="Corners" value={String(e.points.length)} />
      <NumField
        label="Height (m)"
        value={e.height}
        onCommit={(v) => patch({ height: Math.max(0.1, v) })}
      />
      <NumField
        label="Buffer (m)"
        value={e.bufferSize}
        onCommit={(v) => patch({ bufferSize: Math.max(0, v) })}
      />
      <label className="ss-prop-check">
        <input
          type="checkbox"
          checked={e.volumetric}
          onChange={(ev) => patch({ volumetric: ev.target.checked })}
        />
        <span>Volumetric</span>
      </label>
      <p className="ss-prop-hint">
        Drag corners and edges in the viewport for precise shaping; values here
        tune height, buffer, and volume.
      </p>
    </>
  );
}

function TripwireProps({ e }: { e: TripwireEntity }) {
  return (
    <>
      <NameRow entity={e} />
      <ColorRow entity={e} />
      <ReadField label="Length" value={`${fmt(polylineLength(e.points))} m`} />
      <ReadField label="Points" value={String(e.points.length)} />
      <ReadField
        label="Direction"
        value={`${fmt(e.a[0])}, ${fmt(e.a[1])} → ${fmt(e.b[0])}, ${fmt(e.b[1])}`}
      />
      <p className="ss-prop-hint">
        Crossings are counted A → B. Redraw the line in the viewport to reshape
        it.
      </p>
    </>
  );
}

function CameraProps({
  e,
  calibrateHref,
  onCalibrate,
}: {
  e: CameraEntity;
  calibrateHref?: string;
  onCalibrate?: () => void;
}) {
  const patch = (p: Partial<CameraEntity>) =>
    getViewportState().updateEntity(e.id, p);
  const setPos = (i: number, v: number) => {
    const position: [number, number, number] = [...e.position];
    position[i] = v;
    patch({ position });
  };
  return (
    <>
      <NameRow entity={e} />
      <div className="ss-prop-grid3">
        <NumField label="X" value={e.position[0]} onCommit={(v) => setPos(0, v)} />
        <NumField label="Y" value={e.position[1]} onCommit={(v) => setPos(1, v)} />
        <NumField label="Z" value={e.position[2]} onCommit={(v) => setPos(2, v)} />
      </div>
      <NumField
        label="FOV (°)"
        value={e.fov}
        step={1}
        onCommit={(v) => patch({ fov: Math.min(120, Math.max(10, v)) })}
      />
      <ReadField
        label="Rotation"
        value={`${fmt(e.rotation[0], 1)}°, ${fmt(e.rotation[1], 1)}°, ${fmt(e.rotation[2], 1)}°`}
      />
      {(onCalibrate || calibrateHref) && (
        onCalibrate ? (
          <button
            type="button"
            className="ss-prop-calibrate"
            onClick={onCalibrate}
          >
            <i className="bi bi-crosshair" aria-hidden="true" />
            <span>Calibrate camera</span>
          </button>
        ) : (
          <a href={calibrateHref} className="ss-prop-calibrate">
            <i className="bi bi-crosshair" aria-hidden="true" />
            <span>Calibrate camera</span>
          </a>
        )
      )}
      <p className="ss-prop-hint">
        Opens in-viewport calibrate (3D + live feed). Pose updates live as you
        place 4+ correspondences; Save writes the same camera PUT as before.
      </p>
    </>
  );
}

function SensorProps({ e }: { e: SensorEntity }) {
  const patch = (p: Partial<SensorEntity>) =>
    getViewportState().updateEntity(e.id, p);
  const setPos = (i: number, v: number) => {
    const position: [number, number, number] = [...e.position];
    position[i] = v;
    patch({ position });
  };
  return (
    <>
      <NameRow entity={e} />
      <ColorRow entity={e} />
      <div className="ss-prop-grid3">
        <NumField label="X" value={e.position[0]} onCommit={(v) => setPos(0, v)} />
        <NumField label="Y" value={e.position[1]} onCommit={(v) => setPos(1, v)} />
        <NumField label="Z" value={e.position[2]} onCommit={(v) => setPos(2, v)} />
      </div>
      <NumField
        label="Radius (m)"
        value={e.radius}
        onCommit={(v) => patch({ radius: Math.max(0.5, v) })}
      />
    </>
  );
}

function MarkProps({ e }: { e: MarkEntity }) {
  return (
    <>
      <ReadField label="Class" value={e.className} />
      <ReadField
        label="Position"
        value={`${fmt(e.position[0])}, ${fmt(e.position[1])}, ${fmt(e.position[2])}`}
      />
      <ReadField
        label="Heading"
        value={
          Number.isFinite(e.heading)
            ? `${fmt((e.heading * 180) / Math.PI, 1)}°`
            : "—"
        }
      />
      <ReadField label="Speed" value={`${fmt(e.speed)} m/frame`} />
      <p className="ss-prop-hint">
        Live tracked object — values update from the tracker feed. Mark shape
        and color come from the Object Library (B).
      </p>
    </>
  );
}

function EmptyProps({ sceneName }: { sceneName: string }) {
  const entities = useViewportStore((s) => s.entities);
  const counts = { region: 0, tripwire: 0, camera: 0, sensor: 0, mark: 0 };
  for (const e of Object.values(entities)) {
    if (e.type in counts) {
      counts[e.type as keyof typeof counts] += 1;
    }
  }
  return (
    <>
      <ReadField label="Scene" value={sceneName} />
      <ReadField label="Regions" value={String(counts.region)} />
      <ReadField label="Tripwires" value={String(counts.tripwire)} />
      <ReadField label="Cameras" value={String(counts.camera)} />
      <ReadField label="Sensors" value={String(counts.sensor)} />
      <ReadField label="Tracked now" value={String(counts.mark)} />
      <p className="ss-prop-hint">
        Select anything in the viewport or Outliner to inspect it. Press{" "}
        <kbd>G</kbd> to draw a region, <kbd>T</kbd> for a tripwire.
      </p>
    </>
  );
}

/**
 * Properties: Blender-style inspector for the current selection.
 * Numeric fields write through the viewport store so the 3D view updates
 * live. Nothing selected → scene summary.
 */
export function Properties({
  sceneName,
  cameras = [],
}: {
  sceneName: string;
  cameras?: {
    id: string;
    sensorId: string;
    name: string;
    calibrateHref: string;
  }[];
}) {
  const selectedId = useViewportStore((s) => s.selectedId);
  const entities = useViewportStore((s) => s.entities);
  const e = selectedId ? entities[selectedId] : undefined;
  const camBoot =
    e?.type === "camera" ? cameras.find((c) => c.id === e.id) : undefined;

  return (
    <div className="ss-props">
      {!e && <EmptyProps sceneName={sceneName} />}
      {e?.type === "region" && <RegionProps e={e} />}
      {e?.type === "tripwire" && <TripwireProps e={e} />}
      {e?.type === "camera" && (
        <CameraProps
          e={e}
          calibrateHref={camBoot?.calibrateHref}
          onCalibrate={
            camBoot
              ? () =>
                  enterCameraCalibrate({
                    cameraId: camBoot.id,
                    sensorId: camBoot.sensorId,
                    cameraName: camBoot.name,
                  })
              : undefined
          }
        />
      )}
      {e?.type === "sensor" && <SensorProps e={e} />}
      {e?.type === "mark" && <MarkProps e={e} />}
      {e?.type === "child" && (
        <>
          <NameRow entity={e} />
          <p className="ss-prop-hint">Linked child scene.</p>
        </>
      )}
    </div>
  );
}
