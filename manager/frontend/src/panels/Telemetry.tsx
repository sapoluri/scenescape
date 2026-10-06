// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from "react";
import { useMqttConnected } from "../scene/useLiveChrome";
import { useViewportMarksList } from "../viewport/marks";
import type { MarkEntity } from "../viewport/types";
import "./Telemetry.css";

/**
 * Telemetry pane: MQTT connection chip, live tracked-object summary, and a
 * capped event feed of track appearances/disappearances.
 *
 * Live data comes from the viewport scene-graph store (fed by the frozen
 * `ss-scene-objects` event in `viewport/marks.ts`) — no new plumbing.
 */

export const MAX_TELEMETRY_EVENTS = 200;

export type TelemetryEventKind = "track-new" | "track-lost";

export interface TelemetryEvent {
  /** Monotonic sequence number, unique within the feed. */
  seq: number;
  /** Epoch ms when the event was recorded. */
  at: number;
  kind: TelemetryEventKind;
  /** Store mark id, e.g. "mark:42". */
  trackId: string;
  /** Asset3D class name, e.g. "person". */
  className: string;
  /** Last known position [x, y] meters, when known. */
  position?: [number, number];
  /** Per-batch displacement in meters, when known and nonzero. */
  speed?: number;
}

export interface ClassSummary {
  className: string;
  count: number;
  color: string;
}

/**
 * Diff a snapshot of live marks against the previously known ids and return
 * the new feed events plus the updated known-id set. Pure — unit tested.
 * Newest events first; callers cap the feed with `capTelemetryEvents`.
 */
export function diffTelemetryBatch(
  known: ReadonlySet<string>,
  marks: MarkEntity[],
  now: number,
  seqStart: number,
): { events: TelemetryEvent[]; known: Set<string>; nextSeq: number } {
  const events: TelemetryEvent[] = [];
  let seq = seqStart;
  const next = new Set<string>();
  const byId = new Map<string, MarkEntity>();
  for (const m of marks) {
    next.add(m.id);
    byId.set(m.id, m);
  }
  for (const m of marks) {
    if (!known.has(m.id)) {
      const speed = Number.isFinite(m.speed) && m.speed > 0 ? m.speed : undefined;
      events.push({
        seq: seq++,
        at: now,
        kind: "track-new",
        trackId: m.id,
        className: m.className,
        position: [m.position[0], m.position[1]],
        speed,
      });
    }
  }
  // Lost tracks, in previous-known order for stability.
  for (const id of known) {
    if (!next.has(id)) {
      const prev = byId.get(id);
      events.push({
        seq: seq++,
        at: now,
        kind: "track-lost",
        trackId: id,
        className: prev?.className ?? id.replace(/^mark:/, ""),
      });
    }
  }
  // Newest first: reverse so later ids in this batch still read newest-first.
  events.reverse();
  return { events, known: next, nextSeq: seq };
}

/** Keep the newest `MAX_TELEMETRY_EVENTS` events (input already newest-first). */
export function capTelemetryEvents(events: TelemetryEvent[]): TelemetryEvent[] {
  return events.length > MAX_TELEMETRY_EVENTS
    ? events.slice(0, MAX_TELEMETRY_EVENTS)
    : events;
}

/** Per-class live counts from the current marks, sorted by count desc. */
export function summarizeMarks(marks: MarkEntity[]): ClassSummary[] {
  const acc = new Map<string, ClassSummary>();
  for (const m of marks) {
    const cur = acc.get(m.className);
    if (cur) {
      cur.count += 1;
    } else {
      acc.set(m.className, {
        className: m.className,
        count: 1,
        color: m.color,
      });
    }
  }
  return [...acc.values()].sort(
    (a, b) => b.count - a.count || a.className.localeCompare(b.className),
  );
}

function formatTime(at: number): string {
  return new Date(at).toLocaleTimeString([], {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function displayTrackId(trackId: string): string {
  return trackId.replace(/^mark:/, "");
}

/**
 * Live event feed derived from the store's mark list. Resets when the scene
 * changes. Safe to mount once per scene.
 */
export function useTelemetryFeed(sceneId: string): TelemetryEvent[] {
  const marks = useViewportMarksList();
  const [events, setEvents] = useState<TelemetryEvent[]>([]);
  const knownRef = useRef<Set<string>>(new Set());
  const seqRef = useRef(0);

  // New scene: drop the old feed.
  useEffect(() => {
    knownRef.current = new Set();
    seqRef.current = 0;
    setEvents([]);
  }, [sceneId]);

  useEffect(() => {
    const now = Date.now();
    const { events: fresh, known, nextSeq } = diffTelemetryBatch(
      knownRef.current,
      marks,
      now,
      seqRef.current,
    );
    knownRef.current = known;
    seqRef.current = nextSeq;
    if (fresh.length > 0) {
      setEvents((prev) => capTelemetryEvents([...fresh, ...prev]));
    }
  }, [marks]);

  return events;
}

function MqttChip({ connected }: { connected: boolean }): React.JSX.Element {
  return (
    <span
      className={`tl-chip tl-mqtt${connected ? " is-ok" : " is-bad"}`}
      role="status"
      aria-label={connected ? "MQTT connected" : "MQTT disconnected"}
      title={connected ? "MQTT connected" : "MQTT disconnected — live data paused"}
    >
      <span className="tl-dot" aria-hidden="true" />
      {connected ? "MQTT" : "MQTT off"}
    </span>
  );
}

export function Telemetry({ sceneId }: { sceneId: string }): React.JSX.Element {
  const connected = useMqttConnected();
  const marks = useViewportMarksList();
  const events = useTelemetryFeed(sceneId);
  const summary = summarizeMarks(marks);
  const feedRef = useRef<HTMLOListElement>(null);
  const [hovering, setHovering] = useState(false);

  // Newest entries land on top; keep the top in view unless the user is
  // reading (hover pauses auto-scroll).
  useEffect(() => {
    if (!hovering && feedRef.current) {
      feedRef.current.scrollTop = 0;
    }
  }, [events, hovering]);

  return (
    <section className="tl-pane" aria-label="Telemetry">
      <div className="tl-head">
        <MqttChip connected={connected} />
        <span className="tl-now" aria-live="polite">
          {marks.length === 0
            ? "no live objects"
            : `${marks.length} object${marks.length === 1 ? "" : "s"} now`}
        </span>
      </div>

      {summary.length > 0 && (
        <div className="tl-classes" aria-label="Live objects by class">
          {summary.map((c) => (
            <span key={c.className} className="tl-chip tl-class">
              <span
                className="tl-swatch"
                style={{ backgroundColor: c.color }}
                aria-hidden="true"
              />
              {c.className} · {c.count}
            </span>
          ))}
        </div>
      )}

      {events.length === 0 ? (
        <div className="tl-empty">
          <p>No telemetry yet.</p>
          <p className="tl-empty-hint">
            Track appearances and disappearances stream here while the MQTT
            client is connected.
          </p>
        </div>
      ) : (
        <ol
          ref={feedRef}
          className="tl-feed"
          aria-label="Track event feed"
          onMouseEnter={() => setHovering(true)}
          onMouseLeave={() => setHovering(false)}
        >
          {events.map((e) => (
            <li key={e.seq} className={`tl-event tl-${e.kind}`}>
              <span className="tl-time">{formatTime(e.at)}</span>
              <span className="tl-kind">
                {e.kind === "track-new" ? "appeared" : "lost"}
              </span>
              <span className="tl-track">
                {e.className} #{displayTrackId(e.trackId)}
              </span>
              {e.position && (
                <span className="tl-pos">
                  {e.position[0].toFixed(1)}, {e.position[1].toFixed(1)} m
                </span>
              )}
              {typeof e.speed === "number" && (
                <span className="tl-speed">{e.speed.toFixed(1)} m</span>
              )}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
