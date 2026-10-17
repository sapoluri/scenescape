// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useId, useMemo, useRef, useState } from "react";
import { useAppToast } from "../components/ToastProvider";
import { rewriteBrokerUrl } from "../mqtt/client";
import { copyTextToClipboard } from "../scene/editors/copyText";
import { useMqttConnected } from "../scene/useLiveChrome";
import { useViewportMarksList } from "../viewport/marks";
import type { MarkEntity } from "../viewport/types";
import "./Telemetry.css";

/**
 * Telemetry pane: MQTT connection chip (toggles broker connect details),
 * live tracked-object summary, and a capped event feed of track
 * appearances/disappearances.
 *
 * Live data comes from the viewport scene-graph store (fed by the frozen
 * `ss-scene-objects` event in `viewport/marks.ts`) — no new plumbing.
 */

/** Regulated scene topic clients subscribe to for live detections. */
export function sceneDataTopic(sceneId: string): string {
  return sceneId
    ? `scenescape/regulated/scene/${sceneId}`
    : "scenescape/regulated/scene/";
}

/** Native MQTT URL for external clients (port 1883). */
export function mqttNativeUrl(hostname: string): string {
  const host = hostname.trim() || "localhost";
  return `mqtt://${host}:1883`;
}

export type BrokerConnectDetails = {
  websocketUrl: string;
  mqttUrl: string;
  topic: string;
};

/**
 * Display values for the Telemetry broker panel. WebSocket URL matches the
 * UI client (`rewriteBrokerUrl`); MQTT is hostname:1883 for native clients.
 */
export function brokerConnectDetails(
  wssConnection: string,
  sceneId: string,
  hostname: string = typeof window !== "undefined"
    ? window.location.hostname
    : "localhost",
): BrokerConnectDetails {
  return {
    websocketUrl: rewriteBrokerUrl(wssConnection || "wss://localhost/mqtt"),
    mqttUrl: mqttNativeUrl(hostname),
    topic: sceneDataTopic(sceneId),
  };
}

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

function MqttChip({
  connected,
  expanded,
  controlsId,
  onToggle,
}: {
  connected: boolean;
  expanded: boolean;
  controlsId: string;
  onToggle: () => void;
}): React.JSX.Element {
  return (
    <button
      type="button"
      className={`tl-chip tl-mqtt tl-mqtt-toggle${connected ? " is-ok" : " is-bad"}${expanded ? " is-open" : ""}`}
      aria-expanded={expanded}
      aria-controls={controlsId}
      title={
        expanded
          ? "Hide broker connection details"
          : connected
            ? "MQTT connected — show broker details"
            : "MQTT disconnected — show broker details"
      }
      onClick={onToggle}
    >
      <span className="tl-dot" aria-hidden="true" />
      {connected ? "MQTT" : "MQTT off"}
    </button>
  );
}

function BrokerCopyRow({
  label,
  value,
  onCopy,
}: {
  label: string;
  value: string;
  onCopy: (text: string) => void;
}): React.JSX.Element {
  return (
    <div className="tl-broker-row">
      <span className="tl-broker-label">{label}</span>
      <code className="tl-broker-value" title={value}>
        {value}
      </code>
      <button
        type="button"
        className="tl-broker-copy"
        title={`Copy ${label}`}
        aria-label={`Copy ${label}`}
        onClick={() => onCopy(value)}
      >
        <i className="bi bi-clipboard" aria-hidden="true" />
      </button>
    </div>
  );
}

function BrokerPanel({
  id,
  details,
}: {
  id: string;
  details: BrokerConnectDetails;
}): React.JSX.Element {
  const toast = useAppToast();
  const copy = (text: string) => {
    void copyTextToClipboard(text, toast.show);
  };
  return (
    <div className="tl-broker" id={id} role="region" aria-label="Broker connection">
      <BrokerCopyRow
        label="WebSocket"
        value={details.websocketUrl}
        onCopy={copy}
      />
      <BrokerCopyRow label="MQTT" value={details.mqttUrl} onCopy={copy} />
      <BrokerCopyRow label="Topic" value={details.topic} onCopy={copy} />
      <p className="tl-broker-note">
        Forward port <strong>1883</strong> in the deployment config if you need
        native MQTT clients. Browser and proxy clients use the WebSocket URL
        (typically via 443/80).
      </p>
    </div>
  );
}

export function Telemetry({
  sceneId,
  wssConnection = "",
}: {
  sceneId: string;
  /** Bootstrap WSS URL; rewritten for display to match the live client. */
  wssConnection?: string;
}): React.JSX.Element {
  const connected = useMqttConnected();
  const marks = useViewportMarksList();
  const events = useTelemetryFeed(sceneId);
  const summary = summarizeMarks(marks);
  const feedRef = useRef<HTMLOListElement>(null);
  const [hovering, setHovering] = useState(false);
  const [brokerOpen, setBrokerOpen] = useState(false);
  const brokerPanelId = useId();
  const details = useMemo(
    () => brokerConnectDetails(wssConnection, sceneId),
    [wssConnection, sceneId],
  );

  // Newest entries land on top; keep the top in view unless the user is
  // reading (hover pauses auto-scroll).
  useEffect(() => {
    if (!hovering && feedRef.current) {
      feedRef.current.scrollTop = 0;
    }
  }, [events, hovering]);

  useEffect(() => {
    if (!brokerOpen) {
      return;
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setBrokerOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [brokerOpen]);

  return (
    <section className="tl-pane" aria-label="Telemetry">
      <div className="tl-head">
        <MqttChip
          connected={connected}
          expanded={brokerOpen}
          controlsId={brokerPanelId}
          onToggle={() => setBrokerOpen((v) => !v)}
        />
        <span className="tl-now" aria-live="polite">
          {marks.length === 0
            ? "no live objects"
            : `${marks.length} object${marks.length === 1 ? "" : "s"} now`}
        </span>
      </div>

      {brokerOpen ? (
        <BrokerPanel id={brokerPanelId} details={details} />
      ) : null}

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
            client is connected. Click the MQTT chip for broker URLs.
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
