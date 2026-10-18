// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent as ReactMouseEvent,
} from "react";
import { useAppToast } from "../components/ToastProvider";
import { useSceneEventTelemetry } from "../mqtt/useSceneEventTelemetry";
import { copyTextToClipboard } from "../scene/editors/copyText";
import {
  pointInPolygon,
  useGeometryEventTelemetry,
} from "./geometryTelemetry";
import type { SceneSensorBootstrap } from "../scene/types";
import { useMqttConnected } from "../scene/useLiveChrome";
import { useViewportMarksList } from "../viewport/marks";
import { getViewportState, useViewportStore } from "../viewport/store";
import type { MarkEntity } from "../viewport/types";
import {
  brokerConnectDetails,
  capTelemetryEvents,
  defaultSelectedTopicKeys,
  filterTelemetryByTopics,
  filterTelemetryEvents,
  kindLabel,
  MAX_TELEMETRY_EVENTS,
  mqttNativeUrl,
  sceneDataTopic,
  sceneTopicCatalog,
  summarizeMarks,
  toggleTopicKey,
  TOPIC_KEY_REGION,
  TOPIC_KEY_REGULATED,
  TOPIC_KEY_SCENE_DATA,
  TOPIC_KEY_TRIPWIRE,
  topicKeyForKind,
  type BrokerConnectDetails,
  type ClassSummary,
  type TelemetryEvent,
  type TelemetryEventKind,
  type TopicCatalogEntry,
} from "./telemetryModel";
import "./Telemetry.css";

/**
 * Telemetry pane: MQTT chip (broker URLs + selectable scene topic catalog),
 * live tracked-object summary, and a capped feed of track / region / tripwire
 * events.
 */

export {
  brokerConnectDetails,
  capTelemetryEvents,
  defaultSelectedTopicKeys,
  filterTelemetryByTopics,
  filterTelemetryEvents,
  MAX_TELEMETRY_EVENTS,
  mqttNativeUrl,
  sceneDataTopic,
  sceneTopicCatalog,
  summarizeMarks,
  kindLabel,
  toggleTopicKey,
  TOPIC_KEY_REGION,
  TOPIC_KEY_REGULATED,
  TOPIC_KEY_SCENE_DATA,
  TOPIC_KEY_TRIPWIRE,
  topicKeyForKind,
};
export type {
  BrokerConnectDetails,
  ClassSummary,
  TelemetryEvent,
  TelemetryEventKind,
  TopicCatalogEntry,
};

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
        topicKey: TOPIC_KEY_REGULATED,
        trackId: m.id,
        className: m.className,
        position: [m.position[0], m.position[1]],
        speed,
      });
    }
  }
  for (const id of known) {
    if (!next.has(id)) {
      const prev = byId.get(id);
      events.push({
        seq: seq++,
        at: now,
        kind: "track-lost",
        topicKey: TOPIC_KEY_REGULATED,
        trackId: id,
        className: prev?.className ?? id.replace(/^mark:/, ""),
      });
    }
  }
  events.reverse();
  return { events, known: next, nextSeq: seq };
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

function eventRowKey(e: TelemetryEvent): string {
  return `${e.kind}-${e.seq}-${e.trackId}-${e.entityId ?? ""}-${e.at}`;
}

function feedRowLabel(e: TelemetryEvent): string {
  const who = e.className
    ? `${e.className}${e.trackId ? ` #${displayTrackId(e.trackId)}` : ""}`
    : e.entityName || e.entityId || "event";
  if (e.entityName && (e.kind === "region-enter" || e.kind === "region-exit")) {
    return `${who} · ${e.entityName}`;
  }
  if (e.entityName && e.kind === "tripwire-cross") {
    return `${who} · ${e.entityName}`;
  }
  return who;
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

function TopicFilterRow({
  entry,
  selected,
  onToggle,
  onCopy,
}: {
  entry: TopicCatalogEntry;
  selected: boolean;
  onToggle: () => void;
  onCopy: (text: string) => void;
}): React.JSX.Element {
  const onCopyClick = (ev: ReactMouseEvent) => {
    ev.stopPropagation();
    onCopy(entry.topic);
  };
  return (
    <div
      className={`tl-topic-row${selected ? " is-selected" : ""}`}
      role="option"
      aria-selected={selected}
      tabIndex={0}
      title={
        selected
          ? `Hide ${entry.label} in the feed`
          : `Show ${entry.label} in the feed`
      }
      onClick={onToggle}
      onKeyDown={(ev: ReactKeyboardEvent) => {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          onToggle();
        }
      }}
    >
      <span className="tl-broker-label">{entry.label}</span>
      <code className="tl-broker-value" title={entry.topic}>
        {entry.topic}
      </code>
      <button
        type="button"
        className="tl-broker-copy"
        title={`Copy ${entry.label}`}
        aria-label={`Copy ${entry.label}`}
        onClick={onCopyClick}
      >
        <i className="bi bi-clipboard" aria-hidden="true" />
      </button>
    </div>
  );
}

function BrokerPanel({
  id,
  details,
  topics,
  selectedTopicKeys,
  onToggleTopic,
}: {
  id: string;
  details: BrokerConnectDetails;
  topics: TopicCatalogEntry[];
  selectedTopicKeys: readonly string[];
  onToggleTopic: (key: string) => void;
}): React.JSX.Element {
  const toast = useAppToast();
  const copy = (text: string) => {
    void copyTextToClipboard(text, toast.show);
  };
  const selected = useMemo(
    () => new Set(selectedTopicKeys),
    [selectedTopicKeys],
  );
  return (
    <div
      className="tl-broker"
      id={id}
      role="region"
      aria-label="Broker connection"
    >
      <BrokerCopyRow
        label="WebSocket"
        value={details.websocketUrl}
        onCopy={copy}
      />
      <BrokerCopyRow label="MQTT" value={details.mqttUrl} onCopy={copy} />
      <p className="tl-broker-note">
        Forward port <strong>1883</strong> in the deployment config if you need
        native MQTT clients. Browser and proxy clients use the WebSocket URL
        (typically via 443/80). Click topics below to filter the feed
        (multi-select).
      </p>
      {topics.length > 0 ? (
        <div className="tl-broker-topics" aria-label="Scene MQTT topics">
          <div className="tl-broker-topics-head">Topics</div>
          <div
            className="tl-broker-topics-list"
            role="listbox"
            aria-multiselectable="true"
            aria-label="Filter feed by topic"
          >
            {topics.map((t) => (
              <TopicFilterRow
                key={t.key}
                entry={t}
                selected={selected.has(t.key)}
                onToggle={() => onToggleTopic(t.key)}
                onCopy={copy}
              />
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

export function Telemetry({
  sceneId,
  wssConnection = "",
  sensors = [],
}: {
  sceneId: string;
  /** Bootstrap WSS URL; rewritten for display to match the live client. */
  wssConnection?: string;
  sensors?: SceneSensorBootstrap[];
}): React.JSX.Element {
  const connected = useMqttConnected();
  const marks = useViewportMarksList();
  const trackEvents = useTelemetryFeed(sceneId);
  const mqttSceneEvents = useSceneEventTelemetry(sceneId);
  const geometryEvents = useGeometryEventTelemetry(sceneId);
  const events = useMemo(
    () =>
      capTelemetryEvents(
        [...trackEvents, ...mqttSceneEvents, ...geometryEvents].sort(
          (a, b) => b.at - a.at || b.seq - a.seq,
        ),
      ),
    [trackEvents, mqttSceneEvents, geometryEvents],
  );
  const summary = summarizeMarks(marks);
  const liveMarkIds = useMemo(
    () => new Set(marks.map((m) => m.id)),
    [marks],
  );
  const geometryIdKey = useViewportStore((s) => {
    const ids: string[] = [];
    for (const e of Object.values(s.entities)) {
      if (e.type === "region" || e.type === "tripwire") {
        ids.push(e.id);
      }
    }
    return ids.sort().join("\0");
  });
  const geometryIds = useMemo(
    () => new Set(geometryIdKey ? geometryIdKey.split("\0") : []),
    [geometryIdKey],
  );
  const feedRef = useRef<HTMLOListElement>(null);
  const [hovering, setHovering] = useState(false);
  const [brokerOpen, setBrokerOpen] = useState(false);
  const [classFilter, setClassFilter] = useState<string | null>(null);
  const [selectedTopicKeys, setSelectedTopicKeys] = useState<string[]>(
    defaultSelectedTopicKeys,
  );
  const brokerPanelId = useId();
  const details = useMemo(
    () => brokerConnectDetails(wssConnection),
    [wssConnection],
  );
  const topics = useMemo(
    () => sceneTopicCatalog(sceneId, sensors),
    [sceneId, sensors],
  );
  const visibleEvents = useMemo(
    () =>
      filterTelemetryEvents(
        filterTelemetryByTopics(events, selectedTopicKeys),
        classFilter,
      ),
    [events, selectedTopicKeys, classFilter],
  );

  useEffect(() => {
    setClassFilter(null);
    setSelectedTopicKeys(defaultSelectedTopicKeys());
  }, [sceneId]);

  useEffect(() => {
    if (!hovering && feedRef.current) {
      feedRef.current.scrollTop = 0;
    }
  }, [visibleEvents, hovering]);

  useEffect(() => {
    if (!brokerOpen) {
      return;
    }
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") {
        setBrokerOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [brokerOpen]);

  const toggleClassFilter = (name: string) => {
    setClassFilter((prev) => (prev === name ? null : name));
  };

  const onToggleTopic = (key: string) => {
    setSelectedTopicKeys((prev) => toggleTopicKey(prev, key));
  };

  const onFeedRowActivate = (e: TelemetryEvent) => {
    const state = getViewportState();
    if (e.trackId && liveMarkIds.has(e.trackId)) {
      state.select(e.trackId);
      return;
    }
    if (e.entityId) {
      const ent = state.entities[e.entityId];
      if (ent && (ent.type === "region" || ent.type === "tripwire")) {
        state.select(e.entityId);
      }
    }
  };

  const isSelectable = (e: TelemetryEvent): boolean => {
    if (e.trackId && liveMarkIds.has(e.trackId)) {
      return true;
    }
    return Boolean(e.entityId && geometryIds.has(e.entityId));
  };

  const regionOccupancy = useMemo(() => {
    let n = 0;
    for (const e of Object.values(getViewportState().entities)) {
      if (e.type !== "region" || !e.visible || e.points.length < 3) {
        continue;
      }
      for (const m of marks) {
        if (pointInPolygon(m.position[0], m.position[1], e.points)) {
          n += 1;
        }
      }
    }
    return n;
  }, [marks, geometryIdKey]);

  const emptyFilterHint =
    selectedTopicKeys.length === 0
      ? "Select one or more topics in the MQTT panel to show events."
      : classFilter
        ? "Clear the class filter or select more topics to see events."
        : selectedTopicKeys.includes(TOPIC_KEY_REGION) ||
            selectedTopicKeys.includes(TOPIC_KEY_TRIPWIRE)
          ? `Waiting for enter/exit or crossings. Objects currently in regions: ${regionOccupancy}.`
          : "Select region / tripwire topics, or wait for matching events.";

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
        <BrokerPanel
          id={brokerPanelId}
          details={details}
          topics={topics}
          selectedTopicKeys={selectedTopicKeys}
          onToggleTopic={onToggleTopic}
        />
      ) : null}

      {summary.length > 0 && (
        <div className="tl-classes" aria-label="Live objects by class">
          {classFilter ? (
            <button
              type="button"
              className="tl-chip tl-class tl-class-all"
              onClick={() => setClassFilter(null)}
              title="Show all classes"
            >
              All
            </button>
          ) : null}
          {summary.map((c) => (
            <button
              key={c.className}
              type="button"
              className={`tl-chip tl-class${classFilter === c.className ? " is-active" : ""}`}
              aria-pressed={classFilter === c.className}
              title={
                classFilter === c.className
                  ? `Clear ${c.className} filter`
                  : `Filter feed to ${c.className}`
              }
              onClick={() => toggleClassFilter(c.className)}
            >
              <span
                className="tl-swatch"
                style={{ backgroundColor: c.color }}
                aria-hidden="true"
              />
              {c.className} · {c.count}
            </button>
          ))}
        </div>
      )}

      {events.length === 0 ? (
        <div className="tl-empty">
          <p>No telemetry yet.</p>
          <p className="tl-empty-hint">
            Track and region/tripwire events stream here while MQTT is
            connected. Click the MQTT chip for broker URLs and topics.
          </p>
        </div>
      ) : visibleEvents.length === 0 ? (
        <div className="tl-empty">
          <p>
            {classFilter
              ? `No ${classFilter} events for selected topics.`
              : "No events for selected topics."}
          </p>
          <p className="tl-empty-hint">{emptyFilterHint}</p>
        </div>
      ) : (
        <ol
          ref={feedRef}
          className="tl-feed"
          aria-label="Track event feed"
          onMouseEnter={() => setHovering(true)}
          onMouseLeave={() => setHovering(false)}
        >
          {visibleEvents.map((e) => {
            const selectable = isSelectable(e);
            return (
              <li
                key={eventRowKey(e)}
                className={`tl-event tl-${e.kind}${selectable ? " is-selectable" : ""}`}
                title={
                  selectable
                    ? `Select ${feedRowLabel(e)} on the map`
                    : undefined
                }
                {...(selectable
                  ? {
                      role: "button" as const,
                      tabIndex: 0,
                      onClick: () => onFeedRowActivate(e),
                      onKeyDown: (ev: ReactKeyboardEvent) => {
                        if (ev.key === "Enter" || ev.key === " ") {
                          ev.preventDefault();
                          onFeedRowActivate(e);
                        }
                      },
                    }
                  : {})}
              >
                <span className="tl-time">{formatTime(e.at)}</span>
                <span className="tl-kind">{kindLabel(e.kind)}</span>
                <span className="tl-track">{feedRowLabel(e)}</span>
                {e.position && (
                  <span className="tl-pos">
                    {e.position[0].toFixed(1)}, {e.position[1].toFixed(1)} m
                  </span>
                )}
                {typeof e.speed === "number" && (
                  <span className="tl-speed">{e.speed.toFixed(1)} m</span>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
