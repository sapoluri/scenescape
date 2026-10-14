// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useState } from "react";
import { useChildStatusStore } from "../mqtt/childStatusStore";
import { useViewportStore } from "../viewport/store";
import type { ChildSceneEntity, ViewportEntity, ViewportEntityType } from "../viewport/types";
import "./Outliner.css";

const GROUPS: {
  type: ViewportEntityType;
  label: string;
  addHref?: string;
  addLabel?: string;
}[] = [
  { type: "region", label: "Regions" },
  { type: "tripwire", label: "Tripwires" },
  { type: "camera", label: "Cameras", addHref: "?ss=cam-create", addLabel: "Add camera" },
  { type: "sensor", label: "Sensors", addHref: "?ss=sensor-create", addLabel: "Add sensor" },
  { type: "mark", label: "Tracked objects" },
  {
    type: "child",
    label: "Child scenes",
    addHref: "?ss=child-create",
    addLabel: "Link child",
  },
];

const TYPE_ICON: Record<ViewportEntityType, string> = {
  region: "bi-square",
  tripwire: "bi-slash-lg",
  camera: "bi-camera-video",
  sensor: "bi-broadcast",
  mark: "bi-person-walking",
  child: "bi-diagram-2",
};

const COLLAPSED_KEY = "ss-outliner-collapsed";

function readCollapsed(): Set<ViewportEntityType> {
  try {
    const raw = sessionStorage.getItem(COLLAPSED_KEY);
    if (!raw) {
      return new Set();
    }
    const list = JSON.parse(raw) as unknown;
    if (!Array.isArray(list)) {
      return new Set();
    }
    return new Set(list.filter((t): t is ViewportEntityType => typeof t === "string"));
  } catch {
    return new Set();
  }
}

function writeCollapsed(collapsed: Set<ViewportEntityType>): void {
  try {
    sessionStorage.setItem(COLLAPSED_KEY, JSON.stringify([...collapsed]));
  } catch {
    /* private mode / quota */
  }
}

function ChildStatusBadge({ entity }: { entity: ChildSceneEntity }) {
  const remoteId = entity.remoteChildId;
  const entry = useChildStatusStore((s) =>
    remoteId ? s.byRemoteId[remoteId] : undefined,
  );
  if (entity.childType !== "remote" || !remoteId) {
    return (
      <span className="ss-outliner-status ss-outliner-status--local" title="Local child">
        local
      </span>
    );
  }
  const status = entry?.status || "unknown";
  const title =
    status === "error"
      ? entry?.message || "Error"
      : status === "connected"
        ? "Connected"
        : status === "disconnected"
          ? "Disconnected"
          : "Status unknown";
  return (
    <span
      className={`ss-outliner-status ss-outliner-status--${status}`}
      title={title}
      aria-label={title}
    />
  );
}

function OutlinerRow({ entity }: { entity: ViewportEntity }) {
  const selectedId = useViewportStore((s) => s.selectedId);
  const select = useViewportStore((s) => s.select);
  const updateEntity = useViewportStore((s) => s.updateEntity);
  const selected = selectedId === entity.id;

  return (
    <div
      role="option"
      aria-selected={selected}
      tabIndex={0}
      className={`ss-outliner-row${selected ? " is-selected" : ""}`}
      onClick={() => select(entity.id)}
      onKeyDown={(ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          select(entity.id);
        }
      }}
    >
      <button
        type="button"
        className="ss-outliner-eye"
        aria-label={entity.visible ? `Hide ${entity.name}` : `Show ${entity.name}`}
        aria-pressed={entity.visible}
        onClick={(ev) => {
          ev.stopPropagation();
          updateEntity(entity.id, { visible: !entity.visible });
        }}
      >
        <i
          className={`bi ${entity.visible ? "bi-eye" : "bi-eye-slash"}`}
          aria-hidden="true"
        />
      </button>
      <i
        className={`bi ${TYPE_ICON[entity.type]} ss-outliner-icon`}
        aria-hidden="true"
      />
      <span className="ss-outliner-name" title={entity.name}>
        {entity.name}
      </span>
      {entity.type === "mark" && (
        <span
          className="ss-outliner-dot"
          style={{ background: entity.color }}
          aria-hidden="true"
        />
      )}
      {entity.type === "child" && <ChildStatusBadge entity={entity} />}
      {entity.type === "child" && entity.editHref ? (
        <a
          className="ss-outliner-edit"
          href={entity.editHref}
          title={`Edit ${entity.name}`}
          aria-label={`Edit ${entity.name}`}
          onClick={(ev) => ev.stopPropagation()}
        >
          <i className="bi bi-pencil" aria-hidden="true" />
        </a>
      ) : null}
    </div>
  );
}

export type OutlinerProps = {
  isSuperuser?: boolean;
  /** Show Preview control on the Cameras section (scene has cameras). */
  showPreviewToggle?: boolean;
  previewOpen?: boolean;
  onPreviewOpenChange?: (open: boolean) => void;
};

/**
 * Outliner: Blender-style entity tree. Groups mirror the spec entity list
 * (Regions → Tripwires → Cameras → Sensors → Tracked objects → Child scenes)
 * with badge counts and visibility eye toggles. Selection syncs both ways
 * through the viewport store (viewport click, strip click, outliner click).
 *
 * Cameras section can host a Preview toggle that opens the dock Live glance
 * split (see SinglePaneView).
 */
export function Outliner({
  isSuperuser = false,
  showPreviewToggle = false,
  previewOpen = false,
  onPreviewOpenChange,
}: OutlinerProps) {
  const entities = useViewportStore((s) => s.entities);
  const [collapsed, setCollapsed] = useState<Set<ViewportEntityType>>(readCollapsed);

  const toggleGroup = useCallback((type: ViewportEntityType) => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(type)) {
        next.delete(type);
      } else {
        next.add(type);
      }
      writeCollapsed(next);
      return next;
    });
  }, []);

  return (
    <div className="ss-outliner" role="listbox" aria-label="Scene outliner">
      {GROUPS.map(({ type, label, addHref, addLabel }) => {
        const items = Object.values(entities).filter((e) => e.type === type);
        if (items.length === 0 && !addHref) {
          return null;
        }
        const isCollapsed = collapsed.has(type);
        const bodyId = `ss-outliner-body-${type}`;
        return (
          <div
            key={type}
            className={`ss-outliner-group${isCollapsed ? " is-collapsed" : ""}`}
          >
            <div className="ss-outliner-group-header">
              <button
                type="button"
                className="ss-outliner-group-toggle"
                aria-expanded={!isCollapsed}
                aria-controls={bodyId}
                onClick={() => toggleGroup(type)}
                title={isCollapsed ? `Expand ${label}` : `Collapse ${label}`}
              >
                <i
                  className={`bi ${isCollapsed ? "bi-chevron-right" : "bi-chevron-down"}`}
                  aria-hidden="true"
                />
                <span className="ss-outliner-group-label">{label}</span>
                <span className="ss-outliner-badge">{items.length}</span>
              </button>
              {type === "camera" && showPreviewToggle && onPreviewOpenChange ? (
                <label
                  className={`ss-outliner-preview${previewOpen ? " is-on" : ""}`}
                  title={
                    previewOpen
                      ? "Hide live camera preview"
                      : "Show live camera preview in the dock"
                  }
                >
                  <input
                    type="checkbox"
                    className="ss-outliner-preview-input"
                    checked={previewOpen}
                    onChange={(ev) => onPreviewOpenChange(ev.target.checked)}
                    aria-label="Camera preview"
                  />
                  <i
                    className={`bi ${previewOpen ? "bi-camera-video-fill" : "bi-camera-video"}`}
                    aria-hidden="true"
                  />
                  <span className="ss-outliner-preview-label">Preview</span>
                </label>
              ) : null}
              {isSuperuser && addHref ? (
                <a
                  className="ss-outliner-add"
                  href={addHref}
                  title={addLabel}
                  aria-label={addLabel}
                  onClick={(ev) => ev.stopPropagation()}
                >
                  <i className="bi bi-plus" aria-hidden="true" />
                </a>
              ) : null}
            </div>
            <div
              id={bodyId}
              className="ss-outliner-group-body"
              hidden={isCollapsed}
              role="group"
              aria-label={label}
            >
              {items.map((e) => (
                <OutlinerRow key={e.id} entity={e} />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
