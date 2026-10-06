// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useViewportStore } from "../viewport/store";
import type { ViewportEntity, ViewportEntityType } from "../viewport/types";
import "./Outliner.css";

const GROUPS: { type: ViewportEntityType; label: string; addHref?: string; addLabel?: string }[] = [
  { type: "region", label: "Regions" },
  { type: "tripwire", label: "Tripwires" },
  { type: "camera", label: "Cameras", addHref: "?ss=cam-create", addLabel: "Add camera" },
  { type: "sensor", label: "Sensors", addHref: "?ss=sensor-create", addLabel: "Add sensor" },
  { type: "mark", label: "Tracked objects" },
  { type: "child", label: "Child scenes" },
];

const TYPE_ICON: Record<ViewportEntityType, string> = {
  region: "bi-square",
  tripwire: "bi-slash-lg",
  camera: "bi-camera-video",
  sensor: "bi-broadcast",
  mark: "bi-person-walking",
  child: "bi-diagram-2",
};

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
    </div>
  );
}

/**
 * Outliner: Blender-style entity tree. Groups mirror the spec entity list
 * (Regions → Tripwires → Cameras → Sensors → Tracked objects → Child scenes)
 * with badge counts and visibility eye toggles. Selection syncs both ways
 * through the viewport store (viewport click, strip click, outliner click).
 */
export function Outliner({ isSuperuser = false }: { isSuperuser?: boolean }) {
  const entities = useViewportStore((s) => s.entities);

  return (
    <div className="ss-outliner" role="listbox" aria-label="Scene outliner">
      {GROUPS.map(({ type, label, addHref, addLabel }) => {
        const items = Object.values(entities).filter((e) => e.type === type);
        if (items.length === 0 && !addHref) {
          return null;
        }
        return (
          <div key={type} className="ss-outliner-group">
            <div className="ss-outliner-group-header">
              <span>{label}</span>
              <span className="ss-outliner-badge">{items.length}</span>
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
            {items.map((e) => (
              <OutlinerRow key={e.id} entity={e} />
            ))}
          </div>
        );
      })}
    </div>
  );
}
