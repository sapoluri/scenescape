// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { useAppToast } from "../components/ToastProvider";
import { useChildStatusStore } from "../mqtt/childStatusStore";
import { parseDeleteTarget, restDeleteTarget } from "../lib/restDelete";
import type { RestError } from "../lib/rest";
import type {
  SceneCameraBootstrap,
  SceneChildBootstrap,
  SceneSensorBootstrap,
} from "../scene/types";
import {
  removeRegionFromModel,
  removeTripwireFromModel,
} from "../viewport/geometrySync";
import { getViewportState, useViewportStore } from "../viewport/store";
import type {
  ChildSceneEntity,
  ViewportEntity,
  ViewportEntityType,
} from "../viewport/types";
import "./Outliner.css";

const GROUPS: {
  type: ViewportEntityType;
  label: string;
  addHref?: string;
  addLabel?: string;
}[] = [
  { type: "map", label: "Scene map" },
  { type: "region", label: "Regions" },
  { type: "tripwire", label: "Tripwires" },
  {
    type: "camera",
    label: "Cameras",
    addHref: "?ss=cam-create",
    addLabel: "Add camera",
  },
  {
    type: "sensor",
    label: "Sensors",
    addHref: "?ss=sensor-create",
    addLabel: "Add sensor",
  },
  { type: "mark", label: "Tracked objects" },
  {
    type: "child",
    label: "Child scenes",
    addHref: "?ss=child-create",
    addLabel: "Link child",
  },
];

const TYPE_ICON: Record<ViewportEntityType, string> = {
  map: "bi-box",
  region: "bi-square",
  tripwire: "bi-slash-lg",
  camera: "bi-camera-video",
  sensor: "bi-broadcast",
  mark: "bi-person-walking",
  child: "bi-diagram-2",
};

const COLLAPSED_KEY = "ss-outliner-collapsed";

type DeleteTarget = {
  id: string;
  name: string;
  type: ViewportEntityType;
  deleteUrl: string | null;
  isDraft: boolean;
};

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
    return new Set(
      list.filter((t): t is ViewportEntityType => typeof t === "string"),
    );
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
      <span
        className="ss-outliner-status ss-outliner-status--local"
        title="Local child"
      >
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

function resolveDeleteUrl(
  entity: ViewportEntity,
  cameras: SceneCameraBootstrap[],
  sensors: SceneSensorBootstrap[],
  children: SceneChildBootstrap[],
  isSuperuser: boolean,
): string | null {
  if (entity.type === "camera") {
    if (!isSuperuser) {
      return null;
    }
    return cameras.find((c) => c.id === entity.id)?.deleteUrl ?? null;
  }
  if (entity.type === "sensor") {
    if (!isSuperuser) {
      return null;
    }
    return sensors.find((s) => s.id === entity.id)?.deleteUrl ?? null;
  }
  if (entity.type === "child") {
    if (!isSuperuser) {
      return null;
    }
    return (
      entity.deleteUrl ||
      children.find(
        (c) =>
          c.id === entity.id ||
          c.restUid === entity.id ||
          c.restUid === entity.restUid,
      )?.deleteUrl ||
      null
    );
  }
  return null;
}

function canDeleteEntity(
  entity: ViewportEntity,
  cameras: SceneCameraBootstrap[],
  sensors: SceneSensorBootstrap[],
  children: SceneChildBootstrap[],
  isSuperuser: boolean,
): boolean {
  if (entity.type === "mark" || entity.type === "map") {
    return false;
  }
  if (entity.type === "region" || entity.type === "tripwire") {
    return true;
  }
  if (entity.id.startsWith("draft-")) {
    return true;
  }
  return Boolean(
    resolveDeleteUrl(entity, cameras, sensors, children, isSuperuser),
  );
}

function OutlinerRow({
  entity,
  revealed,
  deletable,
  onDelete,
}: {
  entity: ViewportEntity;
  revealed: boolean;
  deletable: boolean;
  onDelete: (entity: ViewportEntity) => void;
}) {
  const selectedId = useViewportStore((s) => s.selectedId);
  const select = useViewportStore((s) => s.select);
  const updateEntity = useViewportStore((s) => s.updateEntity);
  const selected = selectedId === entity.id;
  const rowRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (selected && revealed) {
      rowRef.current?.scrollIntoView({ block: "nearest" });
    }
  }, [selected, revealed]);

  return (
    <div
      ref={rowRef}
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
        aria-label={
          entity.visible ? `Hide ${entity.name}` : `Show ${entity.name}`
        }
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
      {deletable ? (
        <button
          type="button"
          className="ss-outliner-delete"
          title={
            entity.type === "child"
              ? `Unlink ${entity.name}`
              : `Delete ${entity.name}`
          }
          aria-label={
            entity.type === "child"
              ? `Unlink ${entity.name}`
              : `Delete ${entity.name}`
          }
          onClick={(ev) => {
            ev.stopPropagation();
            onDelete(entity);
          }}
        >
          <i className="bi bi-trash" aria-hidden="true" />
        </button>
      ) : null}
    </div>
  );
}

export type OutlinerProps = {
  isSuperuser?: boolean;
  authToken?: string;
  cameras?: SceneCameraBootstrap[];
  sensors?: SceneSensorBootstrap[];
  children?: SceneChildBootstrap[];
  onCamerasChange?: Dispatch<SetStateAction<SceneCameraBootstrap[]>>;
  onSensorsChange?: Dispatch<SetStateAction<SceneSensorBootstrap[]>>;
  onChildrenChange?: Dispatch<SetStateAction<SceneChildBootstrap[]>>;
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
 * split (see SinglePaneView). Delete lives on rows (hover / selected); the
 * keyboard Delete key dispatches the same confirm flow.
 */
export function Outliner({
  isSuperuser = false,
  authToken = "",
  cameras = [],
  sensors = [],
  children = [],
  onCamerasChange,
  onSensorsChange,
  onChildrenChange,
  showPreviewToggle = false,
  previewOpen = false,
  onPreviewOpenChange,
}: OutlinerProps) {
  const toast = useAppToast();
  const entities = useViewportStore((s) => s.entities);
  const selectedType = useViewportStore((s) => {
    const id = s.selectedId;
    return id ? s.entities[id]?.type : undefined;
  });
  const [collapsed, setCollapsed] =
    useState<Set<ViewportEntityType>>(readCollapsed);
  const [deleteTarget, setDeleteTarget] = useState<DeleteTarget | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);

  useEffect(() => {
    if (!selectedType) {
      return;
    }
    setCollapsed((prev) => {
      if (!prev.has(selectedType)) {
        return prev;
      }
      const next = new Set(prev);
      next.delete(selectedType);
      writeCollapsed(next);
      return next;
    });
  }, [selectedType]);

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

  const requestDelete = useCallback(
    (entity: ViewportEntity) => {
      if (
        !canDeleteEntity(entity, cameras, sensors, children, isSuperuser)
      ) {
        return;
      }
      setDeleteTarget({
        id: entity.id,
        name: entity.name,
        type: entity.type,
        deleteUrl: resolveDeleteUrl(
          entity,
          cameras,
          sensors,
          children,
          isSuperuser,
        ),
        isDraft: entity.id.startsWith("draft-"),
      });
    },
    [cameras, sensors, children, isSuperuser],
  );

  // Keyboard Delete / Backspace (from useViewportTools) and any other caller.
  useEffect(() => {
    const onRequest = (ev: Event) => {
      const id = (ev as CustomEvent<{ id?: string }>).detail?.id;
      if (!id) {
        return;
      }
      const entity = getViewportState().entities[id];
      if (!entity) {
        return;
      }
      requestDelete(entity);
    };
    window.addEventListener("ss-request-entity-delete", onRequest);
    return () =>
      window.removeEventListener("ss-request-entity-delete", onRequest);
  }, [requestDelete]);

  const confirmDelete = useCallback(async () => {
    if (!deleteTarget) {
      return;
    }
    setDeleteBusy(true);
    try {
      const { id, type, deleteUrl, isDraft, name } = deleteTarget;
      if (
        (type === "camera" || type === "sensor" || type === "child") &&
        deleteUrl &&
        !isDraft
      ) {
        if (!authToken) {
          throw new Error("Authentication required to delete");
        }
        const target = parseDeleteTarget(deleteUrl);
        if (!target) {
          throw new Error("Unrecognized delete URL");
        }
        await restDeleteTarget(authToken, target);
        getViewportState().removeEntity(id);
        if (type === "camera") {
          onCamerasChange?.((prev) => prev.filter((c) => c.id !== id));
        } else if (type === "sensor") {
          onSensorsChange?.((prev) => prev.filter((s) => s.id !== id));
        } else {
          onChildrenChange?.((prev) =>
            prev.filter((c) => c.id !== id && c.restUid !== id),
          );
        }
        toast.show(
          type === "child" ? `Unlinked ${name}` : `Deleted ${name}`,
          "ok",
        );
        setDeleteTarget(null);
        if (type === "camera" || type === "sensor") {
          window.location.reload();
        }
        return;
      }

      getViewportState().commitHistory();
      if (type === "region") {
        removeRegionFromModel(id);
      } else if (type === "tripwire") {
        removeTripwireFromModel(id);
      }
      getViewportState().removeEntity(id);
      toast.show(
        type === "child" ? `Unlinked ${name}` : `Deleted ${name}`,
        "ok",
      );
      setDeleteTarget(null);
    } catch (err) {
      toast.show((err as RestError).message || "Delete failed", "bad");
      setDeleteBusy(false);
    } finally {
      setDeleteBusy(false);
    }
  }, [
    deleteTarget,
    authToken,
    onCamerasChange,
    onSensorsChange,
    onChildrenChange,
    toast,
  ]);

  const deleteTitle =
    deleteTarget?.type === "child"
      ? "Unlink child scene?"
      : deleteTarget
        ? `Delete ${deleteTarget.type}?`
        : "Delete?";
  const deleteConfirmLabel =
    deleteTarget?.type === "child" ? "Unlink" : "Delete";

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
                <OutlinerRow
                  key={e.id}
                  entity={e}
                  revealed={!isCollapsed}
                  deletable={canDeleteEntity(
                    e,
                    cameras,
                    sensors,
                    children,
                    isSuperuser,
                  )}
                  onDelete={requestDelete}
                />
              ))}
            </div>
          </div>
        );
      })}
      <ConfirmDialog
        open={deleteTarget != null}
        title={deleteTitle}
        confirmLabel={deleteConfirmLabel}
        danger
        busy={deleteBusy}
        onConfirm={() => void confirmDelete()}
        onCancel={() => {
          if (!deleteBusy) {
            setDeleteTarget(null);
          }
        }}
      >
        {deleteTarget?.type === "child" ? (
          <p>
            Are you sure you want to unlink <strong>{deleteTarget.name}</strong>{" "}
            from this scene?
          </p>
        ) : (
          <>
            <p>
              Are you sure you want to delete{" "}
              <strong>{deleteTarget?.name}</strong>?
            </p>
            {deleteTarget &&
            (deleteTarget.type === "camera" ||
              deleteTarget.type === "sensor") &&
            !deleteTarget.isDraft ? (
              <p>This action cannot be undone.</p>
            ) : null}
          </>
        )}
      </ConfirmDialog>
    </div>
  );
}
