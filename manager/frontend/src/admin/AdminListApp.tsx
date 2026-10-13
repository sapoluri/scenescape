// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from "react";
import { PageHeader } from "../components/PageHeader";
import { ACTION_ICONS, iconForActionLabel } from "../components/actionIcons";
import "./AdminListApp.css";

export type AdminListAction = {
  label: string;
  href: string;
  tone?: "default" | "danger";
  id?: string;
};

export type AdminListCell = {
  text?: string;
  href?: string;
  /** When set, show a color chip beside the text (e.g. mark color). */
  swatch?: string;
};

export type AdminListRow = {
  id: string;
  cells: AdminListCell[];
  actions?: AdminListAction[];
};

export type AdminListBootstrap = {
  title: string;
  breadcrumbs?: { label: string; href?: string }[];
  primaryAction?: { label: string; href: string; id?: string };
  columns: string[];
  rows: AdminListRow[];
  emptyMessage?: string;
  isSuperuser?: boolean;
};

type Props = {
  bootstrap: AdminListBootstrap;
};

function CellContent({ cell }: { cell: AdminListCell }) {
  const label = cell.text || "—";
  const body = cell.href ? <a href={cell.href}>{label}</a> : label;
  if (!cell.swatch) {
    return body;
  }
  return (
    <span className="ss-admin-swatch-cell">
      <span
        className="ss-admin-swatch"
        style={{ backgroundColor: cell.swatch }}
        title={cell.swatch}
        aria-hidden="true"
      />
      {body}
    </span>
  );
}

/** Case-insensitive match across row cell text. Pure — unit tested. */
export function filterAdminRows(
  rows: AdminListRow[],
  query: string,
): AdminListRow[] {
  const q = query.trim().toLowerCase();
  if (!q) {
    return rows;
  }
  return rows.filter((row) =>
    row.cells.some((cell) => (cell.text || "").toLowerCase().includes(q)),
  );
}

/** Full list page: header + data table (Django supplies bootstrap JSON only). */
export function AdminListApp({ bootstrap }: Props) {
  const [query, setQuery] = useState("");
  const visible = useMemo(
    () => filterAdminRows(bootstrap.rows, query),
    [bootstrap.rows, query],
  );
  const showActions =
    Boolean(bootstrap.isSuperuser) &&
    bootstrap.rows.some((r) => (r.actions || []).length > 0);
  const titleLower = bootstrap.title.toLowerCase();
  const hasRows = bootstrap.rows.length > 0;
  const noMatch = hasRows && visible.length === 0;

  const actions = bootstrap.primaryAction ? (
    <a
      className="ss-btn ss-btn--primary"
      href={bootstrap.primaryAction.href}
      id={bootstrap.primaryAction.id}
    >
      {bootstrap.primaryAction.label}
    </a>
  ) : null;

  return (
    <div className="ss-admin-list hide-fullscreen">
      <PageHeader
        title={bootstrap.title}
        breadcrumbs={bootstrap.breadcrumbs || []}
        actions={actions}
      />
      {hasRows ? (
        <div className="ss-admin-search-row">
          <input
            type="search"
            className="ss-admin-search"
            placeholder={`Search ${titleLower}…`}
            value={query}
            onChange={(ev) => setQuery(ev.target.value)}
            aria-label={`Search ${titleLower}`}
          />
        </div>
      ) : null}
      <div className="ss-table-card ss-admin-table-card">
        {!hasRows ? (
          <p className="ss-table-empty">
            {bootstrap.emptyMessage || "Nothing to show."}
          </p>
        ) : noMatch ? (
          <p className="ss-table-empty">
            No {titleLower} match “{query.trim()}”.
          </p>
        ) : (
          <table className="table ss-admin-table">
            <thead>
              <tr>
                {bootstrap.columns.map((col) => (
                  <th key={col}>{col}</th>
                ))}
                {showActions ? <th>Actions</th> : null}
              </tr>
            </thead>
            <tbody>
              {visible.map((row) => (
                <tr key={row.id}>
                  {row.cells.map((cell, i) => (
                    <td key={`${row.id}-${i}`}>
                      <CellContent cell={cell} />
                    </td>
                  ))}
                  {showActions ? (
                    <td>
                      <div className="ss-table-actions">
                        {(row.actions || []).map((action) => {
                          const kind = iconForActionLabel(action.label);
                          const icon = kind ? ACTION_ICONS[kind] : null;
                          const danger = action.tone === "danger";
                          return (
                            <a
                              key={`${row.id}-${action.label}-${action.href}`}
                              className={`ss-table-action${
                                danger ? " is-danger" : ""
                              }${icon ? " is-icon" : ""}`}
                              href={action.href}
                              id={action.id}
                              title={action.label}
                              aria-label={action.label}
                            >
                              {icon ? (
                                <i
                                  className={`bi ${icon}`}
                                  aria-hidden="true"
                                />
                              ) : (
                                <span>{action.label}</span>
                              )}
                            </a>
                          );
                        })}
                      </div>
                    </td>
                  ) : null}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
