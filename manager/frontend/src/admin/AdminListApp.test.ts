// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { filterAdminRows, type AdminListRow } from "./AdminListApp";

const rows: AdminListRow[] = [
  {
    id: "1",
    cells: [{ text: "Dock Cam" }, { text: "Loading Dock" }],
  },
  {
    id: "2",
    cells: [{ text: "Yard PTZ" }, { text: "North Yard" }],
  },
  {
    id: "3",
    cells: [{ text: "dock staging" }, { text: "Loading Dock" }],
  },
];

describe("filterAdminRows", () => {
  it("returns all rows on empty query", () => {
    expect(filterAdminRows(rows, "")).toEqual(rows);
    expect(filterAdminRows(rows, "   ")).toEqual(rows);
  });

  it("matches case-insensitively across cell text", () => {
    const out = filterAdminRows(rows, "DOCK");
    expect(out.map((r) => r.id).sort()).toEqual(["1", "3"]);
  });

  it("matches scene / secondary columns", () => {
    expect(filterAdminRows(rows, "yard").map((r) => r.id)).toEqual(["2"]);
  });

  it("returns an empty list when nothing matches", () => {
    expect(filterAdminRows(rows, "zzz")).toEqual([]);
  });
});
