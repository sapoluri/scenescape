// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import type { AdminListBootstrap } from "../admin/AdminListApp";
import { adaptListBootstrapForWorkspace } from "./OpenAdminListDialog";

const BASE: AdminListBootstrap = {
  title: "Cameras",
  primaryAction: {
    label: "+ New Camera",
    href: "/cameras/?ss=cam-create",
    id: "new-camera",
  },
  columns: ["Camera Name", "Camera ID", "Scene"],
  rows: [
    {
      id: "1",
      cells: [
        { text: "Cam A" },
        { text: "cam_a" },
        {
          text: "Retail",
          href: "/3bc091c7-e449-46a0-9540-29c499bca18c/?from=cam-list",
        },
      ],
      actions: [
        {
          label: "Manage",
          href: "/cameras/?ss=calibrate-cam&id=1",
        },
        {
          label: "Delete",
          href: "/cam/delete/1/",
          tone: "danger",
        },
      ],
    },
    {
      id: "2",
      cells: [
        { text: "Orphan" },
        { text: "orphan" },
        { text: "--" },
      ],
      actions: [
        {
          label: "Edit",
          href: "/cameras/?ss=cam-edit&id=orphan",
        },
      ],
    },
  ],
  isSuperuser: true,
};

describe("adaptListBootstrapForWorkspace", () => {
  it("rewrites create to same-page sheet query", () => {
    const adapted = adaptListBootstrapForWorkspace(BASE, "cameras");
    expect(adapted.primaryAction?.href).toBe("?ss=cam-create");
    expect(adapted.primaryAction?.id).toBe("new-camera");
  });

  it("sends calibrate to the owning scene", () => {
    const adapted = adaptListBootstrapForWorkspace(BASE, "cameras");
    expect(adapted.rows[0].actions?.[0].href).toBe(
      "/3bc091c7-e449-46a0-9540-29c499bca18c/?ss=calibrate-cam&id=1",
    );
  });

  it("keeps delete hrefs and rewrites orphan edit", () => {
    const adapted = adaptListBootstrapForWorkspace(BASE, "cameras");
    expect(adapted.rows[0].actions?.[1].href).toBe("/cam/delete/1/");
    expect(adapted.rows[1].actions?.[0].href).toBe("?ss=cam-edit&id=orphan");
  });
});
