// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from "vitest";
import { assertWorkspaceWriteAllowed, useEditModeStore, workspaceEditingAllowed } from "./editMode";
import { restForm, restJson } from "./rest";
import { registerSceneForAutocalibration, startCameraAutocalibration } from "./autocalibration";
import { startMeshGeneration } from "./meshGeneration";
import { useViewportStore } from "../viewport/store";
import { enterCameraCalibrate, useCalibrateStore } from "../viewport/calibrate";

afterEach(() => {
  useEditModeStore.setState({ workspaceActive: false, isSuperuser: false, canEdit: false, editEnabled: true });
  vi.unstubAllGlobals();
});

describe("workspace edit permission", () => {
  it("refuses editing tool and calibration requests but permits navigation and measurement", () => {
    useEditModeStore.getState().enterWorkspace(false);
    useViewportStore.getState().setTool("live");
    useViewportStore.getState().setTool("camera");
    expect(useViewportStore.getState().activeTool).toBe("live");
    useViewportStore.getState().setTool("measure");
    expect(useViewportStore.getState().activeTool).toBe("measure");
    useViewportStore.getState().setOrtho(true);
    expect(useViewportStore.getState().ortho).toBe(true);
    enterCameraCalibrate({ cameraId: "camera", sensorId: "camera1", cameraName: "Camera" });
    expect(useCalibrateStore.getState().active).toBe(false);
  });

  it("keeps viewers read-only even if they try enabling editing", () => {
    useEditModeStore.getState().enterWorkspace(false);
    useEditModeStore.getState().setEditing(true);
    expect(useEditModeStore.getState().canEdit).toBe(false);
    expect(workspaceEditingAllowed()).toBe(false);
    expect(() => assertWorkspaceWriteAllowed("GET")).not.toThrow();
    expect(() => assertWorkspaceWriteAllowed("POST")).toThrow();
    expect(useEditModeStore.getState().editEnabled).toBe(true);
  });

  it("lets superusers switch between editing and viewing", () => {
    useEditModeStore.getState().enterWorkspace(true);
    expect(workspaceEditingAllowed()).toBe(true);
    useEditModeStore.getState().setEditing(false);
    expect(workspaceEditingAllowed()).toBe(false);
    useEditModeStore.getState().setEditing(true);
    expect(workspaceEditingAllowed()).toBe(true);
  });

  it("tracks open write surfaces so view-only cannot discard an active editor", () => {
    useEditModeStore.getState().enterWorkspace(true);
    useEditModeStore.getState().setEditSurface("camera-sheet", true);
    expect(useEditModeStore.getState().editSurfaceIds).toEqual(["camera-sheet"]);
    useEditModeStore.getState().setEditSurface("camera-sheet", false);
    expect(useEditModeStore.getState().editSurfaceIds).toEqual([]);
  });

  it("blocks JSON and form writes before any network call", async () => {
    useEditModeStore.getState().enterWorkspace(false);
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);
    await expect(restJson("PUT", "/camera/test", "test-token", {})).rejects.toMatchObject({ status: 403 });
    await expect(restForm("POST", "/import", "test-token", new FormData())).rejects.toMatchObject({ status: 403 });
    await expect(startMeshGeneration("test-token", "scene")).rejects.toMatchObject({ status: 403 });
    await expect(registerSceneForAutocalibration("scene")).rejects.toMatchObject({ status: 403 });
    await expect(startCameraAutocalibration("camera", "image", [])).rejects.toMatchObject({ status: 403 });
    expect(fetch).not.toHaveBeenCalled();
  });
});