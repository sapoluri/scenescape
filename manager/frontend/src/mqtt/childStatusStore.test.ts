// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it } from "vitest";
import {
  getChildStatusState,
  reduceChildStatusPayload,
  useChildStatusStore,
} from "./childStatusStore";

describe("reduceChildStatusPayload", () => {
  it("maps connected / disconnected", () => {
    expect(reduceChildStatusPayload("connected")).toEqual({
      status: "connected",
    });
    expect(reduceChildStatusPayload("disconnected")).toEqual({
      status: "disconnected",
    });
  });

  it("treats isConnected probe as disconnected until a real status", () => {
    expect(reduceChildStatusPayload("isConnected")).toEqual({
      status: "disconnected",
    });
  });

  it("maps other text to error with message", () => {
    expect(reduceChildStatusPayload("broker down")).toEqual({
      status: "error",
      message: "broker down",
    });
  });

  it("maps empty to unknown", () => {
    expect(reduceChildStatusPayload("")).toEqual({ status: "unknown" });
  });
});

describe("useChildStatusStore", () => {
  beforeEach(() => {
    useChildStatusStore.getState().clear();
  });

  it("stores status by remote id", () => {
    getChildStatusState().setStatus("remote-a", { status: "connected" });
    expect(useChildStatusStore.getState().byRemoteId["remote-a"]).toEqual({
      status: "connected",
    });
  });
});
