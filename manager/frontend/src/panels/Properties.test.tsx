// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { parseCameraOptics } from "../lib/cameraOptics";
import { useCalibrateStore } from "../viewport/calibrate/calibrateStore";
import { useViewportStore } from "../viewport/store";
import { Properties } from "./Properties";
import type { ManualIntrinsics } from "../viewport/calibrate/types";

vi.mock("../viewport/store", async importOriginal => {
  const original = await importOriginal<typeof import("../viewport/store")>();
  return { ...original, useViewportStore: Object.assign(
    (selector: (state: ReturnType<typeof original.getViewportState>) => unknown) => selector(original.getViewportState()),
    original.useViewportStore,
  ) };
});

vi.mock("../viewport/calibrate/calibrateStore", async importOriginal => {
  const original = await importOriginal<typeof import("../viewport/calibrate/calibrateStore")>();
  return { ...original, useCalibrateStore: Object.assign(
    (selector: (state: ReturnType<typeof original.getCalibrateState>) => unknown) => selector(original.getCalibrateState()),
    original.useCalibrateStore,
  ) };
});

const originalViewport = useViewportStore.getState();
afterEach(() => {
  useCalibrateStore.getState().exit();
  useViewportStore.setState(originalViewport);
});

describe("calibration manual-input controls", () => {
  const states = (["dfov", "focal", "neither"] as ManualIntrinsics[])
    .flatMap(mode => [true, false].map(manualDistortion => ({ mode, manualDistortion })));
  it.each(states)("renders $mode mode with manual distortion=$manualDistortion", ({ mode, manualDistortion }) => {
    useCalibrateStore.getState().enter({ cameraId: "camera", sensorId: "camera1", cameraName: "Camera 1" });
    useCalibrateStore.getState().setOptics(parseCameraOptics({}));
    useCalibrateStore.getState().setManualIntrinsics(mode);
    useCalibrateStore.getState().setOpticsLock("k1", manualDistortion);
    useViewportStore.setState({ selectedId: "camera", entities: {
      camera: { id: "camera", type: "camera", name: "Camera 1", visible: true,
        position: [0, 0, 3], rotation: [0, 0, 0], restRotation: [0, 0, 0], target: [0, 0, 0],
        fov: 60, posed: true, color: "#0a84ff" },
    } });
    const markup = renderToStaticMarkup(<Properties sceneName="Scene" sceneId="scene" />);
    expect(markup).toContain("Manually set:");
    expect(markup).toContain("(requires 6+ point pairs)");
    const radios = markup.match(/<input[^>]*type="radio"[^>]*>/g)!;
    expect(radios).toHaveLength(3);
    expect(radios.filter(input => input.includes('checked=""'))).toHaveLength(1);
    expect(radios.find(input => input.includes(`value="${mode}"`))).toContain('checked=""');
    const distortion = markup.match(/<input[^>]*aria-label="Manually set distortion"[^>]*>/)![0];
    expect(distortion.includes('checked=""')).toBe(manualDistortion);
    expect(markup.includes("(requires 8+ point pairs)")).toBe(!manualDistortion);
    expect(markup).not.toContain("ss-prop-optics-lock");
    const numericInputs = markup.match(/<input[^>]*type="number"[^>]*>/g)!;
    const labels = [...markup.matchAll(/<label for="([^"]+)">([^<]+)<\/label>/g)];
    const isReadOnly = (name: string) => {
      const id = labels.find(label => label[2] === name)![1];
      return /readonly=""/i.test(numericInputs.find(input => input.includes(`id="${id}"`))!);
    };
    expect(isReadOnly("Diagonal FOV (\u00b0)")).toBe(mode !== "dfov");
    expect(isReadOnly("fx")).toBe(mode !== "focal");
    expect(isReadOnly("fy")).toBe(mode !== "focal");
    expect(isReadOnly("cx")).toBe(true);
    expect(isReadOnly("cy")).toBe(true);
    for (const name of ["k1", "k2", "k3", "p1", "p2"]) {
      expect(isReadOnly(name)).toBe(!manualDistortion);
    }
    expect(numericInputs.every(input => !input.includes('disabled=""'))).toBe(true);
  });
});