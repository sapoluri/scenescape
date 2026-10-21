// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CameraFeedSplit, cameraFeedWidth } from "./CameraFeedSplit";

afterEach(() => vi.unstubAllGlobals());

describe("shared live/calibration camera pane", () => {
  it("starts at the existing 42 percent proportion and clamps while retaining a preferred width", () => {
    expect(cameraFeedWidth(null, 1000)).toBe(420);
    expect(cameraFeedWidth(500, 1000)).toBe(500);
    expect(cameraFeedWidth(500, 500)).toBeLessThan(250);
    expect(cameraFeedWidth(500, 1000)).toBe(500);
    expect(cameraFeedWidth(500, 0)).toBe(0);
  });

  it.each(["Live", "Calibration"])("renders the same dotted pointer-only divider for %s", mode => {
    vi.stubGlobal("sessionStorage", { getItem: vi.fn().mockReturnValue("500") });
    const markup = renderToStaticMarkup(<CameraFeedSplit active feed={<div>{mode}</div>}>
      <div>Scene</div>
    </CameraFeedSplit>);
    expect(sessionStorage.getItem).toHaveBeenCalledWith("ss-camera-feed-width-px");
    expect(markup).toContain('class="ss-sp-dock-resize ss-camera-feed-resize"');
    expect(markup).toContain('aria-label="Resize camera pane"');
    expect(markup).toContain('title="Resize camera pane"');
    expect(markup).toContain("bi-three-dots-vertical");
    expect(markup).toContain(mode);
    expect(markup).not.toContain("tabindex");
    expect(markup.indexOf("Scene")).toBeLessThan(markup.indexOf('role="separator"'));
    expect(markup.indexOf('role="separator"')).toBeLessThan(markup.indexOf(`>${mode}<`));
  });

  it("keeps the scene mounted without a divider or feed when closed", () => {
    const markup = renderToStaticMarkup(<CameraFeedSplit active={false} feed={<div>Camera feed</div>}>
      <div>Scene</div>
    </CameraFeedSplit>);
    expect(markup).toContain('class="ss-viewport-main"');
    expect(markup).toContain("Scene");
    expect(markup).not.toContain('role="separator"');
    expect(markup).not.toContain("Camera feed");
  });

  it("works when session storage is unavailable", () => {
    vi.stubGlobal("sessionStorage", { getItem: () => { throw new Error("Storage unavailable"); } });
    expect(() => renderToStaticMarkup(<CameraFeedSplit active feed={null}>Scene</CameraFeedSplit>)).not.toThrow();
  });
});