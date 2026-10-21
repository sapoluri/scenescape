// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("expired session handling", () => {
  async function setup() {
    vi.resetModules();
    const end = vi.fn();
    const replace = vi.fn();
    const browser = {
      location: { pathname: "/scene/", replace },
      ssMqttClient: { end },
    };
    vi.stubGlobal("window", browser);
    const { handleExpiredSession } = await import("./session");
    return { handleExpiredSession, end, replace, browser };
  }

  it("disconnects MQTT and logs out once for an invalidated authenticated token", async () => {
    const { handleExpiredSession, end, replace, browser } = await setup();
    handleExpiredSession(new Response(null, { status: 401 }), "old-token");
    handleExpiredSession(new Response(null, { status: 401 }), "old-token");
    expect(end).toHaveBeenCalledExactlyOnceWith(true);
    expect(browser.ssMqttClient).toBeUndefined();
    expect(replace).toHaveBeenCalledExactlyOnceWith("/sign_out/");
  });

  it.each([403, 404, 500, 503])("does not log out on HTTP %s", async status => {
    const { handleExpiredSession, replace } = await setup();
    handleExpiredSession(new Response(null, { status }), "token");
    expect(replace).not.toHaveBeenCalled();
  });

  it("does not redirect unauthenticated requests or the sign-in page", async () => {
    const { handleExpiredSession, replace, browser } = await setup();
    handleExpiredSession(new Response(null, { status: 401 }), "");
    browser.location.pathname = "/sign_in/";
    handleExpiredSession(new Response(null, { status: 401 }), "old-token");
    expect(replace).not.toHaveBeenCalled();
  });

  it("still logs out when MQTT teardown throws", async () => {
    const { handleExpiredSession, end, replace } = await setup();
    end.mockImplementation(() => { throw new Error("Already closed"); });
    expect(() => handleExpiredSession(new Response(null, { status: 401 }), "token")).not.toThrow();
    expect(replace).toHaveBeenCalledWith("/sign_out/");
  });

  it("checks on visibility/focus events without interval polling and removes listeners", async () => {
    await setup();
    const browser = Object.assign(new EventTarget(), { location: { pathname: "/scene/", replace: vi.fn() } });
    const document = Object.assign(new EventTarget(), { visibilityState: "visible" });
    vi.stubGlobal("window", browser);
    vi.stubGlobal("document", document);
    const request = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", request);
    const { watchSessionExpiry } = await import("./session");
    const stop = watchSessionExpiry("token");
    await vi.waitFor(() => expect(request).toHaveBeenCalledOnce());
    browser.dispatchEvent(new Event("focus"));
    document.dispatchEvent(new Event("visibilitychange"));
    expect(request).toHaveBeenCalledOnce();
    expect(request).toHaveBeenCalledWith("/api/v1/session", { credentials: "same-origin", cache: "no-store" });
    stop();
    browser.dispatchEvent(new Event("focus"));
    expect(request).toHaveBeenCalledOnce();
  });

  it("logs out when a visible tab discovers its expired cookie session", async () => {
    const { replace, browser } = await setup();
    const events = Object.assign(new EventTarget(), browser);
    vi.stubGlobal("window", events);
    vi.stubGlobal("document", Object.assign(new EventTarget(), { visibilityState: "visible" }));
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 401 })));
    const { watchSessionExpiry } = await import("./session");
    const stop = watchSessionExpiry("still-valid-api-token");
    await vi.waitFor(() => expect(replace).toHaveBeenCalledWith("/sign_out/"));
    stop();
  });

  it.each(["json", "form"])("handles invalid tokens on existing %s API requests", async kind => {
    const { end, replace } = await setup();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ detail: "Invalid token." }),
      { status: 401, headers: { "Content-Type": "application/json" } },
    )));
    const { restJson, restForm } = await import("./rest");
    const request = kind === "json"
      ? restJson("GET", "/scene/example", "stale-token")
      : restForm("POST", "/scene", "stale-token", new FormData());
    await expect(request).rejects.toMatchObject({ status: 401 });
    expect(end).toHaveBeenCalledExactlyOnceWith(true);
    expect(replace).toHaveBeenCalledWith("/sign_out/");
  });

  it("does not log out for a session-check network failure", async () => {
    const { replace, browser } = await setup();
    vi.stubGlobal("window", Object.assign(new EventTarget(), browser));
    vi.stubGlobal("document", Object.assign(new EventTarget(), { visibilityState: "visible" }));
    const request = vi.fn().mockRejectedValue(new TypeError("Network unavailable"));
    vi.stubGlobal("fetch", request);
    const { watchSessionExpiry } = await import("./session");
    const stop = watchSessionExpiry("token");
    await vi.waitFor(() => expect(request).toHaveBeenCalledOnce());
    expect(replace).not.toHaveBeenCalled();
    stop();
  });
});