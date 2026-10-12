// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from "react";
import { useChromeBootstrap } from "../chrome/useChromeBootstrap";
import { activeNavFromPath } from "../chrome/types";
import "./ViewportMenuBar.css";

type MenuId = "open" | "help" | "admin" | null;

type Props = {
  onOpenScene?: () => void;
  onOpenCameras?: () => void;
  onOpenSensors?: () => void;
  onOpenLibrary?: () => void;
  children?: React.ReactNode;
};

declare global {
  interface Window {
    ssTheme?: {
      initThemeToggle?: () => void;
    };
    jQuery?: (
      el: Element,
    ) => {
      modal: (action: string) => void;
    };
  }
}

/**
 * 3D viewport menu bar: Open (scenes/cameras/sensors + gallery), Library,
 * optional mid-bar controls (Mode/View/Overlays), Help, user/Admin, theme,
 * and SceneScape version on the right.
 *
 * Hard-contract nav ids (`#nav-scenes`, …) live here when the top navbar is
 * hidden on the workspace shell. Menu panels stay in the DOM (hidden when
 * closed) so Selenium presence checks keep working.
 */
export function ViewportMenuBar({
  onOpenScene,
  onOpenCameras,
  onOpenSensors,
  onOpenLibrary,
  children,
}: Props) {
  const chrome = useChromeBootstrap();
  const [menu, setMenu] = useState<MenuId>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const active = activeNavFromPath(
    typeof window !== "undefined" ? window.location.pathname : "/",
  );

  useEffect(() => {
    const onDoc = (ev: MouseEvent) => {
      if (!rootRef.current?.contains(ev.target as Node)) {
        setMenu(null);
      }
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  useEffect(() => {
    window.ssTheme?.initThemeToggle?.();
  }, [chrome]);

  const toggle = (id: MenuId) =>
    setMenu((prev) => (prev === id ? null : id));

  const openAbout = (ev: React.MouseEvent) => {
    ev.preventDefault();
    setMenu(null);
    const modal = document.getElementById("ss-about-modal");
    if (modal && typeof window.jQuery !== "undefined") {
      window.jQuery(modal).modal("show");
    }
  };

  const urls = chrome?.urls;
  const navClass = (id: string) =>
    `ss-vmenu-item${active === id ? " is-active" : ""}`;

  return (
    <div
      ref={rootRef}
      className="ss-viewport-bar ss-viewport-menubar"
      role="toolbar"
      aria-label="Viewport"
    >
      <div className={`ss-vmenu${menu === "open" ? " is-open" : ""}`}>
        <button
          type="button"
          className="ss-viewport-bar-btn"
          data-ss-open-menu
          aria-expanded={menu === "open"}
          aria-haspopup="true"
          onClick={() => toggle("open")}
          title="Open (⌘O also opens the scene gallery)"
        >
          <i className="bi bi-folder2-open" aria-hidden="true" />
          <span>Open</span>
          <i
            className="bi bi-caret-down-fill ss-vmenu-caret"
            aria-hidden="true"
          />
        </button>
        {urls ? (
          <div
            className="ss-vmenu-panel"
            role="menu"
            hidden={menu !== "open"}
          >
            <a
              className={navClass("scenes")}
              id="nav-scenes"
              href={urls.scenes}
              role="menuitem"
              onClick={(ev) => {
                if (onOpenScene) {
                  ev.preventDefault();
                  setMenu(null);
                  onOpenScene();
                }
              }}
            >
              Scenes
            </a>
            <a
              className={navClass("cameras")}
              id="nav-cameras"
              href={urls.cameras}
              role="menuitem"
              onClick={(ev) => {
                if (onOpenCameras) {
                  ev.preventDefault();
                  setMenu(null);
                  onOpenCameras();
                }
              }}
            >
              Cameras
            </a>
            <a
              className={navClass("sensors")}
              id="nav-sensors"
              href={urls.sensors}
              role="menuitem"
              onClick={(ev) => {
                if (onOpenSensors) {
                  ev.preventDefault();
                  setMenu(null);
                  onOpenSensors();
                }
              }}
            >
              Sensors
            </a>
            {chrome?.isKubernetes && urls.models ? (
              <a
                className={navClass("models")}
                id="nav-models"
                href={urls.models}
                role="menuitem"
              >
                Models
              </a>
            ) : null}
          </div>
        ) : null}
      </div>

      {onOpenLibrary ? (
        <button
          type="button"
          className="ss-viewport-bar-btn"
          id="ss-open-library"
          onClick={onOpenLibrary}
          title="Object library (B)"
        >
          <i className="bi bi-collection" aria-hidden="true" />
          <span>Library</span>
        </button>
      ) : null}

      {children}

      <div className="ss-viewport-bar-spacer" aria-hidden="true" />

      {chrome && urls ? (
        <>
          <div className={`ss-vmenu${menu === "help" ? " is-open" : ""}`}>
            <button
              type="button"
              className="ss-viewport-bar-btn"
              id="nav-help"
              aria-expanded={menu === "help"}
              aria-haspopup="true"
              onClick={() => toggle("help")}
            >
              Help
              <i
                className="bi bi-caret-down-fill ss-vmenu-caret"
                aria-hidden="true"
              />
            </button>
            <div
              className="ss-vmenu-panel ss-vmenu-panel--end"
              role="menu"
              hidden={menu !== "help"}
            >
              <a
                className="ss-vmenu-item"
                id="nav-docs"
                href={urls.docs}
                target="docs"
                rel="noopener noreferrer"
                role="menuitem"
              >
                Documentation
              </a>
              <a
                className="ss-vmenu-item"
                id="nav-support"
                href={urls.support}
                target="_blank"
                rel="noopener noreferrer"
                role="menuitem"
              >
                Support
              </a>
              <a
                className="ss-vmenu-item"
                href="#"
                id="nav-about"
                role="menuitem"
                onClick={openAbout}
              >
                About
              </a>
            </div>
          </div>

          <div className={`ss-vmenu${menu === "admin" ? " is-open" : ""}`}>
            <button
              type="button"
              className="ss-viewport-bar-btn"
              id="navbar-username"
              aria-expanded={menu === "admin"}
              aria-haspopup="true"
              onClick={() => toggle("admin")}
            >
              {chrome.username || "Account"}
              <i
                className="bi bi-caret-down-fill ss-vmenu-caret"
                aria-hidden="true"
              />
            </button>
            <div
              className="ss-vmenu-panel ss-vmenu-panel--end"
              role="menu"
              hidden={menu !== "admin"}
            >
              {chrome.isStaff ? (
                <>
                  <a
                    className="ss-vmenu-item"
                    id="nav-admin"
                    href={urls.admin}
                    target="admin"
                    role="menuitem"
                  >
                    Admin
                  </a>
                  <div className="ss-vmenu-sep" role="separator" />
                </>
              ) : null}
              <a
                className="ss-vmenu-item"
                id="nav-sign-out"
                href={urls.signOut}
                role="menuitem"
              >
                Log out
              </a>
            </div>
          </div>

          <div className="ss-theme-toggle-host ss-viewport-theme">
            <div
              className="ss-theme-toggle"
              id="ss-theme-toggle"
              role="group"
              aria-label="Color theme"
            >
              <button
                type="button"
                className="ss-theme-toggle__btn"
                data-ss-theme="light"
                aria-pressed="true"
              >
                Light
              </button>
              <button
                type="button"
                className="ss-theme-toggle__btn"
                data-ss-theme="dark"
                aria-pressed="false"
              >
                Dark
              </button>
            </div>
          </div>

          <div className="ss-viewport-brand" id="home" title={chrome.appName}>
            <span className="ss-viewport-brand-name">{chrome.appName}</span>
            <span className="ss-viewport-brand-version" id="navbar-version">
              Version {chrome.appVersion}
            </span>
          </div>
        </>
      ) : null}
    </div>
  );
}

/** Bottom-right Intel mark — fades in on hover (legacy 3D chrome). */
export function ViewportPoweredBy() {
  const chrome = useChromeBootstrap();
  if (!chrome?.urls) {
    return null;
  }
  return (
    <a
      className="ss-viewport-powered"
      href={chrome.urls.intel}
      target="_blank"
      rel="noopener noreferrer"
      title="Powered by Intel"
    >
      <span className="powered-by-label">Powered by</span>
      <img
        className="intel-logo"
        src={chrome.urls.intelLogo}
        alt="Intel"
      />
    </a>
  );
}
