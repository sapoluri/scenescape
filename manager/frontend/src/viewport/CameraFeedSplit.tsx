// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useLayoutEffect, useRef, useState, type CSSProperties, type PointerEvent, type ReactNode } from "react";
import { clampDockWidth } from "../panels/dockSizing";

const WIDTH_KEY = "ss-camera-feed-width-px";

export function cameraFeedWidth(preferred: number | null, available: number): number {
  return clampDockWidth(preferred ?? available * 0.42, available);
}

function readWidth(): number | null {
  try {
    const width = Number(sessionStorage.getItem(WIDTH_KEY));
    return Number.isFinite(width) && width >= 96 ? width : null;
  } catch {
    return null;
  }
}

export function CameraFeedSplit({ active, children, feed }: {
  active: boolean;
  children: ReactNode;
  feed: ReactNode;
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [preferredWidth, setPreferredWidth] = useState(readWidth);
  const preferredWidthRef = useRef(preferredWidth);
  const [available, setAvailable] = useState(0);
  const drag = useRef<{ pointerId: number; x: number; width: number } | null>(null);
  const width = cameraFeedWidth(preferredWidth, available);

  useLayoutEffect(() => {
    if (!active || !hostRef.current) return;
    const host = hostRef.current;
    const measure = () => setAvailable(host.getBoundingClientRect().width);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(host);
    return () => {
      observer.disconnect();
      drag.current = null;
    };
  }, [active]);

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { pointerId: event.pointerId, x: event.clientX, width };
  };

  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const start = drag.current;
    if (!start || start.pointerId !== event.pointerId) return;
    const next = cameraFeedWidth(start.width + start.x - event.clientX, available);
    preferredWidthRef.current = next;
    setPreferredWidth(next);
  };

  const onPointerUp = (event: PointerEvent<HTMLDivElement>) => {
    if (drag.current?.pointerId !== event.pointerId) return;
    drag.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    if (preferredWidthRef.current != null) {
      try {
        sessionStorage.setItem(WIDTH_KEY, String(Math.round(preferredWidthRef.current)));
      } catch {
        return;
      }
    }
  };

  return (
    <div
      ref={hostRef}
      className={active ? "ss-viewport-split" : "ss-viewport-main"}
      style={active ? { "--ss-camera-feed-width": `${width}px` } as CSSProperties : undefined}
    >
      {children}
      {active ? (
        <div
          className="ss-sp-dock-resize ss-camera-feed-resize"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize camera pane"
          aria-valuenow={Math.round(width)}
          title="Resize camera pane"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
        >
          <i className="bi bi-three-dots-vertical" aria-hidden="true" />
        </div>
      ) : null}
      {active ? feed : null}
    </div>
  );
}