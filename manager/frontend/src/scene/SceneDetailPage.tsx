// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useState } from "react";
import { useSceneMqtt, useCameraStripMqtt } from "../mqtt";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { ToastProvider } from "../components/ToastProvider";
import { LegacyConfirmHost } from "../components/LegacyConfirmHost";
import { SinglePaneView } from "../panels/SinglePaneView";
import { GeometryBootstrap } from "./GeometryBootstrap";
import { SceneWorkspaceSheets } from "../sheets/SceneWorkspaceSheets";
import { deleteViaRest } from "../lib/restDelete";
import { persistSceneGeometry } from "../lib/roiPersist";
import { useMqttConnected, useCameraRates } from "./useLiveChrome";
import type { SceneDetailBootstrap } from "./types";
import "./SceneDetailPage.css";

type Props = {
  bootstrap: SceneDetailBootstrap;
};

function sceneDetailBack(urls: SceneDetailBootstrap["urls"]): {
  href: string;
  label: string;
} {
  const from =
    typeof window !== "undefined"
      ? new URLSearchParams(window.location.search).get("from")
      : null;
  if (from === "cam-list" && urls.camList) {
    return { href: urls.camList, label: "Cameras" };
  }
  if (from === "sensor-list" && urls.sensorList) {
    return { href: urls.sensorList, label: "Sensors" };
  }
  return { href: urls.scenesHome, label: "Scenes" };
}

function SceneDetailInner({ bootstrap }: Props) {
  const { scene, urls, isSuperuser } = bootstrap;
  const [sceneDeleteOpen, setSceneDeleteOpen] = useState(false);
  const [sceneDeleteBusy, setSceneDeleteBusy] = useState(false);
  const [sceneDeleteError, setSceneDeleteError] = useState<string | null>(null);
  const mqttConnected = useMqttConnected();
  const cameraRates = useCameraRates();
  const [cameras, setCameras] = useState(bootstrap.cameras);
  const [sensors, setSensors] = useState(bootstrap.sensors || []);
  const [sceneRate, setSceneRate] = useState("--");

  useSceneMqtt({
    sceneId: scene.id,
    wssConnection: scene.wssConnection || "",
    enabled: true,
  });
  useCameraStripMqtt(true);

  // The 3D viewport saves through window.ssPersistGeometry (frozen bridge).
  useEffect(() => {
    const authToken = bootstrap.authToken || "";
    const sid = scene.id;
    const persist = (options?: { preferHidden?: boolean } | string[]) =>
      persistSceneGeometry(
        authToken,
        sid,
        Array.isArray(options) ? undefined : options,
      );
    window.ssPersistGeometry = persist;
    return () => {
      if (window.ssPersistGeometry === persist) {
        delete window.ssPersistGeometry;
      }
    };
  }, [bootstrap.authToken, scene.id]);

  useEffect(() => {
    const setSceneRateCb = (hz: string) => setSceneRate(hz || "--");
    window.ssSceneTelemetry = {
      ...(window.ssSceneTelemetry || {}),
      setSceneRate: setSceneRateCb,
    };
    const onSceneRate = (ev: Event) => {
      const detail = (ev as CustomEvent<{ hz: string }>).detail;
      if (detail?.hz !== undefined) {
        setSceneRateCb(detail.hz);
      }
    };
    const onClear = () => setSceneRate("--");
    window.addEventListener("ss-scene-rate", onSceneRate);
    window.addEventListener("ss-telemetry-clear", onClear);
    return () => {
      window.removeEventListener("ss-scene-rate", onSceneRate);
      window.removeEventListener("ss-telemetry-clear", onClear);
    };
  }, []);

  const confirmSceneDelete = useCallback(async () => {
    if (!urls.sceneDelete) {
      return;
    }
    setSceneDeleteBusy(true);
    setSceneDeleteError(null);
    try {
      await deleteViaRest(
        urls.sceneDelete,
        bootstrap.authToken || "",
        urls.scenesHome || "/",
      );
    } catch (e) {
      setSceneDeleteBusy(false);
      setSceneDeleteError(e instanceof Error ? e.message : "Delete failed");
    }
  }, [bootstrap.authToken, urls.sceneDelete, urls.scenesHome]);

  const back = sceneDetailBack(urls);
  const deleteImpact = bootstrap.deleteImpact;

  return (
    <div className="ss-scene-detail ss-scene-detail--single-pane">
      <div className="ss-scene-chrome" id="ss-scene-chrome">
        <div className="ss-scene-chrome-map">
          <div className="ss-scene-chrome-start hide-fullscreen">
            <div className="ss-scene-header-actions" role="group" aria-label="Scene">
              <a className="ss-scene-header-back" href={back.href}>
                <i className="bi bi-arrow-left" aria-hidden="true" />
                {back.label}
              </a>
              <h2 className="ss-page-title" id="scene_name">
                {scene.name}
              </h2>
              <a
                className="btn btn-secondary btn-sm"
                id="export-scene"
                href="#"
                title={`Export ${scene.name}`}
              >
                <i className="bi bi-box-arrow-up" aria-hidden="true" />
              </a>
              {isSuperuser ? (
                <a
                  className="btn btn-secondary btn-sm"
                  id="scene-edit"
                  href="?ss=scene-manage"
                  title={`Edit ${scene.name}`}
                  aria-label={`Edit ${scene.name}`}
                >
                  <i className="bi bi-pencil" aria-hidden="true" />
                </a>
              ) : null}
              {isSuperuser && urls.sceneDelete ? (
                <button
                  type="button"
                  className="btn btn-secondary btn-sm ss-icon-btn--danger"
                  id="scene-delete"
                  title={`Delete ${scene.name}`}
                  aria-label={`Delete ${scene.name}`}
                  onClick={() => {
                    setSceneDeleteError(null);
                    setSceneDeleteOpen(true);
                  }}
                >
                  <i className="bi bi-trash" aria-hidden="true" />
                </button>
              ) : null}
              <span
                id="mqtt_status"
                className={`scene-detail-mqtt-pill${mqttConnected ? " connected" : ""}`}
                title={mqttConnected ? "MQTT connected" : "MQTT disconnected"}
                data-ss-mqtt={mqttConnected ? "connected" : "disconnected"}
                aria-label={mqttConnected ? "MQTT connected" : "MQTT disconnected"}
              >
                <i className="bi bi-arrow-down-up" aria-hidden="true" />
              </span>
            </div>
          </div>
          <div className="ss-scene-chrome-end hide-fullscreen">
            <div className="scene-rate ss-scene-rate telemetry-hide">
              Rate: &nbsp;<span id="scene-rate">{sceneRate}</span>&nbsp; Hz
            </div>
          </div>
        </div>
      </div>
      <div className="ss-workspace-body ss-workspace-body--single">
        <SinglePaneView
          sceneId={scene.id}
          sceneName={scene.name}
          assetMarkColors={bootstrap.assetMarkColors}
          cameras={cameras}
          sensors={sensors}
          cameraRates={cameraRates}
          authToken={bootstrap.authToken}
          isSuperuser={isSuperuser}
        />
      </div>
      <GeometryBootstrap
        sceneId={scene.id}
        initialRegions={bootstrap.regions || []}
        initialTripwires={bootstrap.tripwires || []}
      />
      <SceneWorkspaceSheets
        sceneId={scene.id}
        authToken={bootstrap.authToken}
        isSuperuser={isSuperuser}
        isKubernetes={Boolean(bootstrap.isKubernetes)}
        scenes={bootstrap.scenes || []}
        cameras={cameras}
        sensors={sensors}
        onCamerasChange={setCameras}
        onSensorsChange={setSensors}
        onChildrenChange={() => {}}
        mapUrl={null}
        mapScale={scene.scale}
      />
      <ConfirmDialog
        open={sceneDeleteOpen}
        title="Delete scene?"
        confirmLabel="Delete scene"
        danger
        busy={sceneDeleteBusy}
        onConfirm={confirmSceneDelete}
        onCancel={() => {
          if (!sceneDeleteBusy) {
            setSceneDeleteOpen(false);
          }
        }}
      >
        <p>
          Are you sure you want to delete <strong>{scene.name}</strong>?
        </p>
        <p>If you proceed, the following cannot be undone:</p>
        <ul>
          <li>The scene will be permanently deleted</li>
          {(deleteImpact?.sensors ?? 0) > 0 ? (
            <li>
              {deleteImpact?.sensors} camera(s) and/or sensor(s) will be
              orphaned
            </li>
          ) : null}
          {(deleteImpact?.regions ?? 0) > 0 ? (
            <li>{deleteImpact?.regions} region(s) will be deleted</li>
          ) : null}
          {(deleteImpact?.tripwires ?? 0) > 0 ? (
            <li>{deleteImpact?.tripwires} tripwire(s) will be deleted</li>
          ) : null}
        </ul>
        {sceneDeleteError ? (
          <p className="ss-confirm-error">{sceneDeleteError}</p>
        ) : null}
      </ConfirmDialog>
    </div>
  );
}

export function SceneDetailPage({ bootstrap }: Props) {
  return (
    <ToastProvider>
      <LegacyConfirmHost>
        <SceneDetailInner bootstrap={bootstrap} />
      </LegacyConfirmHost>
    </ToastProvider>
  );
}
