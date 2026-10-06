<!--
SPDX-FileCopyrightText: (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Phase 1.0 Inventory — Single-Pane 3D UI Rebuild

Read-only inventory for the single-pane rebuild (build plan:
`docs/design/single-pane-ui.md`, §7). Branch: `feature/single-pane-ui`
(cut from `upstream/feature/django-swappable`).

Sources: `.github/skills/manager-ui/SKILL.md`,
`docs/design/manager-ui-backend-contract.md`,
`docs/adr/0019-host-independent-manager-ui.md`,
`manager/frontend/src` (as of commit `7764d1b1`).

---

## 1. Frozen contracts

From `.github/skills/manager-ui/SKILL.md` — "Hard contracts (freeze)".
Stable DOM ids, `window` APIs, and events that UI tests and hybrid bridges
depend on. **Change only with matching test updates in the same PR.**
Do not rename `#ss-admin-list-root`, table action hrefs, or map ids.
Do not add new required template sibling ids or new `window.ss*` APIs.

### 1.1 Map host (built by `ensureSceneDetailDom`, not Django HTML)

| Name | Kind | Where defined | Notes |
|---|---|---|---|
| `#ss-map-host` | DOM id | `src/lib/ensureSceneDetailDom.ts` | Map parking / adopt root |
| `#map` | DOM id | `ensureSceneDetailDom` | Map image container |
| `#svgout` | DOM id | `ensureSceneDetailDom` | Visible map SVG (React when `ssUseReactMap`; Snap keeps `#svgout-snap`) |
| `#scale` | DOM id | `ensureSceneDetailDom` | Scene scale |
| `#scene` | DOM id | template hidden node | Scene id / metadata |
| `#fullscreen`, `#show-trails`, `#show-telemetry`, `#coloring-switch` | DOM ids | `ensureSceneDetailDom` | Map chrome; centered via `#ss-map-toggles-slot` inside `#map-controls` |
| `#ss-scene-chrome` / `.ss-scene-header-actions` | DOM id / class | scene chrome | Back + title + export/3d/edit/delete + rate |
| `#id_rois`, `#tripwires` | DOM ids | bootstrap → hidden inputs | Hidden geometry JSON |
| `#id_child_rois`, `#child_tripwires`, `#child_sensors` | DOM ids | bootstrap → hidden inputs | Child overlay JSON |

### 1.2 Toolbar / tabs

`#new-roi`, `#save-rois`, `#empty-new-roi`, `#new-tripwire`,
`#save-trips`, `#empty-new-tripwire`, `#live-view`,
`#ss-tab-regions`, `#ss-tab-tripwires`, `#ss-tab-cameras`,
`#ss-tab-sensors`, `#ss-tab-children`, `#ss-tab-mqtt`,
`#regions`, `#trips`, `#roi-fields`, `#tripwire-fields`,
`#no-regions`, `#no-tripwires`, `#mqtt_status`, `#broker`,
`#scene-edit`, `#3d-view`.
(Pane ids `#cameras`, `#trips`, … stay hard contracts; React-owned panels
live in `SceneSidePanel`.)

### 1.3 ROI / tripwire cards

`#form-roi_{uuid}`, `#form-tripwire_{uuid}`; `.roi-title`,
`.tripwire-title`, `.roi-remove`, `.tripwire-remove`,
`.roi-volumetric`, `.roi-height`, `.roi-buffer`, `.green_min`,
`.yellow_min`, `.red_min`, `.range_max`; SVG `g.roi` / `g.tripwire`,
`adding-roi` / `adding-tripwire`.

### 1.4 Cameras / sensors / children

| Name | Kind | Notes |
|---|---|---|
| `#ss-cameras-mount`, `.snapshot-image[topic]`, `#rate-{sensorId}`, `.camera-card` | DOM | Camera strip (React `useCameraStripMqtt` drives `.snapshot-image`) |
| `#ss-sensors-mount`, `.singleton`, `.area-json`, `.sensor-id` | DOM | Sensors (`SensorLayer`) |
| `#ss-children-mount`, `.child-card`, `#mqtt_status_remote_{id}` | DOM | Children; tab badges via `publishSceneTabCounts` / `ss-tab-counts` |
| `?ss=child-edit&id={restUid}` | URL query | ChildSheet; `restUid` = local UUID, `remote_child_id`, or ChildScene pk |
| `a[id^='sensor_calibrate_']` | DOM | Scene-tab Manage → calibrate |
| `?ss=calibrate-sensor&id={pk}`, `a.ss-table-action[title='Manage']` | URL / DOM | List → calibrate |
| `#ss-sensor-calibrate-form`, `#ss-sensor-cal-area` (`scene`/`circle`/`poly`) | DOM | Calibrate area |
| `#ss-sensor-cal-cx`, `#ss-sensor-cal-cy`, `#ss-sensor-cal-r`, `#ss-sensor-cal-pts` | DOM | Circle / poly fields |
| `svg.ss-sensor-area-map`, `.ss-sensor-area-coverage`, `.ss-sensor-area-handle` | DOM | Area preview map |
| `button[form='ss-sensor-calibrate-form']` (`.ss-btn--dirty` when unsaved) | DOM | Calibrate Save |

### 1.5 3D chrome (legacy viewport)

`#scene-detail-link`, `#scene-detail-button` → `sceneDetail`;
`#2d-button`, `#3d-button` → orthographic / perspective toggle.

### 1.6 Navbar (React chrome island: `ui/chrome.js`, `#ss-chrome-root` + `ss-chrome-bootstrap`)

`#nav-help`, `#nav-docs`, `#nav-support`, `#nav-about` /
`#ss-about-modal`, `#nav-admin` (staff, under account menu),
`#navbar-username`, `#ss-theme-toggle`, `#home`, `#navbar-version`,
`#nav-scenes`, `#nav-cameras`, `#nav-sensors`, `#nav-models` (K8s),
`#nav-object-library`, `#nav-sign-out`, `#login-submit` (sign-in form).

### 1.7 `window` APIs — React freeze (hybrid still required)

`window.ssMap`, `window.ssRoiEditors`, `window.ssPersistGeometry`
(sscape `saveRois` still calls persist),
`window.ssAttachSceneMqttClient` / `window.ssMqttClient` (shared transport),
`window.ssToast` / `window.ssConfirm` (legacy JS only),
`window.ssSceneTelemetry`,
`window.ssSyncRoiColorSectors` / `window.ssReapplyRoiColors`.
React call sites go through `src/lib/legacyBridge.ts`; under
`ssUseReactMap`, fit/number go only through `window.ssMap`.

**Not** React freeze (sscape-only — do not call from React):
`fitSceneMapDisplay`, `numberRois` / `numberTripwires`,
`stringifyRois` / `stringifyTripwires`, `ssEnsureMqttScene`,
`ssRefreshCameraSnapshots`, `ssDrawSingletonSensors`,
`ssRemoveSingletonSensor`, `getRoiValues`, `saveRois`.

### 1.8 Custom events (frozen)

`ss-roi-form-add`, `ss-tripwire-form-add`, `ss-scene-rate`,
`ss-camera-rate`, `ss-telemetry-clear`, `ss-map-host-ready`,
`ss-tab-counts`, `ss-scene-tab`, `ss-roi-dirty`, `ss-trip-dirty`,
`ss-mqtt-status`, `ss-mqtt-connected`, `ss-singleton`,
`ss-scene-objects`, `ss-show-trails`.

Live-mark pipeline: legacy `sscape.js` dispatches `ss-scene-objects`
with `CustomEvent<{ objects?: SceneObjectMark[] }>`; React
`MarksLayer` listens and plots. (`src/scene/map/MarksLayer.tsx:39-50`)

### 1.9 Island mount roots (frozen)

| Island | Mount root | Bootstrap |
|---|---|---|
| Chrome (nav/about/theme) | `#ss-chrome-root` | `GET /api/v1/ui-bootstrap/?page=chrome` |
| Scenes home | `#ss-scenes-home-app` | `?page=scenes` |
| Scene detail | `#ss-scene-detail-root` | `?page=scene&id=` |
| Admin lists | `#ss-admin-list-root` | `?page=cameras\|sensors\|assets` |
| List sheets | query-driven, no dedicated root | `?page=list-sheets&id=cam\|sensor\|asset` |
| Models directory | `#ss-models-directory-root` | `?page=models` |
| Sign-in | `#ss-sign-in-root` | `?page=sign-in` |
| Destructive actions | creates `#ss-destructive-actions-root` | none |

---

## 2. Backend contracts

Long-term boundary per backend-contract doc: **HTTP + bootstrap JSON + MQTT**.
Base REST path: `/api/v1`. Client: `manager/frontend/src/lib/rest.ts`.
Auth: `Authorization: Token <authToken>` (portable); session+CSRF only for
rare legacy forms (transitional). New island code must use `lib/rest.ts`,
`lib/session.ts`, `lib/restDelete.ts`.

### 2.1 REST endpoints the UI calls

| Area | Method + path | Purpose |
|---|---|---|
| Camera | `POST /api/v1/camera` | Create camera |
| Camera | `GET\|PUT /api/v1/camera/{uid}` | Read / update camera |
| Sensor | `POST /api/v1/sensor` | Create sensor |
| Sensor | `GET\|PUT\|DELETE /api/v1/sensor/{uid}` | Read / update / delete sensor |
| Child | `POST /api/v1/child` | Create child scene link |
| Child | `GET\|PUT /api/v1/child/{uid}` | Read / update child (`{uid}` = pk, `child_id`, or `remote_child_id`) |
| Child | `POST /api/v1/childscene/preview-geospatial-transform/` | Preview geospatial transform |
| Scene | `POST /api/v1/scene` | Create scene |
| Scene | `GET\|PUT /api/v1/scene/{uid}` | Read / update scene |
| Scene | `GET /api/v1/scenes` | List scenes |
| Scene | `POST /api/v1/import-scene/` | Import scene |
| Region | `GET /api/v1/regions?scene=` | List regions for scene |
| Region | `POST /api/v1/region` | Create region |
| Region | `PUT\|DELETE /api/v1/region/{uid}` | Update / delete region |
| Tripwire | `GET /api/v1/tripwires?scene=` | List tripwires for scene |
| Tripwire | `POST /api/v1/tripwire` | Create tripwire |
| Tripwire | `PUT\|DELETE /api/v1/tripwire/{uid}` | Update / delete tripwire |
| Asset (Object Library) | `POST /api/v1/asset` | Create Asset3D row |
| Asset | `GET\|PUT /api/v1/asset/{uid}` | Read / update Asset3D row |
| Model directory | `GET\|POST\|DELETE /api/v1/model-directory/` | Models tree (Token) |
| Mesh | `POST /api/v1/scene/{uuid}/generate-mesh/`, `GET …/generate-mesh-status/` | Mesh generation (Token) |
| Mapping | `GET /mapping-service/status/` | Mapping service status (Token) |
| Deletes | `DELETE /api/v1/{camera\|sensor\|scene\|child\|asset}/{uid}` | Entity deletes via `lib/restDelete.ts` (ManageThing) |
| UI bootstrap | `GET /api/v1/ui-bootstrap/?page=chrome\|scenes\|scene\|cameras\|sensors\|assets\|models\|list-sheets\|sign-in&id=` | Host-independent bootstrap (Session or Token) |

Sheet deep links: `?ss=<action>&id=` — actions
`cam-create|cam-edit|sensor-create|sensor-edit|child-create|child-edit|
scene-create|scene-edit|scene-manage|scene-import|asset-create|asset-edit|
calibrate-cam|calibrate-sensor` (`src/lib/sheetQuery.ts`).

### 2.2 MQTT topics

Transport: WSS broker URL from `scene.wssConnection` (or `wss://{host}/mqtt`).
App prefix: `scenescape`. Topic fragments in
`manager/frontend/src/mqtt/topics.ts` — keep in sync with
`manager/static/js/constants.js`.

| Direction | Topic pattern | Role |
|---|---|---|
| Sub | `scenescape/regulated/scene/{sceneId}` | Scene telemetry + tracked-object marks |
| Sub | `scenescape/event/+/` + `{sceneId}` + `/+/+` | Region / tripwire events |
| Sub | `scenescape/sys/child/status/+` | Remote child connectivity |
| Sub | `scenescape/image/camera/+` | Live camera strip frames |
| Sub | `scenescape/image/calibration/camera/{sensorId}` | Calibrate |
| Pub | `scenescape/cmd/camera/{sensorId}` | `getimage` / calibration commands |
| Pub | `scenescape/sys/child/status/{remoteId}` | `isConnected` probe |

UI-created event topics: `scenescape/event/region/{sceneId}/{uuid}/count`,
`scenescape/event/tripwire/{sceneId}/{uuid}/objects`.

Message shapes:
- **Marks** (`SceneObjectMark`, `src/scene/map/liveMarks.ts`):
  `{ id: string|number, type?: string, translation?: number[] | {x?, y?}, tag_id?: string|number, persistent_data?: Record<string,unknown> }`.
  Translation is in meters; mark colors come from bootstrap `assetMarkColors`
  (asset name → color).
- **Camera frames** (`src/mqtt/useCameraStripMqtt.ts`): JSON
  `{ image?: string }` (base64 JPEG) → applied as
  `data:image/jpeg;base64,…` to `img.snapshot-image`; strip polls via
  `publish(scenescape/cmd/camera/{sensorId}, "getimage")`.

### 2.3 Bootstrap JSON shapes (`GET /api/v1/ui-bootstrap/`)

- **chrome**: `authenticated`, `username`, `isStaff`, `isKubernetes`,
  `appName`, `appVersion`, `appGitCommit`, `docsVersion`,
  `urls{home,scenes,cameras,sensors,models,assets,admin,signOut,docs,support,intel,intelLogo}`,
  `activeNav?` (`scenes|cameras|sensors|models|assets`).
- **scenes**: `authToken`, `isSuperuser`,
  `scenes[]{id,name,georeferenced,thumbnailUrl,mapUrl,detailUrl,detail3dUrl,manageUrl,deleteUrl?,counts{sensors,regions,tripwires}}`.
- **scene** (id = scene UUID): `scene{id,name,scale,mapUrl,thumbnailUrl,wssConnection?,outputLla?,georeferenced?}`,
  `cameras[]{id,sensorId,name,calibrateHref,cmdTopic,deleteUrl}`,
  `sensors[]{id,sensorId,name,iconUrl,areaJson,calibrateHref,editHref,deleteUrl}`,
  `children[]{id,name,childType,remoteChildId,detailUrl,thumbnailUrl,mapUrl,restUid,editHref,deleteUrl}`,
  `regions`, `tripwires` (geometry JSON arrays),
  `assetMarkColors` (asset name → mark color),
  `counts{sensors,regions,tripwires,children}`,
  `urls{scenesHome,camList?,sensorList?,scene3d,sceneEdit,sceneDelete,camCreate}`,
  `authToken`, `isSuperuser`, `isKubernetes`, `appVersion`, `appGitCommit?`,
  `googleMapsApiKey?`, `mapboxApiKey?`, `deleteImpact?`,
  `childRoiJson?`, `childTripwireJson?`, `childSensorJson?`,
  `scenes[]` (for pickers: `id`, `name`, `georeferenced?`, `mapUrl?`).
- **cameras|sensors|assets**: `title`, `breadcrumbs[]`,
  `primaryAction?{label,href,id?}`, `columns[]`,
  `rows[](id,cells[],actions[])`, `emptyMessage`, `isSuperuser`
  (no `authToken`; sheets bootstrap carries Token).
- **list-sheets** (id = `cam|sensor|asset`): `authToken`, `isSuperuser`,
  `kind`, `defaultSceneId`, `isKubernetes`, `cameras?`/`sensors?`,
  `scenes[]`.
- **models**: `isSuperuser` only; tree via Token model-directory API.
- **sign-in**: `appName`, `appVersion`, `intelLogo`, `intelHref`,
  `nextUrl?`, `docsVersion?`; posts to `POST /sign_in/`
  (`Accept: application/json`) → `{ ok, redirect }`.

---

## 3. ADR-19 constraints (host-independent manager UI)

- **Frozen boundary is HTTP + bootstrap JSON + MQTT.** DOM ids and
  `window.ss*` bridges are transitional debt — keep them working, do not
  extend them, do not add new ones. New code uses bootstrap + Token fetch +
  React-owned mounts (`lib/rest.ts`, `lib/session.ts`, `lib/bootstrap.ts`,
  `lib/legacyBridge.ts`).
- **Dual host path must stay aligned**: Django thin mounts (templates supply
  roots + scripts) and the static shell (`static/ui/shell.html` + `spa.js`)
  are both valid. New UI must work against the bootstrap API, not assume
  Django template siblings.
- **React owns** scene MQTT connect, camera strip, local sensors, chrome,
  sign-in, and live marks on React-map scenes. Snap / `sscape.js` keep child
  ROI/tripwire/sensor overlays and calibrate until retired.
- **Destructive/setup APIs are Token-auth** (deletes, model-directory, mesh
  generate/status). Do not parse CSRF cookies in feature modules.
- **Do not reopen Snap / calibrate iframe work.** Keep `meet` aspect on the
  scene map (never `slice`/cover stretch).
- Layout shells stay distinct: browse column (admin lists, 64rem) vs
  workspace (scene detail full-bleed). Tokens: edit only
  `src/tokens/ss-tokens.css`; the `make -C manager ui-build` step syncs the
  static copy.
- Framework swap (FastAPI) and full React 3D viewport were **deferred** —
  this rebuild's Phase 1 is exactly that deferred 3D epic, against the same
  frozen contract. ADR 18 (3D child-scene placement widget) already shipped;
  do not regress it.

---

## 4. Frontend map

### 4.1 Entry points and mount scheme

Vite multi-entry build (`vite.config.ts` `rollupOptions.input`);
bundles land in `manager/backend/manager/static/ui/` via
`make -C manager ui-build` (or `SKIP_UI=1`).

| Entry (`src/`) | Bundle | Mounts |
|---|---|---|
| `chrome-main.tsx` | `chrome` | `#ss-chrome-root` (navbar, theme toggle, about) |
| `spa-main.tsx` | `spa` | Static shell: `#ss-chrome-root`, `#ss-spa-root`, scene-detail + scenes-home roots |
| `sign-in-main.tsx` | `sign-in` | `#ss-sign-in-root` |
| `main.tsx` | `scene-detail` | `#ss-scene-detail-root` → `SceneDetailApp` (map pane, side panel tabs, camera strip, MQTT) |
| `admin-list-main.tsx` | `admin-list` | `#ss-admin-list-root` (Cameras / Sensors / Object Library lists) |
| `scenes-home-main.tsx` | `scenes-home` | `#ss-scenes-home-app` (thumbnail gallery) |
| `list-sheets-main.tsx` | `list-sheets` | Query-driven sheets (`?ss=…`, no dedicated root) |
| `models-directory-main.tsx` | `models-directory` | `#ss-models-directory-root` |
| `destructive-actions-main.tsx` | `destructive-actions` | Creates `#ss-destructive-actions-root` (delete interceptor → Token REST) |

Scene detail composition (`src/scene/`): `SceneDetailApp` →
`SceneDetailPage` + `SceneMapPane` (2D SVG map host) + `SceneSidePanel`
(Regions/Tripwires/Cameras/Sensors/Children/MQTT tabs) + `CameraStrip` /
`CameraStripEnhancer` + `TabToolbar` + `WorkspaceSplitter`.
Sheets (`src/sheets/`, e.g. `ChildSheet`) handle create/edit flows.

### 4.2 three.js status

- **React frontend**: `three@^0.185.1` (package.json). Used only in
  `src/placement/`: `ChildPlacementCanvas.tsx` (OrbitControls +
  TransformControls from `three/addons`), `loadSceneObject.ts` (GLTFLoader),
  `poseThree.ts` / `placementAxes.ts` (math). Unit-tested
  (`*.test.ts` colocated). No three.js in the scene-detail viewport today —
  the 3D viewport is greenfield for Phase 1.
- **Legacy 3D view**: `manager/backend/manager/templates/sscape/base_3d.html`
  (+ `templates/scene/scene_detail.html` extends it) loads
  `manager/backend/manager/static/js/scenescape3d.js` (736 lines, module).
  It imports `* as THREE` from `/static/assets/three.module.js` (vendored at
  build time by `manager/Makefile` — not in git), plus OrbitControls,
  GLTFLoader, lil-gui, Stats, `assetmanager.js`, `thing/managers/*`.
  Z-up world (`THREE.Object3D.DEFAULT_UP = (0,0,1)`), persp/ortho cameras,
  scene GLB + Asset3D GLB marks via `assetManager.plot()` on
  `scenescape/regulated/scene/{id}` messages, camera-frame projection.
  **Phase 1.4 deletes** `base_3d.html`, `scenescape3d.js`, and legacy 3D CSS
  (`static/css/scenescape.css`) after parity.

### 4.3 MQTT data flow (scene detail)

1. `useSceneMqtt({sceneId, wssConnection})` connects (WSS, `mqtt.min.js`
   client), sets `window.ssMqttClient`, wires `#connect` / `#disconnect`
   buttons, and calls `attachLegacySceneHandlers` → legacy `sscape.js`
   subscribes (regulated scene, events, child status) and dispatches
   `ss-scene-objects` for marks.
2. `useCameraStripMqtt` subscribes `scenescape/image/camera/+`, publishes
   `getimage` to `scenescape/cmd/camera/{sensorId}`, applies base64 JPEG
   frames to `.snapshot-image` elements.
3. `MarksLayer` listens for `ss-scene-objects` → `plotLiveMarks(objects)`
   on `#svgout` (SVG `g.mark`, `transform` attribute; trails optional).
4. `SensorLayer` draws local sensors; `MqttSettingsPanel` owns broker UI.

### 4.4 Placement (must not regress — ADR 18)

`src/placement/` is the shipped 3D child-scene placement widget:
`ChildPlacementWorkspace` (full-viewport 3D chrome, same panel pattern as
calibrate) + `ChildPlacementCanvas` (three.js: parent map/GLB as world,
child as movable Object3D, translate/rotate/scale gizmo) mounted from
`sheets/ChildSheet.tsx`. Pose model: `SceneEulerPose`
`{translation:[x,y,z], rotation:[x,y,z] (XYZ degrees, Z-up), scale:[x,y,z]}`.
Helpers: `poseThree.ts` (Euler↔matrix), `placementAxes.ts`,
`loadSceneObject.ts` (GLTF load), `sceneGeometry.ts`. Reuse its gizmo and
pose-math patterns for the Phase 1 viewport tools.

### 4.5 UI library / styling notes for the rebuild

- Tokens: `src/tokens/ss-tokens.css` (`--ss-*` light/dark). React islands
  import via `tokens/tokens.css`.
- React island CSS bundles to `static/ui/manager-ui.css`; Django chrome
  stays in `manager/backend/manager/static/css/` (barrel `style.css`).
- Sign-in island posts `POST /sign_in/` with `Accept: application/json`.

---

## 5. Verification commands

Run from `manager/frontend` (Node 20+, `npm install` first):

```bash
npm run typecheck   # tsc -p tsconfig.json --noEmit
npm run lint        # eslint -c eslint.config.js src/
npm run test        # vitest run  (6 test files: lib/geospatialSnapshot, mqtt/client, placement ×4)
npm run build       # typecheck + vite build  → bundles to manager/backend/manager/static/ui/
npm run dev         # vite dev server
```

Repo-level: `make -C manager ui-build` (builds frontend into Django
static; `SKIP_UI=1` to skip).

Conventions that must hold for any new viewport code:
- No new `window.ss*` globals; no new required template sibling ids.
- New REST goes through `lib/rest.ts` with Token auth.
- Keep `meet` aspect on map surfaces; do not reopen Snap/calibrate iframes.
- Tests colocated as `*.test.ts(x)`; run `npm run test` before pushing.
