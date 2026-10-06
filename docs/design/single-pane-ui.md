# Scenescape — Single-Pane 3D UI: Design Spec

**Status:** concept mockup · **Target repo:** `open-edge-platform/scenescape`, branch `feature/django-swappable`
**Mockup:** [`scenescape-single-pane.html`](./scenescape-single-pane.html) — open in any browser, no build step, no network (three.js r185 vendored in `./vendor/three/`).

## 1. The problem

Today the Manager UI is a hybrid:

- Django-rendered pages (scene list, admin lists for cameras/sensors/models, forms, sign-in)
- React/Vite "islands" mounted into those pages (scene detail map pane, side panel tabs, camera strip, chrome/navbar)
- Legacy JS (`sscape.js`, Snap SVG) still owning the 2D map on some pages
- A **separate legacy 3D viewport** (`base_3d.html`) disconnected from the 2D scene-detail page

Result: users context-switch between a 2D form-driven page and a separate 3D view to do one job (lay out regions, tripwires, cameras, sensors; watch tracked objects). Geometry edits happen in form fields far from the thing being edited.

## 2. The concept

**One window. One 3D viewport. Everything else is a panel around it.**

- The 3D scene is the application — not a page you navigate to.
- All editing happens **in the viewport** (drag regions, draw tripwires, place cameras) with the Properties panel as a precision companion — never as the primary interface.
- No page reloads, no separate 2D/3D modes, no Django forms for scene content.

### Layout anatomy

```
┌────────────────────────────────────────────────────────────────┐
│ Menu bar: File Edit Scene Camera Sensor View Window Help   MQTT │
├────────────────────────────────────────────────────────────────┤
│ [Dock 7 · Warehouse] [North Parking Lot] [+]                   │  scene tabs (Phase 3)
├────────────────────────────────────────────────────────────────┤
│ Tool name │  contextual tool options (like Compositor's header) │  tool header
├────┬───────────────────────────────────────────────┬───────────┤
│ T  │ View:[Top|Front|Side|Persp] Overlays:[Grid…] Mode:[Live|Replay] │ Outliner │ ← viewport header bar (docked)
│ o  ├───────────────────────────────────────────────┤ Properties│
│ o  │                                               │ /Telemetry│
│ l  │              3D VIEWPORT                      │           │
│ r  │         (three.js, Blender-style)             │  (right   │
│ a  │                                               │   panel,  │
│ i  │  [camera strip overlay]      [tool hint pill]  │   300px)  │
│ l  │  [replay timeline — Replay mode only]         │           │
├────┴───────────────────────────────────────────────┴───────────┤
│ status bar: tool · cursor coords · objects · fps · MQTT        │
└────────────────────────────────────────────────────────────────┘
```

## 3. Design inputs

### 3a. Menu structure & color — Robbie Tilton's Compositor

Compositor is a Photoshop-style Mac image editor. Verified against live screenshots of robbietilton.com/composer (Oct 2026; five app windows in the hero composite, no demo video):

- **Native menu bar:** Compositor · File · Edit · View · Select · Image · Filter · Layer · Window · Help. (Our web adaptation: File · Edit · Scene · Camera · Sensor · View · Window · Help — the image-editing menus Select/Image/Filter/Layer map to the scene-domain menus Scene/Camera/Sensor.)
- **Document tabs** under the menu bar (multi-scene tabs in our UI).
- **Contextual options bar** under the tabs: the active tool's options inline (Compositor: Freehand/Polygonal, New/Add/Subtract, Anti-alias, Expand/Contract, zoom). Our tool header mirrors this per tool.
- **Left vertical single-column toolbar** with tool icons + shortcut badges; foreground/background color chips at its bottom.
- **Right docked panel** ("Layers" in Compositor: blend dropdown, opacity slider, layer stack with eye toggles and thumbnails, adjustment layers double-click-to-edit). Ours: Outliner / Properties / Telemetry tabs.
- **Floating dark dialogs** for adjustments (independent, rounded, blue OK buttons) — our calibrate/manage flows follow this instead of full-page forms.
- **Bottom status bar** with zoom, document size, and per-tool keyboard hints — ours shows tool, cursor coords, object count, fps, MQTT state.
- **Color scheme (verified):** dark charcoal panels `#1E1E1E`–`#2B2B2E`, dialogs `#232326`, text white/light gray, selection accent macOS blue `#0A82FF` (OK buttons, active tool pills, checkbox ticks, slider fills, selected rows).
- **Micro-interactions:** dirty-state affordances, inline rename, drag-to-scrub numbers (Compositor/Photoshop number scrubbing — recommended for threshold fields).

### 3b. 3D interaction — Blender / three.js

- **Navigation (Blender defaults):** `MMB` orbit · `Shift+MMB`/`RMB` pan · wheel zoom · `1/3/7` front/side/top · `5` persp/ortho toggle. Numpad-style view presets also in the viewport header bar.
- **Transform tools:** `V` select · `G` move · `R` rotate · `S` scale — three.js `TransformControls` gizmo with Blender's axis colors (X red `#ff5c5c`, Y green `#6fe36f`, Z blue `#5c8dff`), optional unit snapping from the tool header.
- **Viewport overlays** (Blender's overlay popover): grid, camera frustums, object trails, labels — toggles in the docked viewport header bar, persisted per user.
- **Viewport header bar** (Blender's 3D viewport header / Rerun's view top bar, in Compositor's options-bar language): a slim docked bar atop the viewport grouping View presets (`1/3/7`), a Persp|Ortho segmented control, overlay options as **checkboxes** (Grid, Frustum, Trails, Labels — like Compositor's "Anti-alias"), and the Live|Replay mode switch. The active choice in any segmented control is a **solid blue pill** (Compositor's "Freehand"/"Add" pattern). No floating control clusters — all viewport chrome is panel-docked.
- **Scene collection / Outliner** (Blender's Outliner): hierarchical tree with visibility eye toggles, selection sync both ways.
- **Properties panel** (Blender's Properties): tabbed inspector for the selection; numeric fields edit live in the viewport.
- **Status bar** with active-tool hints (Blender's status bar shows keymaps per tool).

### 3c. Domain model — Scenescape (from `feature/django-swappable` scrape)

Entity types carried over 1:1: **Scene → Regions (ROIs) → Tripwires → Cameras → Sensors → Tracked objects → Child scenes**, plus MQTT telemetry and the model directory.

## 4. Feature mapping (current → new)

| Today (hybrid) | Single-pane |
|---|---|
| Scenes Home thumbnail gallery | **Open Scene… browser** (`⌘O`, Phase 1): a Compositor-style floating dialog — same-page gallery control with search, live viewport thumbnails, per-scene stats (objects · cameras · modified), and a New Scene card. No separate page, by design. **Multi-scene tab strip** (Phase 3): tabs are *opened* scenes for instant switching; the gallery dialog browses *all* scenes and opening one adds a tab. The mockup's tab strip illustrates the Phase 3 target UX. |
| Scene detail 2D SVG map (`SceneMapPane`) | The 3D viewport itself; ROIs/tripwires drawn as volumetric overlays |
| Side panel tabs (Regions/Tripwires/Cameras/Sensors/Children/MQTT) | Outliner tree (left-grouped, badge counts) — same data, one list |
| ROI/tripwire Django form cards (`#form-roi_{uuid}`) | In-viewport creation (click-drag) + Properties inspector for exact values |
| Camera strip (`CameraStrip`) | Camera strip overlay, bottom-left of viewport; click → live feed in place |
| Calibrate pages (`?ss=calibrate-*`) | In-viewport calibrate mode — split multi-view layout (3D + camera feed), wizard in tool header, same REST calls (see Phase 2.3) |
| Admin lists (Cameras, Sensors, Models) | **Folded fully into the single pane** (decided): Cameras/Sensors as Outliner groups, Models as a browser drawer — all REST-backed. The SPA never navigates away to a list page. The actual Django admin site stays outside the SPA and opens in a new tab when needed. |
| Object Library = `Asset3D` rows (Django admin list — class name, GLB file, rotation/translation/scale, sizes, buffers, mark color, tracking/physics params — all as blind forms, no visualization) | **Library drawer** (`B`): searchable per-class cards (mark color, GLB filename), docked left. **Mark editor** (click a card): the scene isolates to a stage showing the class's GLB with its **default pose applied**, exactly as the 3D view composes marks — *tracker transform (scene controller) × default pose × GLB*. The old form's rotation/translation/scale fields become a rotate/move/scale **gizmo on the pose frame** with the fields syncing live underneath; size/buffer render as footprint overlays; a **“Simulate tracker”** toggle drives the mark around a path with velocity heading (honoring `rotation_from_velocity`) so the default orientation can be verified against travel direction — the white arrow marks the tracker forward. Properties sections mirror the model: Model (class name, GLB + Replace), Default pose, Mark (color, size, buffer, tracking radius, shift type, project-to-map), Physics (geometric center, mass, center of mass, is-static, TTL). Save → REST `PUT /api/asset3d`. Live viewport marks also read the library (mark color, footprint size, default rotation, velocity heading). |
| Sheets (`ChildSheet`, manage sheets) | Properties panel modes — no modal sheets for scene content |
| Legacy 3D view (`base_3d.html`) | **Deleted** — the viewport *is* the app |
| Scene settings (`?ss=scene-manage`) | Properties panel with nothing selected |
| MQTT/telemetry tab | Telemetry pane (live event feed) + status-bar connection chip |

## 5. Interaction spec (normative for the build)

1. **Single selection model.** Viewport click, Outliner click, camera-strip click all select the same entity. `Esc` clears.
2. **Creation is gestural.** Region: click-drag on floor → footprint; height/volumetric/color from tool header. Tripwire: two clicks. Camera/sensor: one click (mount height from header). All creations immediately select the new entity and focus its name field.
3. **Gizmo editing is primary; fields are precision.** Dragging the gizmo updates the model (and the Properties fields, debounced). Typing in Properties updates the viewport live.
4. **Dirty gating everywhere.** Calibrate/manage saves show `Save / Saving… / Saved` (keep today's convention).
5. **Live data never blocks the UI.** MQTT telemetry (object positions, crossings, sensor status) streams into the viewport; if the socket drops, the viewport keeps working and the status chip goes amber.
6. **Keyboard-first, mouse-complete.** Every tool has a key; every key action is also clickable. `?` opens the shortcut sheet.
7. **Undo/redo** via a command stack over entity mutations (create/delete/transform/property) — new; doesn't exist today.

## 6. Visual spec

- **Typography:** system UI stack (`-apple-system, "SF Pro Text", Segoe UI, Inter`), 12px base, 11px dense; mono for coordinates/IDs/timestamps.
- **Density:** 38px menu bar · 34px tabs · 42px tool header · 54px tool rail · 300px right panel · 28px status bar.
- **Chrome palette, dark (verified from Compositor):** window `#141518`, panels `#1e1e22`, panel headers/dialogs `#26262c`, inputs `#17181c`, borders `#34353d`, text `#e9eaee` / dim `#a2a4ae` / faint `#6d6f7a`, selection accent macOS blue `#0a84ff` (+16% soft fill), floating overlays `rgba(23,24,29,.88)`.
- **Chrome palette, light:** window `#e8e8ea`, panels `#f2f2f4`, panel headers/dialogs `#e3e3e7`, inputs `#ffffff`, borders `#d4d4d9`, text `#1c1c1e` / dim `#55565e` / faint `#8a8b95`, accent `#0a68ff` (+12% soft fill), floating overlays `rgba(250,250,252,.9)`.
- **Theming rules:** every chrome surface reads `--ss-*`-style tokens (no hardcoded colors outside video-feed content); the 3D viewport adapts too — floor, grid, and scene background/fog recolor on toggle (entity/semantic colors stay constant); toggle lives in the menu bar and under Window menu; mockup proves both via `data-theme`.
- **Entity colors:** region `#30d158` (green) / `#ffd60a` (yellow) per occupancy thresholds; tripwire `#ff9f0a`; camera `#0a84ff`; sensor `#bf5af2`; tracked object `#64d2ff`; child scene `#ffd60a` dashed. Selection = white highlight + blue panel selection.
- **Regions** render as translucent volumetric boxes (height = volumetric height) with edge outlines — the 3D-native replacement for 2D SVG `g.roi` polygons.
- **Camera frustums** render as wireframe pyramids to the floor target + faint volume fill; clicking a frustum selects the camera.
- **Motion:** 150–250ms ease-out for panel/selection transitions; no animation on data updates (telemetry must feel instant).

## 7. Technical build plan (for the agent)

**Stack:** keep `manager/frontend` (React 19 + Vite + three 0.185). The mockup proves the three.js patterns; port them into React components.

### Phase separation contract (read this first)
- **Phase 1 ships alone.** It needs zero new backend services and zero Rerun code. Everything it touches already exists: the REST endpoints and MQTT topics listed under “Data contracts to reuse”. Phase 1 is done when the single-pane UI reaches parity with the Django pages + React islands and the legacy pages are deleted (1.4).
- **Phase 2 is purely additive.** It adds exactly one backend piece (the always-on recorder service, 2.0) and plugs into Phase 1 through two seams — nothing else in Phase 1 changes:
  1. `ReplayProvider` (frontend): `listRecordings(sceneId)` / `createReplayView(recording)` — Phase 1 ships a stub; Phase 2 provides the Rerun-backed implementation.
  2. Recorder sinks (backend): `RrdSink` first, `McapSink`/`JsonlSink` later — the recordings API carries a `provider` field so formats can mix.
- **The mockup's in-memory 10 Hz recorder is a Phase 2 prototype only.** It must not be mistaken for the Phase 2 recorder service and must not be wired into Phase 1 editing paths. Replay mode is view-only in both phases: it never mutates Scenescape entities.
- Ordering rule: no Phase 2 work starts until Phase 1.4 cutover is merged. Phase 3 (multi-scene tabs) starts after Phase 2; it reuses the Phase 1 viewport component and scene-graph store per tab — it does not fork them. The calibration split-view (2.3) reuses the Phase 1 viewport component — it does not fork it.

## Phase 1 — single-pane rebuild

### Phase 1.0 — inventory (do not skip)
- Read `.github/skills/manager-ui/SKILL.md` — **hard DOM/window contracts are frozen** (`#ss-map-host`, `#ss-cameras-mount`, `window.ssMap`, events, etc.). The redesign must keep these working until the legacy pages are deleted; do not rename them in the same PR as the redesign.
- Read `docs/design/manager-ui-backend-contract.md` (REST + MQTT + bootstrap) and `docs/adr/0019-host-independent-manager-ui.md`.

### Phase 1.1 — viewport foundation
- New `src/viewport/` module: renderer, persp/ortho cameras, OrbitControls (Blender bindings), TransformControls, grid, labels layer, view presets. Mount as a full-viewport React component replacing `SceneMapPane`.
- Scene-graph store: entities keyed by id (`zustand` or equivalent), mirroring the mockup's entity model; selectors per type.

### Phase 1.2 — tools & editing
- Tool system: `select/move/rotate/scale/region/tripwire/camera/sensor/measure/live` with the tool-header options pattern; creation gestures raycast against the floor plane; command stack for undo/redo.
- Port the mockup's `makeFrustum`, volumetric region, tripwire, sensor-ring builders.

### Phase 1.3 — panels
- Outliner (replaces side-panel tabs), Properties inspector (replaces Django form cards + sheets), Telemetry pane (replaces MQTT tab), camera strip overlay.
- **Object Library drawer** (`B`, docked left): per-class (`Asset3D`) cards; clicking one opens the **mark editor** — an isolated stage showing the class GLB with its default pose applied (tracker × pose × GLB, exactly like the live view), gizmo-manipulable default pose with live-synced fields, footprint overlays, a “Simulate tracker” toggle that drives the mark with velocity heading so orientation can be verified, and Properties sections mirroring the model (Model / Default pose / Mark / Physics). Save → REST `PUT /api/asset3d`. This replaces the blind transform forms.
- Scene switching in Phase 1 is the **Open Scene… dialog** (`⌘O`) — the same-page gallery control. The multi-scene tab strip is Phase 3.

### Phase 1.4 — cutover
- Route `/scene/:id` renders only the single-pane app. Delete `base_3d.html`/legacy 3D CSS, Snap/SVG map code, and the Django form-card templates once the React editors reach parity. Keep `manager-ui/SKILL.md` contracts until then.

### Data contracts to reuse (no backend changes needed)
- REST: regions/tripwires CRUD, sensor PUT/DELETE, camera PUT, child `GET/POST /api/v1/child/{uid}`, models `GET/POST/DELETE /api/v1/model-directory/`.
- MQTT: existing scene topics for telemetry, camera frames, marks — same subscriptions as `src/mqtt/useSceneMqtt`.
- Bootstrap JSON stays the initial scene-graph seed (replaces the 2D map bootstrap).

## Phase 2 — Rerun integration & replayability

**Goal:** add time as a first-class citizen — record scene telemetry *and* camera frames, replay them scrubbed on a Rerun-style timeline, and use multi-view layouts for calibration. This is the one Phase-2 item that adds capability, not just chrome.

**Why Rerun:** its core strength is multimodal synchronized playback — 3D entities and image/video frames logged with timestamps onto one timeline, scrubbed together. That is exactly "MQTT telemetry + camera frames in one replay." `rerun-sdk` is Python/Rust, Apache-2.0/MIT — license-compatible with this repo.

### Phase 2.0 — recording service (the one backend addition)

Today there is no replay store: `tools/mqtt_recorder.py` is a manual CLI that dumps one topic to JSON-lines for N seconds. Build a small always-on recorder service:

- Subscribes to each scene's MQTT topics (telemetry, object tracks, sensor status — the same topics `src/mqtt/useSceneMqtt` uses).
- Captures camera frames (MJPEG/WebRTC snapshots at a configured cadence, e.g. 1–5 fps for replay; full-rate live viewing stays on the existing stream path).
- Writes **time-partitioned `.rrd` files via `rerun-sdk`** (e.g. one file per scene per hour) to object storage / local volume. `.rrd` is chosen over JSONL: it carries typed entities + timestamps + images in one container the viewer reads natively.
- Retention policy per scene (e.g. 7/30 days); recordings listed via a small REST endpoint: `GET /api/v1/recordings/?scene=<id>` → `[{id, scene, start, end, size, url}]`.

### Phase 2.1 — logging schema (entity mapping)

- Application ID = scene id (Rerun groups recordings and blueprints by application ID — one scene's recordings share one blueprint).
- Timelines: `log_time` (wall clock) + `frame` (monotonic index).
- Entity paths mirror the Phase-1 scene graph: `scene/<name>/regions/<id>` (Boxes3D), `…/cameras/<id>/image` (Image archetype), `…/objects/<track_id>` (Boxes3D/Points3D with per-frame positions), `…/sensors/<id>` (Points3D coverage or custom).
- Static scene geometry (region boxes, tripwires, camera poses) logged once as timeless; only tracks, frames, and sensor state are per-frame.

### Phase 2.2 — Replay mode in the UI

- A **Live / Replay toggle** in the viewport header bar (Mode section). Live = today's Phase-1 viewport on the MQTT socket. Replay = Rerun viewer embedded in the viewport area.
- Embedding: `@rerun-io/web-viewer-react` in `manager/frontend` (preferred — programmatic control, React-native); iframe of a **self-hosted** viewer build as fallback. Ship a saved blueprint (`.rbl`) with blueprint/selection/time panels collapsed so Replay opens chrome-light and on-brand; user can expand panels for deep analysis.
- Recording picker: scene tab context → "Replay…" opens a recording list (from the Phase 2.0 endpoint); selecting one loads its `.rrd` URL into the viewer.
- **Display overrides, not edits:** in Replay mode the Properties panel offers Rerun-style per-entity *display* overrides (e.g. recolor a region's rendering) without touching stored thresholds — keeps the "fields are precision, viewport is truth" contract clean.

### Phase 2.3 — calibration multi-view

Calibration is inherently dual-view: pick correspondence points on the **2D camera image** while watching the **3D frustum/coverage update live**. Implement as a viewport layout mode (not a page):

- Entering Calibrate (inspector button or Camera menu) splits the viewport: left = 3D scene with live frustum + coverage preview, right = the camera's live feed with a point-picking overlay.
- Wizard in the tool header: 1) select camera → 2) click 4+ known points in the feed (snapping guides) → 3) frustum re-solves live in 3D → 4) dirty-gated Save (`Save / Saving… / Saved`, same REST `camera PUT` as today).
- Exiting restores the single 3D view. This replaces the `?ss=calibrate-*` pages from the Phase-1 mapping table.

### Phase 2.4 — reuse vs. build

| Need | Reuse (Rerun) | Build (custom) |
|---|---|---|
| Recording format + timestamps | `.rrd` via `rerun-sdk` | — |
| Timeline scrub / play / loop | Rerun viewer | — |
| Multi-view layouts (3D + image + chart) | Rerun viewer + blueprints | — |
| Display overrides | Rerun viewer overrides | — |
| Region/tripwire/camera editing UI | — | Phase-1 single-pane editor (Rerun viewer is view-only) |
| Calibration tools | — | Custom (2.3), same REST calls |
| Recorder service + retention API | — | Small new service (2.0) |

### Phase 2.5 — caveats (do not skip)

- **Version lock:** the `.rrd` format still evolves — pin the `rerun-sdk` version and the web-viewer version together; mismatched versions cannot read each other's files. Record the SDK version in each recording's metadata.
- **Self-host the viewer.** Do not hotlink `app.rerun.io` for sensitive footage — it is third-party JS reading your recordings. Serve the viewer build from the manager's static assets; the `.rrd` URL needs `Access-Control-Allow-Origin` scoped to the viewer origin.
- **Styling boundary:** the embedded Rerun viewer is egui-styled and will not match the Compositor chrome — acceptable because Replay is a distinct mode, and the collapsed-panel blueprint keeps it visually quiet.

### Phase 2.6 — backend pluggability (no tight coupling to Rerun)

Rerun is the reference backend, not a hard dependency. Two seams keep the UI backend-agnostic:

**Frontend seam — `ReplayProvider` interface** (`manager/frontend/src/replay/`):

```ts
interface RecordingMeta { id: string; sceneId: string; start: number; end: number;
                          size: number; provider: string; url: string }
interface ReplayProvider {
  id: string;                                   // 'rerun' | 'mcap' | 'jsonl' | …
  listRecordings(sceneId: string): Promise<RecordingMeta[]>;
  // Returns the React component rendered in the viewport area during Replay mode.
  createReplayView(recording: RecordingMeta): React.ComponentType;
}
```

- The Replay mode shell (Live/Replay toggle, recording picker, time-range display) talks only to this interface — it never imports Rerun types.
- `RerunProvider` implements it by rendering the embedded web viewer with the scene's saved blueprint.
- A future provider (MCAP/Foxglove, plain JSONL, …) implements the same interface. Providers whose format decodes to simple frames may additionally render through a **shared neutral frame renderer** (our three.js viewport + our timeline UI) instead of bringing their own viewer component — same interface, either rendering path.
- Provider is selected per deployment via config (`replay.provider: rerun`), not per code change.

**Mockup prototype (already working in `scenescape-single-pane.html`):** the Live/Replay toggle in the viewport header bar switches the viewport into Replay mode — object tracks are recorded at 10 Hz into an in-memory buffer (standing in for the recorder service + `.rrd`, behind the same `{frames, events}` shape the provider seam specifies), and Replay scrubs them on a Rerun-style timeline with play/pause, event dots for region crossings, and a recording picker. Edit tools are disabled in Replay (view-only contract); `Space` toggles playback.

**Backend seam — recorder sinks:**

- The Phase 2.0 recorder service writes through a pluggable sink: `RrdSink` (rerun-sdk) today; `JsonlSink` / `McapSink` later. Sink choice is deployment config.
- The recordings REST endpoint stays neutral: `{id, scene, start, end, size, provider, url}` — the `provider` field tells the UI which `ReplayProvider` handles it. Mixed deployments (old `.rrd` + new format) keep working.

Net: swapping Rerun for another backend = implement one provider + one sink, no UI rewrite.

## Phase 3 — multi-scene tabs

- Tab strip under the menu bar: each tab is an **opened scene** (scene-graph store instance per tab, viewport component reused). Switching tabs is instant — no reload, no route change.
- The **Open Scene… dialog** (Phase 1) becomes the gallery: it browses *all* scenes; opening one from the gallery adds a tab (or focuses it if already open). Tabs = working set, gallery = library.
- Tab interactions: `+` opens the gallery dialog, middle-click/`⌘W` closes a tab (dirty state blocks with a Compositor-style confirm), drag to reorder, right-click for duplicate/rename/close-others.
- Deep-linking stays per-scene (`/scene/:id` focuses or opens the tab); the SPA still never navigates away to a list page.
- Starts after Phase 2; touches only the shell (tab strip + per-tab store scoping), not the viewport, tools, or panels.

## 9. Open questions for the user

1. Ortho top-down as the *default* view (faithful to today's 2D map) or perspective (Blender-like)? Mockup defaults to perspective.
2. ~~Keep the Django admin lists as separate pages, or fold Cameras/Sensors/Models fully into the Outliner + a models drawer?~~ — **Decided:** fold fully into the single pane (Outliner groups + Models drawer); the SPA never navigates away; Django admin itself opens in a new tab.
3. ~~Multi-scene tabs vs. the existing Scenes Home gallery — tabs, gallery, or both?~~ — **Decided:** same-page gallery dialog in Phase 1; multi-scene tab strip (tabs = opened scenes) is **Phase 3**.
4. Light theme: keep the existing theme toggle, or dark-only like Compositor?

## 10. Files in this folder

- `scenescape-single-pane.html` — interactive mockup (open in browser)
- `vendor/three/` — vendored three.js r185 (module + Orbit/TransformControls), so the mockup works offline
- `DESIGN-SPEC.md` — this file
