<!--
SPDX-FileCopyrightText: (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Manager UI

Remaining work after the 2D React rewrite.

Conventions, layout shells, tokens, and hard DOM contracts:
[`.github/skills/manager-ui/SKILL.md`](../skills/manager-ui/SKILL.md).
Package notes: [`manager/frontend/README.md`](../../manager/frontend/README.md).

## 1. 3D scene viewport (epic)

**Not started.** Legacy Three.js surface remains:

- Entry: `manager/backend/manager/static/js/scenescape3d.js` plus modules under
  `static/js/thing/`, `viewport.js`, managers, etc.
- Mount: `base_3d.html` — no React root / no `manager/frontend` 3D entry.
- Scene detail links out (`#3d-view` → `urls.scene3d`). 3D chrome already
  deep-links back to 2D (`#scene-detail-link`, `#scene-detail-button`,
  per-camera calibrate URLs).

Replace or wrap with a React-owned shell that reuses MQTT / auth patterns
from the 2D rewrite. Keep the **workspace** shell (full-bleed). Do **not**
fold into 2D trickle PRs.

Precursor: non-georeferenced child linking already ships a thin Z-up
placement canvas (`manager/ui/src/placement/`) with `poseThree` conversion
and TransformControls. Reuse that pose/gizmo module in the viewport; do
not wrap `scenescape3d.js` for hierarchy placement.

Suggested slices:

1. Inventory: entry points, MQTT topics, asset load path, Django mounts.
2. Thin React mount + bootstrap JSON (parity shell; keep Three under the hood).
3. Port interaction chrome (layers, selection, camera controls) into React.
4. Retire legacy script load path when UI tests cover 3D BAT.

Gate: document any new 3D contract ids in the manager-ui skill before deleting
legacy globals; UI BAT green for scene 3D view when that suite exists.

## 2. Path to a swappable backend (epic)

**Status:** Intermediate step in progress — contract frozen; Django remains
host. Do **not** start a framework swap.

The React islands already prefer REST + bootstrap JSON (`authToken`, scene
payloads) over pure server-rendered forms. Grow the UI against the frozen
HTTP/MQTT contract
([`docs/design/manager-ui-backend-contract.md`](../../docs/design/manager-ui-backend-contract.md)),
then optionally put a different server behind that contract later.

### What already helps

- Vite islands (`scenes-home`, `scene-detail`, admin lists, sheets) own more
  chrome each release.
- Bootstrap `json_script` + REST tokens reduce template form coupling.
- MQTT and auth patterns are reusable if the wire contract is frozen.
- **Frozen contract doc** (bootstrap, REST, auth modes, MQTT, DOM debt).
- Portable client helpers: `manager/frontend/src/lib/session.ts`,
  `lib/bootstrap.ts`, Token CRUD via `lib/rest.ts`.

### What still binds us to Django

- Page shells, session/auth, and static serving still come from Django
  templates (`base.html`, list pages, etc.). Scene detail content is a
  single React root + bootstrap `json_script` only.
- Hard DOM/window contracts (`#map-controls`, `#ss-map-host`, `window.ss*`,
  legacy map JS) still exist at **runtime** (built from bootstrap) for
  hybrid `sscape.js` — not as Django template siblings. See the manager-ui
  skill hard-contract tables.
- Domain logic (scene map/GLB upload, serializers, permissions, MQTT wiring)
  lives in Django models/views without a backend-agnostic service boundary.
- CSRF deletes / mesh / model-directory keep some calls on the Django
  session request cycle.

### Suggested slices (order matters)

1. **Freeze the contract.** ✅ Documented in
   [`docs/design/manager-ui-backend-contract.md`](../../docs/design/manager-ui-backend-contract.md).
   Keep TS bootstrap types + OpenAPI + that doc in sync. Treat skill hard
   contracts as debt to retire, not as the long-term boundary.
2. **Retire template/DOM coupling.** ✅ Scene detail template is root +
   bootstrap only; map host / toggles / geometry hiddens come from
   `ensureSceneDetailDom`. Remaining: nav/about still Django `base.html`;
   shrink `window.ss*` as legacy map JS retires.
3. **Auth as a portable session.** ✅ Helpers landed (`lib/session.ts`);
   Token path is primary. Still replace CSRF DeleteViews / mesh /
   model-directory with Token REST before calling auth “done.”
4. **Extract domain behind HTTP.** Scene map upload/align/thumbnail, CRUD,
   and permissions callable without importing Django models from the UI path.
   Prefer thin API handlers over template views for anything the React app
   touches.
5. **Host independence last.** Only after (1)–(4): serve the SPA/static UI
   from a non-Django host (or reverse proxy) pointed at the API. Do **not**
   attempt a framework swap before the UI can run without template-injected
   DOM and `window.ss*` bridges.

Gate: UI BAT and manager functional tests green against the frozen contract;
hard-contract table in the manager-ui skill shrinks as IDs move behind React
ownership; no new `window.ss*` or Django-only DOM requirements for new UI.

Out of scope for trickle PRs: rewriting the tracker/controller stack, or
replacing Django in one shot.

## 3. Optional: how-to updates

When chrome labels, open paths, or nav targets change, update
`docs/user-guide/how-to-guides/` (see
[`.github/skills/documentation-how/SKILL.md`](../skills/documentation-how/SKILL.md)).
