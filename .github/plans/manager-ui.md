<!--
SPDX-FileCopyrightText: (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Manager UI

Remaining work after the 2D React rewrite.

Conventions, layout shells, tokens, and hard DOM contracts:
[`.github/skills/manager-ui/SKILL.md`](../skills/manager-ui/SKILL.md).
Package notes: [`manager/frontend/README.md`](../../manager/frontend/README.md).
Frozen UI↔backend contract:
[`docs/design/manager-ui-backend-contract.md`](../../docs/design/manager-ui-backend-contract.md).

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
placement canvas (`manager/frontend/src/placement/`) with `poseThree`
conversion and TransformControls. Reuse that pose/gizmo module in the
viewport; do not wrap `scenescape3d.js` for hierarchy placement.

Suggested slices:

1. Inventory: entry points, MQTT topics, asset load path, Django mounts.
2. Thin React mount + bootstrap JSON (parity shell; keep Three under the hood).
3. Port interaction chrome (layers, selection, camera controls) into React.
4. Retire legacy script load path when UI tests cover 3D BAT.

Gate: document any new 3D contract ids in the manager-ui skill before deleting
legacy globals; UI BAT green for scene 3D view when that suite exists.

## 2. Path to a swappable backend (epic)

**Status:** Intermediate step — contract frozen, scene-detail DOM decoupled,
entity deletes on Token REST. Django remains host. Do **not** start a
framework swap.

### Progress

| Slice | Status |
| --- | --- |
| 1. Freeze the contract | ✅ Done |
| 2. Retire template/DOM coupling (scene detail) | ✅ Done (nav/`window.ss*` debt remains) |
| 3. Auth as a portable session | ✅ Token deletes + `ss-auth-bootstrap`; mesh / model-directory still CSRF |
| 4. Extract domain behind HTTP | 🔶 Next (mesh, model-directory, service layer) |
| 5. Host independence / framework swap | ⬜ Deferred |

Commits on this path: contract + session helpers; `ensureSceneDetailDom`;
Token REST deletes (`lib/restDelete.ts`, ManageThing pk fallback for cams).

The React islands prefer REST + bootstrap JSON over server-rendered forms.
Grow against
[`docs/design/manager-ui-backend-contract.md`](../../docs/design/manager-ui-backend-contract.md);
optionally put a different server behind that contract later.

### What already helps

- Vite islands own chrome; scene detail template is root + bootstrap only.
- Map host / toggles / geometry hiddens built by `ensureSceneDetailDom`.
- Frozen contract; `lib/session.ts`, `lib/bootstrap.ts`, `lib/rest.ts`,
  `lib/restDelete.ts`; page-wide `ss-auth-bootstrap`.
- Destructive-actions + scene/camera/child panels delete via Token REST.

### What still binds us to Django

- `base.html` nav/about, session login, static serving.
- Runtime hard DOM / `window.ss*` for hybrid `sscape.js`.
- Domain logic in Django models/views (no backend-agnostic service layer).
- CSRF session calls: mesh generate/status, model-directory.

### Suggested slices (order matters)

1. **Freeze the contract.** ✅
2. **Retire template/DOM coupling.** ✅ Scene detail root + bootstrap.
3. **Auth as a portable session.** ✅ Token path for CRUD + entity deletes;
   `ss-auth-bootstrap` on authenticated pages. Mesh / model-directory still
   session+CSRF.
4. **Extract domain behind HTTP.** **← next focus.** Token-auth
   model-directory; mesh generate/status under `/api/v1`; thin service
   layer for scene map upload/align.
5. **Host independence last.** Only after (1)–(4).

### Next concrete work (slice 4)

1. **Token-auth model-directory** (drop session+CSRF on that API).
2. **Mesh generate/status under `/api/v1`** (or mapping proxy) with Token.
3. **Thin service layer** in Manager for scene map upload/align.
4. Only then reconsider host independence.

Gate: UI BAT and manager functional tests green against the frozen contract;
hard-contract table in the manager-ui skill shrinks as IDs move behind React
ownership; no new `window.ss*` or Django-only DOM requirements for new UI.

Out of scope for trickle PRs: rewriting the tracker/controller stack, or
replacing Django in one shot.

## 3. Optional: how-to updates

When chrome labels, open paths, or nav targets change, update
`docs/user-guide/how-to-guides/` (see
[`.github/skills/documentation-how/SKILL.md`](../skills/documentation-how/SKILL.md)).
