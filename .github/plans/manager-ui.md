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

## 1. Path to a swappable backend (epic)

**Status:** Intermediate step largely done for UI↔API boundary work. Django
remains host. Do **not** start a framework swap.

### Done

| Slice | What landed |
| --- | --- |
| 1. Freeze the contract | [`docs/design/manager-ui-backend-contract.md`](../../docs/design/manager-ui-backend-contract.md); skill/plan pointers |
| 2. Scene-detail DOM decoupling | Template = root + bootstrap; `ensureSceneDetailDom` builds map host |
| 3. Portable auth (deletes) | `ss-auth-bootstrap`, `lib/session` / `lib/restDelete`; Token DELETE for entities |
| 4a. Token model-directory | Token auth on `ModelDirectory` |
| 4b. Token mesh generate/status | `/api/v1/scene/<uuid>/generate-mesh[-status]/`; `mesh_http.py` |
| 4c. Domain service (map) | `manager/services/scene_map.py` — align / thumbnail / finalize; model + serializer thin wrappers |
| **D (partial)** | `lib/legacyBridge.ts` — React→legacy typed facade; React no longer uses `ssToast`/`ssConfirm`/`ssRoiDirty`/`ssMapReact` directly; skill freeze trimmed |

Key commits: `02ed8ce9d`, `847bcf272`, `a8faccb72`, `25d201950`, `0d2cbac74`.

### Left (ordered)

| # | Item | Notes |
| --- | --- | --- |
| **D** | Finish **`window.ss*` / hybrid `sscape.js`** shrink | MQTT client + Snap sensor/camera overlay still on window; React calls via `legacyBridge` |
| **E** | **`base.html` shell** (nav/about/login) | Optional; not blocking API swap if SPA is separate later |
| **F** | **Host independence** (serve SPA off Django) | Only after D is green enough |
| **G** | **Framework swap** (e.g. FastAPI) | Explicitly deferred — optional after F |

### Progress table

| Slice | Status |
| --- | --- |
| 1. Freeze the contract | ✅ Done |
| 2. Retire template/DOM coupling (scene detail) | ✅ Done (nav/`window.ss*` debt remains) |
| 3. Auth as a portable session | ✅ Token CRUD + deletes + model-directory + mesh |
| 4. Extract domain behind HTTP | ✅ Map upload/align/thumbnail in `services/scene_map.py` |
| 5. Shrink `window.ss*` / hybrid JS | 🔄 Partial — `legacyBridge`; MQTT/Snap ownership next |
| 6. Host independence / framework swap | ⬜ Deferred (F → G) |

### What still binds us to Django

- `base.html` nav/about, session login, static serving.
- Runtime hard DOM / `window.ss*` for hybrid `sscape.js` (MQTT + Snap overlays).
- Legacy `/scene/generate-mesh…` session URLs remain for compatibility; UI
  uses Token `/api/v1/…`.

Gate: UI BAT and manager functional tests green against the frozen contract;
hard-contract table in the manager-ui skill shrinks as IDs move behind React
ownership; no new `window.ss*` or Django-only DOM requirements for new UI.
New React code must use `lib/legacyBridge` (or React APIs) — not ad-hoc
`window.ss*`.

Out of scope for trickle PRs: rewriting the tracker/controller stack, or
replacing Django in one shot.

## 2. Optional: how-to updates

When chrome labels, open paths, or nav targets change, update
`docs/user-guide/how-to-guides/` (see
[`.github/skills/documentation-how/SKILL.md`](../skills/documentation-how/SKILL.md)).
