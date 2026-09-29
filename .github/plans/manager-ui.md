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

**Status:** Intermediate API/UI boundary work largely done. Django still
**hosts** the UI. Do **not** start a framework swap until **E** and **F** are
done — those are what make the host swappable. **G** (pick another framework)
is optional after that.

### Done

| Slice | What landed |
| --- | --- |
| 1. Freeze the contract | [`docs/design/manager-ui-backend-contract.md`](../../docs/design/manager-ui-backend-contract.md); skill/plan pointers |
| 2. Scene-detail DOM decoupling | Template = root + bootstrap; `ensureSceneDetailDom` builds map host |
| 3. Portable auth (deletes) | `ss-auth-bootstrap`, `lib/session` / `lib/restDelete`; Token DELETE for entities |
| 4a. Token model-directory | Token auth on `ModelDirectory` |
| 4b. Token mesh generate/status | `/api/v1/scene/<uuid>/generate-mesh[-status]/`; `mesh_http.py` |
| 4c. Domain service (map) | `manager/services/scene_map.py` — align / thumbnail / finalize; model + serializer thin wrappers |
| **D** | React owns scene MQTT connect + camera strip + local sensors; `legacyBridge` only for Snap map/ROI; `ssAttachSceneMqttClient` keeps marks on Snap |
| **E** | React chrome island (`ui/chrome.js`) owns nav/about/theme; `base.html` = mount + `ss-chrome-bootstrap` |

Key commits: `02ed8ce9d`, `847bcf272`, `a8faccb72`, `25d201950`, `0d2cbac74`,
`cddf492b0`, `b1a4281dd`.

### Left (ordered)

| # | Item | Required for swappable host? | Notes |
| --- | --- | --- | --- |
| **F** | **Host independence** (serve SPA off Django) | **Yes** | Static/CDN serves the SPA; Django (or later FastAPI) is API + MQTT only |
| **G** | **Framework swap** (e.g. FastAPI) | No — optional | Only after F; same frozen contract, different server |

### Progress table

| Slice | Status |
| --- | --- |
| 1. Freeze the contract | ✅ Done |
| 2. Retire template/DOM coupling (scene detail) | ✅ Done |
| 3. Auth as a portable session | ✅ Token CRUD + deletes + model-directory + mesh |
| 4. Extract domain behind HTTP | ✅ Map upload/align/thumbnail in `services/scene_map.py` |
| 5. Shrink `window.ss*` / hybrid JS | ✅ Done for scene-detail MQTT/cameras/sensors; Snap marks remain |
| 6a. React chrome shell (**E**) | ✅ Done — nav/about/theme via `ss-chrome-bootstrap` |
| 6b. Host independence (**F**) | ⬜ Required next |
| 7. Framework swap (**G**) | ⬜ Optional after F |

### What still binds us to Django as host

- Django still **serves** page HTML shells and static files (**F**).
- Session `sign_in/` form remains Django (acceptable until a static login).
- Snap overlay + `ssAttachSceneMqttClient` for live marks/trails.
- Legacy `/scene/generate-mesh…` session URLs remain for compatibility; UI
  uses Token `/api/v1/…`.

Gate: UI BAT and manager functional tests green against the frozen contract;
hard-contract table in the manager-ui skill shrinks as IDs move behind React
ownership; no new `window.ss*` or Django-only DOM requirements for new UI.
New React code must use `lib/legacyBridge` / `src/mqtt` (or React APIs) — not
ad-hoc `window.ss*`.

Out of scope for trickle PRs: rewriting the tracker/controller stack, or
replacing Django in one shot (**G**).

## 2. Optional: how-to updates

When chrome labels, open paths, or nav targets change, update
`docs/user-guide/how-to-guides/` (see
[`.github/skills/documentation-how/SKILL.md`](../skills/documentation-how/SKILL.md)).
