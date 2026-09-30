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

**Status:** Intermediate API/UI boundary + host independence path landed.
Django can still serve templates as a dual path; the UI can also load from
`static/ui/shell.html` + `/api/v1/ui-bootstrap/`. Do **not** start a framework
swap until operators use the static path. **G** remains optional.

### Done

| Slice | What landed |
| --- | --- |
| 1. Freeze the contract | [`docs/design/manager-ui-backend-contract.md`](../../docs/design/manager-ui-backend-contract.md) |
| 2. Scene-detail DOM decoupling | Template = root + bootstrap; `ensureSceneDetailDom` |
| 3. Portable auth (deletes) | Token DELETE for entities |
| 4a–4c | Token model-directory + mesh; `services/scene_map.py` |
| **D** | React MQTT + camera strip + sensors; Snap marks via shared client |
| **E** | React chrome island (`ui/chrome.js`); `ss-chrome-bootstrap` |
| **F** | `GET /api/v1/ui-bootstrap/`; `shell.html` + `spa.js`; islands fetch when no `json_script` |

Key commits: `02ed8ce9d` … `6fabb3ea5` (E); dual-path retirement through
`dfaa201be` (Token-only deletes) + skill freeze trim.

### Left (ordered)

| # | Item | Required for swappable host? | Notes |
| --- | --- | --- | --- |
| **G** | **Framework swap** (e.g. FastAPI) | No — optional | Same frozen contract; cut Django templates when ready |

### Progress table

| Slice | Status |
| --- | --- |
| 1–5 | ✅ Done |
| 6a. React chrome shell (**E**) | ✅ Done |
| 6b. Host independence (**F**) | ✅ Done (static shell + bootstrap API; list pages on ui-bootstrap) |
| 7. Framework swap (**G**) | ⬜ Optional |

### What still binds us optionally to Django templates

- Session `sign_in/` form remains Django.
- Snap marks / calibrate legacy JS still load from `/static/js`.
- List / scene pages still have thin Django HTML mounts (roots + scripts);
  static `shell.html` can serve the same islands via ui-bootstrap.

Gate: UI BAT green on Django dual-path; static shell smoke for `/` and
`/<uuid>/` against ui-bootstrap.

Out of scope for trickle PRs: rewriting the tracker/controller stack, or
replacing Django in one shot (**G**).

## 2. Optional: how-to updates

When chrome labels, open paths, or nav targets change, update
`docs/user-guide/how-to-guides/` (see
[`.github/skills/documentation-how/SKILL.md`](../skills/documentation-how/SKILL.md)).
