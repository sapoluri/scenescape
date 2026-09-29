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
| 3. Portable auth (deletes) | `ss-auth-bootstrap`, `lib/session` / `lib/restDelete`; Token DELETE for scene/cam/sensor/child/asset |
| 4a. Token model-directory | `ModelDirectory` + `modelDirectoryApi.ts` use Token auth |

Key commits: `02ed8ce9d` (contract), `847bcf272` (DOM), `a8faccb72` (Token deletes).

### Left (ordered)

| # | Item | Notes |
| --- | --- | --- |
| **B** | Token **mesh** generate/status | Still `/scene/generate-mesh…` + CSRF |
| **C** | Thin **domain service layer** | Map upload/align/thumbnail callable without Django views importing models into UI path |
| **D** | Shrink **`window.ss*` / hybrid `sscape.js`** | Runtime hard contracts remain until React owns MQTT/map fully |
| **E** | **`base.html` shell** (nav/about/login) | Optional; not blocking API swap if SPA is separate later |
| **F** | **Host independence** (serve SPA off Django) | Only after B–D are green enough |
| **G** | **Framework swap** (e.g. FastAPI) | Explicitly deferred — optional after F |

### Progress table

| Slice | Status |
| --- | --- |
| 1. Freeze the contract | ✅ Done |
| 2. Retire template/DOM coupling (scene detail) | ✅ Done (nav/`window.ss*` debt remains) |
| 3. Auth as a portable session | ✅ Token CRUD + deletes + model-directory; mesh still CSRF |
| 4. Extract domain behind HTTP | 🔶 **Next** — B → C above |
| 5. Host independence / framework swap | ⬜ Deferred (F → G) |

### What still binds us to Django

- `base.html` nav/about, session login, static serving.
- Runtime hard DOM / `window.ss*` for hybrid `sscape.js`.
- Domain logic in Django models/views (no backend-agnostic service layer).
- CSRF session calls: mesh generate/status.

Gate: UI BAT and manager functional tests green against the frozen contract;
hard-contract table in the manager-ui skill shrinks as IDs move behind React
ownership; no new `window.ss*` or Django-only DOM requirements for new UI.

Out of scope for trickle PRs: rewriting the tracker/controller stack, or
replacing Django in one shot.

## 2. Optional: how-to updates

When chrome labels, open paths, or nav targets change, update
`docs/user-guide/how-to-guides/` (see
[`.github/skills/documentation-how/SKILL.md`](../skills/documentation-how/SKILL.md)).
