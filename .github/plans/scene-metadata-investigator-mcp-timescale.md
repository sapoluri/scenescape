<!--
SPDX-FileCopyrightText: (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Plan: Scene Metadata (Investigator-First) — Timescale Event Log, SQL Graph Projection, MCP

## Goal

Deliver a **Scene Metadata** service that gives investigator agents a durable,
queryable history of tracking and **spatial analytics** (regions and tripwires),
with **operator** and **integrator** personas supported on the same read path.
Storage is **TimescaleDB** (PostgreSQL extension). **Graph traversals** use
relational edge tables and SQL—no property-graph database.

This plan implements **Option A** primary ingest (`scenescape/data/scene/{scene_id}/{category}`)
for track lifecycle milestones, with the explicit clarification that **investigator
MVP requires full-fidelity analytics event ingest** on
`scenescape/event/region/...` and `scenescape/event/tripwire/...` in the same
milestone—not a later phase.

Related: [ADR 13 Phase 1–2](../../docs/adr/0013-controller-breakdown-microservices.md)
(Scene State Persistence + Scene Graph foundation),
[work-with-spatial-analytics-data.md](../../docs/user-guide/how-to-guides/work-with-spatial-analytics-data.md).

---

## Non-goals (initial milestones)

- Replacing Manager PostgreSQL or ReID vector stores (VDMS/Qdrant).
- Logging every high-rate `track_updated` geometry frame (milestones only on scene stream).
- MCP write tools (POI enroll, config changes).
- Stream Manager / video clip correlation.

---

## Architecture

```text
MQTT (broker)
  ├─ scenescape/data/scene/{scene_id}/{category}     → track lifecycle normalizer
  └─ scenescape/event/{region|tripwire}/...          → analytics normalizer (full fidelity)
           ↓
    scene-metadata ingest
           ↓
  TimescaleDB hypertable `scenescape_events`
           ↓
  SQL graph projection → `scenescape_graph_edges`
           ↓
  Query HTTP API (/v1/...)
           ↓
  MCP server (stdio) — investigator tools first
```

**Deployment name:** `scene-metadata` (Python service, `scene_metadata` package).

---

## Event envelope (canonical row)

| Field | Purpose |
| ----- | ------- |
| `event_id` | UUID primary key |
| `occurred_at` | Source timestamp (RFC 3339) |
| `ingested_at` | Ingest time |
| `deployment_id` | Hard partition (default `default`) |
| `event_type` | See table below |
| `scene_id`, `region_id`, `tripwire_id` | Spatial context |
| `track_id`, `identity_id` | `id` vs global `gid` when present |
| `source_topic`, `source_message_id` | Idempotent ingest |
| `payload` | JSONB remainder |

### Event types (investigator MVP)

| `event_type` | Source |
| ------------ | ------ |
| `track_started`, `track_lost` | Derived from `data/scene` snapshots |
| `region_enter`, `region_exit`, `region_count` | `event/region/.../count` (entered/exited arrays) |
| `region_objects` | `event/region/.../objects` |
| `tripwire_cross` | `event/tripwire/.../objects` |

Future: `identity_assigned`, `reid_match` (reid-service / decision topic).

---

## Timescale schema

- Hypertable: `scenescape_events` partitioned on `occurred_at`.
- Indexes: `(deployment_id, scene_id, occurred_at)`, `(identity_id, occurred_at)`,
  `(region_id, occurred_at)`, `(tripwire_id, occurred_at)`, `(event_type, occurred_at)`.
- Dedup: unique `(deployment_id, source_topic, source_message_id)` when message id set.

### Graph projection (SQL, not graph DB)

Table `scenescape_graph_edges`:

- Columns: `relation`, `src_kind`, `src_id`, `dst_kind`, `dst_id`, `scene_id`,
  `valid_from`, `valid_to`, `payload`.
- Relations (MVP): `SEEN_IN` (track → scene), `ENTERED` / `EXITED` (identity → region),
  `CROSSED` (identity → tripwire).

Projector runs synchronously on ingest for MVP; batch repair job is a follow-up.

---

## Ingest (Option A + analytics)

### Scene track stream (`data/scene`)

- Subscribe: `scenescape/data/scene/+/+`.
- Per `(deployment_id, scene_id, category)` maintain active track id set.
- **Emit:**
  - `track_started` when a track `id` appears.
  - `track_lost` when a track disappears from the snapshot.
- **Do not** emit per-frame `track_updated` in MVP.
- Map `identity_id` from object `id` when it is a global gid; retain local track id separately when both exist.

### Analytics events (investigator-required)

- Subscribe: `scenescape/event/region/+/+/+` and `scenescape/event/tripwire/+/+/+`.
- **No sampling** — every discrete region/tripwire event is stored.
- Parse `entered` / `exited` on count events; tripwire payloads per
  [work-with-spatial-analytics-data.md](../../docs/user-guide/how-to-guides/work-with-spatial-analytics-data.md).

---

## Query API (HTTP, v1)

Investigator-first endpoints:

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `POST` | `/v1/events/query` | Filter by time, scene, types, identity, region, tripwire |
| `GET` | `/v1/identities/{identity_id}/timeline` | Fused track + region + tripwire ordering |
| `GET` | `/v1/regions/{region_id}/activity` | Enter/exit/count in window |
| `GET` | `/v1/tripwires/{tripwire_id}/crossings` | Ordered crossings |
| `GET` | `/v1/health` | Operator: DB + ingest lag |

Graph (SQL):

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `GET` | `/v1/graph/identities/{id}/regions` | Regions linked via `ENTERED`/`EXITED` |
| `GET` | `/v1/graph/identities/{id}/tripwires` | Tripwires linked via `CROSSED` |

Auth: bearer token via env-configured shared secret for MVP; mTLS follow-up.

---

## MCP layer (stdio)

Server: `python -m scene_metadata.mcp_server`

Investigator tools (ship first):

- `query_events`
- `identity_timeline`
- `region_activity`
- `tripwire_crossings`

Operator tools (same backend, narrower defaults):

- `get_system_status` (wraps `/v1/health` + recent ingest stats)

Integrator tools (read-only v1):

- `query_events` with filters on `external_*` when those types exist later

Persona gating: `SCENE_METADATA_MCP_PERSONAS=investigator,operator,integrator` (default all).

---

## Phased delivery

| Phase | Deliverable | Exit criteria |
| ----- | ----------- | ------------- |
| **0** | This plan + service skeleton in repo | Plan merged; package builds |
| **1** | Timescale schema, ingest, store | Unit tests for normalizers; DB integration test |
| **2** | Query API + graph projection | Timeline API returns region + tripwire + track events |
| **3** | MCP investigator tools | Cursor can call `identity_timeline` against a running stack |
| **4** | Compose profile + docs | Demo stack ingests from analytics-enabled scene |

Phases 0–3 and the investigator MQTT path of phase 4 are implemented.
`sample_data/compose/compose.scene-metadata-override.yml` and
`tests/compose/compose.scene-metadata.yml` start Timescale and an anonymous
Mosquitto. Live check (cloud agent, 2026-10-26): publish
`scenescape/data/scene/...`, `scenescape/event/region/.../count`, and
`scenescape/event/tripwire/.../objects`; `GET /v1/identities/{id}/timeline`
returns `track_started`, `region_enter`, and `tripwire_cross`; SQL graph
returns `ENTERED` and `CROSSED`; MCP tools `identity_timeline`,
`region_activity`, `tripwire_crossings`, and `get_system_status` read that
API. A full SceneScape analytics container is not required for this contract;
the ingest topics match the analytics publisher.

---

## Testing

- **Unit:** topic parsers, normalizers, dedup keys, timeline merge (`tests/sscape_tests/scene_metadata/`).
- **Integration:** Timescale container in `tests/compose/compose.scene-metadata.yml`; ingest sample payloads; query timeline.
- **Functional (follow-up):** Full `SCENESCAPE_SPEC` profile with analytics publishing.

---

## Operations

- Env: `SCENE_METADATA_DATABASE_URL`, `SCENE_METADATA_DEPLOYMENT_ID`, MQTT vars aligned with analytics/reid (`BROKER`, auth, certs).
- Retention: configurable `SCENE_METADATA_RETENTION_DAYS` (Timescale drop_chunks policy follow-up).
- Metrics: ingest rate, lag (`ingested_at - occurred_at`), projection errors (OTel follow-up).

---

## Open decisions

1. Shared Postgres vs dedicated Timescale instance for edge demos (MVP: dedicated DB in compose profile).
2. When to add `reid-service` decision topic ingest for `identity_assigned`.
3. Materialized views for hot graph queries vs on-demand SQL only (start on-demand).
