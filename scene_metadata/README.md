<!--
SPDX-FileCopyrightText: (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Scene Metadata Service

Investigator-first **event log** (TimescaleDB) and **SQL graph projection** for
Scenescape, with an MCP stdio server on top.

Plan: [`.github/plans/scene-metadata-investigator-mcp-timescale.md`](../.github/plans/scene-metadata-investigator-mcp-timescale.md)

## Build

```bash
make scene_metadata
```

## Run locally (HTTP + MQTT ingest)

```bash
export SCENE_METADATA_DATABASE_URL=postgresql://scenescape:scenescape@localhost:5432/scene_metadata
export SCENE_METADATA_MQTT_ENABLED=false   # when testing API only
python3 -m scene_metadata.server
```

## MCP (stdio)

```bash
export SCENE_METADATA_API_URL=http://127.0.0.1:8080
python3 -m scene_metadata.mcp_server
```

Investigator tools: `query_events`, `identity_timeline`, `region_activity`,
`tripwire_crossings`. Operator: `get_system_status`.

`query_events` takes an `attributes` map of metadata name to label. Names are
not registered in the server. A classifier field is queryable as soon as it
is stored on an event, including fields from
`person-attributes-recognition-crossroad-0238` (`is_male`, `has_backpack`, …)
and any later name. A match is a classificationPolicy object whose `label`
equals the requested value, or a scalar stored under that key.

```json
{"attributes": {"is_male": "male", "scarf_color": "crimson"}}
```

## Verified ingest path

With Timescale and Mosquitto (`tests/compose/compose.scene-metadata.yml`):

1. `SCENE_METADATA_MQTT_ENABLED=true BROKER=127.0.0.1:1883 python3 -m scene_metadata.server`
2. Publish `scenescape/data/scene/{scene}/person`, `scenescape/event/region/{scene}/{region}/count`, and `scenescape/event/tripwire/{scene}/{tripwire}/objects`.
3. `GET /v1/identities/{id}/timeline` returns `track_started`, `region_enter`, and `tripwire_cross`.
4. `python3 -m scene_metadata.mcp_server` exposes the same reads (`mcp>=1.12,<2`).

## Ingest topics

- `scenescape/data/scene/+/+` — track_started / track_lost (Option A)
- `scenescape/event/region/+/+/+` — region analytics (full fidelity)
- `scenescape/event/tripwire/+/+/+` — tripwire crossings (full fidelity)
