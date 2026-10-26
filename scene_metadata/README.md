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
`tripwire_crossings`, `identity_attributes`. Operator: `get_system_status`.

`query_events` takes an `attributes` map of metadata name to label. Names are
not registered in the server. A classifier field is queryable as soon as it
is stored on an event, including fields from
`person-attributes-recognition-crossroad-0238` (`is_male`, `has_bag`, …)
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

## Person attribute accuracy

Lab_000 tracks have no appearance attributes. This check does not compare
against that dataset. It runs
`person-attributes-recognition-crossroad-0238` (Open Model Zoo 2023.0, FP32,
CPU) on two fixed BGR frames and requires the event log, the `HAS_ATTRIBUTE`
projection, and the MCP tools to return that model's own output.

Each of the seven heads is stored as a classificationPolicy object. `label` is
`true` when the score is above 0.5. `confidence` is the score. `model_name` is
the model id. The query server does not list the head names.

A machine with internet access downloads the IR (about 87 MB) into
`scene_metadata/.cache/` or `PERSON_ATTR_MODEL_DIR`. The weights are not in
git. Timescale must already be running. Python packages: `openvino` and
`numpy` (`pip install 'openvino==2025.3.0' numpy`).

```bash
export SCENE_METADATA_DATABASE_URL=postgresql://scenescape:scenescape@127.0.0.1:5432/scene_metadata
PYTHONPATH=scene_common/src:scene_metadata/src python3 -m scene_metadata.person_attribute_check
```

## Ingest topics

- `scenescape/data/scene/+/+` — track_started / track_lost (Option A)
- `scenescape/event/region/+/+/+` — region analytics (full fidelity)
- `scenescape/event/tripwire/+/+/+` — tripwire crossings (full fidelity)
