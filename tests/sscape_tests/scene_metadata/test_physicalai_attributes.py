# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lab_000 attribute facts land in the log, the graph, and the query API."""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scene_metadata.db import Database
from scene_metadata.physicalai_demo import assert_attribute_results, expected_labels
from scene_metadata.physicalai_loader import SCENE_ID, load_directory
from scene_metadata.query_app import create_app
from scene_metadata.store import EventStore

DATA = Path(os.environ.get(
  'PHYSICALAI_LAB_DIR',
  '/workspace/tools/reid_evaluation/output/physicalai-lab-data',
))

pytestmark_data = pytest.mark.skipif(
  not os.environ.get('SCENE_METADATA_DATABASE_URL') or not (DATA / 'ground_truth.json').is_file(),
  reason='Timescale and PhysicalAI Lab_000 files are required',
)

QUERY_SOURCES = (
  'attributes.py',
  'store.py',
  'query_app.py',
  'mcp_server.py',
  'projection.py',
)


def test_query_path_does_not_name_lab000_attributes():
  """The filter and the graph projection accept whatever key was stored."""
  root = Path(__file__).resolve().parents[3] / 'scene_metadata' / 'src' / 'scene_metadata'
  source = '\n'.join((root / name).read_text(encoding='utf-8') for name in QUERY_SOURCES)
  assert 'north_entries' not in source
  assert 'cameras_before_north' not in source


@pytestmark_data
def test_lab000_attributes_are_stored_projected_and_queryable():
  """HTTP checks the same reads the MCP tools call."""
  import uuid

  deployment = f'lab-attr-{uuid.uuid4().hex[:8]}'
  os.environ['SCENE_METADATA_DEPLOYMENT_ID'] = deployment
  ground_truth = json.loads((DATA / 'ground_truth.json').read_text(encoding='utf-8'))
  expected = expected_labels(ground_truth)
  db = Database()
  db.open()
  inserted = EventStore(db).insert_events(load_directory(DATA, deployment))
  assert inserted > 0
  client = TestClient(create_app(db))

  def fetch_events(attributes, event_types=None):
    body = {'scene_id': SCENE_ID, 'attributes': attributes, 'limit': 50}
    if event_types:
      body['event_types'] = event_types
    response = client.post('/v1/events/query', json=body)
    assert response.status_code == 200
    return sorted({row['identity_id'] for row in response.json()['events']})

  def fetch_attributes(identity):
    response = client.get(f'/v1/graph/identities/{identity}/attributes')
    assert response.status_code == 200
    rows = response.json()['attributes']
    assert rows
    assert {row['model_name'] for row in rows} == {'physicalai-ground-truth'}
    assert {row['confidence'] for row in rows} == {1.0}
    return {row['name']: row['label'] for row in rows}

  assert_attribute_results(fetch_events, fetch_attributes, expected, ground_truth)
  rejected = client.post('/v1/events/query', json={'attributes': {'': '1'}})
  assert rejected.status_code == 422
  db.close()
