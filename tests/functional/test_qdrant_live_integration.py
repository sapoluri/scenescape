#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Live integration tests against a running Qdrant instance (no full compose stack)."""

import json
import uuid

import numpy as np
import pytest

from controller.qdrant_adapter import QdrantDatabase


@pytest.fixture
def qdrant_db():
  db = QdrantDatabase(hostname="localhost", port=6333, use_tls=False)
  db.connect()
  if not db.connected:
    pytest.skip("Qdrant is not available on localhost:6333")
  return db


def test_live_schema_and_vector_operations(qdrant_db):
  set_name = f"reid_test_{uuid.uuid4().hex[:8]}"
  qdrant_db.set_name = set_name
  qdrant_db.similarity_metric = "L2"
  qdrant_db.ensureSchema(256)

  vec1 = np.random.rand(256).astype(np.float32)
  vec2 = vec1 + np.random.rand(256).astype(np.float32) * 0.01
  vec3 = np.random.rand(256).astype(np.float32)

  qdrant_db.addEntry(
    "uuid-1", "track-1", "person", [vec1], set_name=set_name,
    gender={"label": "Female", "confidence": 0.95},
    run_id="integration-test")
  qdrant_db.addEntry(
    "uuid-2", "track-2", "person", [vec3], set_name=set_name,
    gender={"label": "Male", "confidence": 0.95},
    run_id="integration-test")

  matches = qdrant_db.findMatches(
    "person", [vec2], set_name=set_name, k_neighbors=2,
    gender={"label": "Female", "confidence": 0.95})
  assert matches and matches[0]
  assert matches[0][0]["uuid"] == "uuid-1"


def test_live_persist_attributes(qdrant_db):
  set_name = f"reid_persist_{uuid.uuid4().hex[:8]}"
  qdrant_db.set_name = set_name
  qdrant_db.similarity_metric = "L2"
  qdrant_db.ensureSchema(256)

  vec = np.random.rand(256).astype(np.float32)
  qdrant_db.addEntry(
    "persist-uuid", "track-9", "person", [vec], set_name=set_name,
    persist={"gender": "Female", "timestamp": 100})
  qdrant_db.addEntry(
    "persist-uuid", "track-9", "person", [vec], set_name=set_name,
    persist={"gender": "Male", "timestamp": 200})

  attrs = qdrant_db.getPersistedAttributes("persist-uuid", set_name=set_name)
  assert attrs == {"gender": "Male"}
