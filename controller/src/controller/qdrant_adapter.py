# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import threading
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from controller.reid import ReIDDatabase
from controller.reid_constants import (
  COSINE_SIMILARITY_TOLERANCE,
  K_NEIGHBORS,
  SCHEMA_MARKER_COLLECTION,
  SCHEMA_NAME,
  SIMILARITY_METRIC,
)
from controller.reid_constraints import build_query_constraints
from scene_common import log

DEFAULT_HOSTNAME = os.getenv("QDRANT_HOSTNAME", "qdrant.scenescape.intel.com")
DEFAULT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
DEFAULT_API_KEY = os.getenv("QDRANT_API_KEY")
DEFAULT_USE_TLS = os.getenv("QDRANT_USE_TLS", "false").strip().lower() in (
  "1", "true", "yes", "on")
DEFAULT_CONFIDENCE_THRESHOLD = float(
  os.getenv(
    "QDRANT_CONFIDENCE_THRESHOLD",
    os.getenv("VDMS_CONFIDENCE_THRESHOLD", "0.8")))


class QdrantDatabase(ReIDDatabase):
  def __init__(self, set_name=SCHEMA_NAME,
               similarity_metric=SIMILARITY_METRIC, dimensions=None,
               confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
               hostname=DEFAULT_HOSTNAME, port=DEFAULT_PORT,
               api_key=DEFAULT_API_KEY, use_tls=DEFAULT_USE_TLS):
    self.set_name = set_name
    self.similarity_metric = similarity_metric
    self.dimensions = dimensions
    self.confidence_threshold = confidence_threshold
    self.hostname = hostname
    self.port = port
    self.api_key = api_key
    self.use_tls = use_tls
    self.client = None
    self.connected = False
    self.lock = threading.Lock()
    self._schema_lock = threading.Lock()
    self._schema_ready = False
    return

  def _usesInnerProductMetric(self):
    """Return True when descriptor metric is Inner Product."""
    metric = str(self.similarity_metric).strip().upper()
    return metric == "IP"

  def _isValidSimilarityScore(self, score):
    """Validate similarity score according to active metric semantics."""
    try:
      value = float(score)
    except (TypeError, ValueError):
      return False

    if not np.isfinite(value):
      return False

    if self._usesInnerProductMetric() and (
        value < -(1.0 + COSINE_SIMILARITY_TOLERANCE) or
        value > (1.0 + COSINE_SIMILARITY_TOLERANCE)):
      return False

    return True

  def _qdrantDistance(self):
    """Map configured descriptor metric to Qdrant distance function."""
    if self._usesInnerProductMetric():
      return models.Distance.DOT
    return models.Distance.EUCLID

  def _toSimilarityScore(self, qdrant_score):
    """
    Convert Qdrant query score to VDMS-compatible _distance semantics.

    query_points returns positive Euclidean distance and the raw dot product
    for DOT metrics. Older search() returned negative Euclidean distance.
    """
    if self._usesInnerProductMetric():
      return float(qdrant_score)
    return float(abs(qdrant_score))

  def _createClient(self):
    return QdrantClient(
      host=self.hostname,
      port=self.port,
      api_key=self.api_key,
      https=self.use_tls,
      prefer_grpc=False,
      check_compatibility=False,
    )

  def connect(self, hostname=None):
    if hostname is not None:
      self.hostname = hostname
    try:
      with self.lock:
        self.client = self._createClient()
        self.client.get_collections()
        self.connected = True
      if self.dimensions is not None:
        with self._schema_lock:
          self.ensureSchemaInner(
            int(self.dimensions),
            str(self.similarity_metric).strip().upper(),
            "connect")
          self._schema_ready = True
    except Exception as e:
      self.connected = False
      log.warning(f"Failed to connect to Qdrant: {e}")
    return

  def _ensureClient(self):
    if self.client is None or not self.connected:
      raise RuntimeError("Qdrant client is not connected")

  def _collectionExists(self, collection_name):
    self._ensureClient()
    try:
      self.client.get_collection(collection_name)
      return True
    except (UnexpectedResponse, ValueError):
      return False

  def _createCollection(self, collection_name, dimensions, metric):
    self._ensureClient()
    distance = models.Distance.DOT if str(metric).strip().upper() == "IP" else models.Distance.EUCLID
    self.client.create_collection(
      collection_name=collection_name,
      vectors_config=models.VectorParams(size=dimensions, distance=distance),
    )

  def _ensureMarkerCollection(self):
    if self._collectionExists(SCHEMA_MARKER_COLLECTION):
      return
    self.client.create_collection(
      collection_name=SCHEMA_MARKER_COLLECTION,
      vectors_config=models.VectorParams(size=1, distance=models.Distance.EUCLID),
    )

  def _markerPointId(self, set_name):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"reid-schema-marker:{set_name}"))

  def _writeSchemaMarker(self, dimensions, metric, skip_exists_check=False):
    self._ensureMarkerCollection()
    if not skip_exists_check:
      marker_exists, _, _ = self._readSchemaMarker()
      if marker_exists:
        log.debug(
          f"_writeSchemaMarker: Marker already exists for '{self.set_name}', skipping write")
        return

    point_id = self._markerPointId(self.set_name)
    self.client.upsert(
      collection_name=SCHEMA_MARKER_COLLECTION,
      points=[
        models.PointStruct(
          id=point_id,
          vector=[0.0],
          payload={
            "set_name": self.set_name,
            "dimensions": int(dimensions),
            "metric": str(metric).strip().upper(),
          },
        )
      ],
      wait=True,
    )

  def _readSchemaMarker(self):
    if not self._collectionExists(SCHEMA_MARKER_COLLECTION):
      return False, None, None

    point_id = self._markerPointId(self.set_name)
    try:
      points = self.client.retrieve(
        collection_name=SCHEMA_MARKER_COLLECTION,
        ids=[point_id],
        with_payload=True,
      )
    except Exception:
      return False, None, None

    if not points:
      return False, None, None

    payload = points[0].payload or {}
    if payload.get("set_name") != self.set_name:
      return False, None, None

    dimensions = payload.get("dimensions")
    metric = payload.get("metric")
    try:
      dimensions = int(dimensions) if dimensions is not None else None
    except (TypeError, ValueError):
      dimensions = None
    if metric is not None:
      metric = str(metric)
    return True, dimensions, metric

  def addSchema(self, set_name, similarity_metric, dimensions):
    try:
      if self._collectionExists(set_name):
        return False
      self._createCollection(set_name, dimensions, similarity_metric)
      return True
    except Exception as e:
      log.warning(
        f"Failed to add collection '{set_name}' to Qdrant: {e}")
      return False

  def ensureSchemaInner(self, requested_dimensions, expected_metric, caller):
    """
  Core attempt-first schema setup shared by connect() and ensureSchema().
  Attempt collection creation first; verify against schema marker when the
  collection already exists.
    """
    collection_exists = self._collectionExists(self.set_name)
    if not collection_exists:
      self._createCollection(self.set_name, requested_dimensions, expected_metric)
      log.info(
        f"{caller}: Created collection '{self.set_name}' "
        f"({requested_dimensions}D, {expected_metric})")
      self._writeSchemaMarker(requested_dimensions, expected_metric, skip_exists_check=True)
      self.dimensions = requested_dimensions
      return

    log.debug(
      f"{caller}: Collection '{self.set_name}' already exists; "
      "verifying against schema marker.")
    marker_exists, marker_dimensions, marker_metric = self._readSchemaMarker()

    if not marker_exists:
      schema_exists, schema_dimensions, schema_metric = self.findSchemaMetadata(self.set_name)
      if not schema_exists or schema_dimensions is None or schema_metric is None:
        raise RuntimeError(
          f"{caller}: '{self.set_name}' exists but no schema marker found, and collection "
          "metadata could not be read for verification. Recreate the collection to continue.")
      if str(schema_metric).strip().upper() != expected_metric:
        raise RuntimeError(
          f"{caller}: '{self.set_name}' uses metric {schema_metric}, expected {expected_metric}. "
          "Recreate the collection with matching metric.")
      if schema_dimensions != requested_dimensions:
        raise RuntimeError(
          f"{caller}: '{self.set_name}' has {schema_dimensions} dimensions, "
          f"expected {requested_dimensions}. "
          "Recreate the collection with matching dimensions.")
      log.warning(
        f"{caller}: '{self.set_name}' exists but no schema marker found; "
        "writing marker for future instances.")
      self._writeSchemaMarker(requested_dimensions, expected_metric, skip_exists_check=True)
      self.dimensions = requested_dimensions
      return

    if marker_dimensions is None or marker_metric is None:
      raise RuntimeError(
        f"{caller}: '{self.set_name}' schema marker returned no dimensions "
        f"for verification (dimensions={marker_dimensions}, metric={marker_metric}). "
        "Cannot safely confirm compatibility.")

    if str(marker_metric).strip().upper() != expected_metric:
      raise RuntimeError(
        f"{caller}: '{self.set_name}' uses metric {marker_metric}, "
        f"expected {expected_metric}. "
        "Recreate the collection with matching metric.")
    if marker_dimensions != requested_dimensions:
      raise RuntimeError(
        f"{caller}: '{self.set_name}' has {marker_dimensions} dimensions, "
        f"expected {requested_dimensions}. "
        "Recreate the collection with matching dimensions.")

    log.info(
      f"{caller}: Verified existing collection '{self.set_name}' "
      f"against schema marker ({marker_dimensions}D, {marker_metric})")
    self.dimensions = requested_dimensions

  def ensureSchema(self, dimensions):
    with self._schema_lock:
      requested_dimensions = int(dimensions)
      if self._schema_ready:
        if int(self.dimensions) != requested_dimensions:
          raise ValueError(
            f"ReID schema already initialized with {self.dimensions} dimensions; "
            f"incoming vector has {requested_dimensions} dimensions. "
            "Restart the controller and flush the Qdrant collection to change dimensions.")
        return
      self.ensureSchemaInner(
        requested_dimensions,
        str(self.similarity_metric).strip().upper(),
        "ensureSchema")
      self._schema_ready = True

  def _buildPayload(self, uuid_value, rvid, object_type, persist=None, **metadata):
    properties = {
      "uuid": f"{uuid_value}",
      "rvid": f"{rvid}",
      "type": f"{object_type}",
    }

    if persist:
      persist = persist.copy()
      persist_timestamp = persist.pop('timestamp')
      properties["persist"] = json.dumps(persist)
      properties["persist_timestamp"] = persist_timestamp
      log.debug(
        f"[Qdrant] addEntry: Storing persist keys={list(persist.keys())} for uuid={uuid_value}")

    for key, value in metadata.items():
      if isinstance(value, dict):
        if 'label' in value:
          properties[key] = str(value['label'])
          log.debug(
            f"[Qdrant] addEntry: Extracted label '{value['label']}' from {key} metadata dict")
        else:
          properties[key] = json.dumps(value)
          log.debug(f"[Qdrant] addEntry: Serialized {key} as JSON (no label field)")
      else:
        properties[key] = str(value)

    return properties

  def addEntry(self, uuid_value, rvid, object_type, reid_vectors, set_name=SCHEMA_NAME,
               persist=None, **metadata):
    self._ensureClient()
    properties = self._buildPayload(uuid_value, rvid, object_type, persist=persist, **metadata)
    normalize_embeddings = self._usesInnerProductMetric()
    points = []

    for reid_vector in reid_vectors:
      prepared_reid = self.prepareReidDict(
        reid_vector,
        self.dimensions,
        normalize_embeddings=normalize_embeddings)
      if prepared_reid is None:
        continue

      vec_array = prepared_reid["embedded_vector"]
      points.append(models.PointStruct(
        id=str(uuid.uuid4()),
        vector=vec_array.tolist(),
        payload=properties.copy(),
      ))

    if not points:
      log.warning(
        "addEntry: No valid vectors to add (all skipped due to dimension mismatch "
        "or uninitialized dimensions)")
      return

    try:
      with self.lock:
        self.client.upsert(collection_name=set_name, points=points, wait=True)
    except Exception as e:
      log.error(f"addEntry: Failed to upsert {len(points)} vectors to Qdrant: {e}")
    return

  def getPersistedAttributes(self, uuid_value, set_name=SCHEMA_NAME):
    self._ensureClient()
    query_filter = self._buildQdrantFilter({"uuid": ["==", f"{uuid_value}"]})
    try:
      points, _ = self.client.scroll(
        collection_name=set_name,
        scroll_filter=query_filter,
        limit=1000,
        with_payload=True,
        with_vectors=False,
      )
    except Exception as e:
      log.debug(f"[Qdrant] getPersistedAttributes: Query failed for uuid={uuid_value}: {e}")
      return {}

    if not points:
      log.debug(f"[Qdrant] getPersistedAttributes: No entry found for uuid={uuid_value}")
      return {}

    points_with_persist = [
      point for point in points
      if isinstance(point.payload, dict) and
      isinstance(point.payload.get('persist'), str) and
      point.payload.get('persist').strip() and
      point.payload.get('persist') != 'Missing property'
    ]

    if not points_with_persist:
      log.debug(f"[Qdrant] getPersistedAttributes: No persist data found for uuid={uuid_value}")
      return {}

    latest = max(
      points_with_persist,
      key=lambda point: point.payload.get('persist_timestamp', 0))
    try:
      return json.loads(latest.payload['persist'])
    except (json.JSONDecodeError, TypeError, KeyError) as e:
      log.warning(
        f"[Qdrant] getPersistedAttributes: Failed to deserialize persist for "
        f"uuid={uuid_value}: {e}")
      return {}

  def findSchema(self, set_name):
    schema_exists, _ = self.findSchemaDetails(set_name)
    return schema_exists

  def findSchemaDetails(self, set_name):
    schema_exists, schema_dimensions, _ = self.findSchemaMetadata(set_name)
    return schema_exists, schema_dimensions

  def findSchemaMetadata(self, set_name):
    if not self._collectionExists(set_name):
      return False, None, None

    marker_exists, marker_dimensions, marker_metric = self._readSchemaMarker()
    if marker_exists:
      return True, marker_dimensions, marker_metric

    try:
      collection = self.client.get_collection(set_name)
      schema_dimensions = collection.config.params.vectors.size
      distance = collection.config.params.vectors.distance
      if distance == models.Distance.DOT:
        schema_metric = "IP"
      else:
        schema_metric = "L2"
      return True, int(schema_dimensions), schema_metric
    except Exception as e:
      log.warning(f"findSchemaMetadata: Failed to read collection '{set_name}': {e}")
      return False, None, None

  def _buildQdrantFilter(self, query_constraints):
    must_conditions = []
    for key, constraint in query_constraints.items():
      if not isinstance(constraint, (list, tuple)) or len(constraint) < 2:
        continue
      operator = str(constraint[0]).strip()
      value = constraint[1]
      if operator != "==":
        log.debug(f"[Qdrant] Skipping unsupported constraint operator '{operator}' for {key}")
        continue
      must_conditions.append(models.FieldCondition(
        key=key,
        match=models.MatchValue(value=str(value)),
      ))

    if not must_conditions:
      return None
    return models.Filter(must=must_conditions)

  def _buildQueryConstraints(self, object_type, **constraints):
    return build_query_constraints(
      object_type,
      confidence_threshold=self.confidence_threshold,
      **constraints)

  def findMatches(self, object_type, reid_vectors, set_name=SCHEMA_NAME,
                  k_neighbors=K_NEIGHBORS, **constraints):
    log.debug(
      f"[Qdrant] findMatches called: object_type={object_type}, k_neighbors={k_neighbors}")
    log.debug(f"[Qdrant] findMatches constraints received: {constraints}")

    self._ensureClient()
    query_constraints = self._buildQueryConstraints(object_type, **constraints)
    query_filter = self._buildQdrantFilter(query_constraints)
    log.debug(f"[Qdrant] Executing TIER 1 find with constraints: {query_constraints}")

    normalize_embeddings = self._usesInnerProductMetric()
    query_vectors = []
    for reid_vector in reid_vectors:
      vec_array = self.prepareReidVector(
        reid_vector,
        self.dimensions,
        normalize_embeddings=normalize_embeddings)
      if vec_array is None:
        continue
      query_vectors.append(vec_array.tolist())

    if not query_vectors:
      log.warning("findMatches: No valid vectors for similarity search")
      return None

    result = []
    for query_vector in query_vectors:
      try:
        with self.lock:
          response = self.client.query_points(
            collection_name=set_name,
            query=query_vector,
            query_filter=query_filter,
            limit=k_neighbors,
            with_payload=True,
          )
          hits = response.points
      except Exception as e:
        log.warning(f"[Qdrant] findMatches search failed: {e}")
        result.append([])
        continue

      valid_entities = []
      for hit in hits:
        payload = hit.payload or {}
        similarity = self._toSimilarityScore(hit.score)
        if not self._isValidSimilarityScore(similarity):
          log.warning(
            f"findMatches: Discarding entity with invalid similarity score "
            f"{similarity} for metric {self.similarity_metric}")
          continue
        valid_entities.append({
          "uuid": payload.get("uuid"),
          "rvid": payload.get("rvid"),
          "_distance": similarity,
        })

      result.append(valid_entities)

    log.debug(
      f"[Qdrant] findMatches returned {len(result)} per-vector result item(s) from "
      f"{len(query_vectors)} valid query vector(s)")
    return result
