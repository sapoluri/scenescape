# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
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
from controller.reid_env import (
  get_reid_api_key,
  get_reid_ca_cert,
  get_reid_confidence_threshold,
  get_reid_hostname,
  get_reid_port,
  get_reid_use_tls,
)
from scene_common import log


class QdrantDatabase(ReIDDatabase):
  def __init__(self, set_name=SCHEMA_NAME,
               similarity_metric=SIMILARITY_METRIC, dimensions=None,
               confidence_threshold=None,
               hostname=None, port=None,
               api_key=None, use_tls=None, ca_cert=None):
    self.set_name = set_name
    self.similarity_metric = similarity_metric
    self.dimensions = dimensions
    self.confidence_threshold = (
      get_reid_confidence_threshold() if confidence_threshold is None
      else confidence_threshold)
    self.hostname = get_reid_hostname() if hostname is None else hostname
    resolved_port = get_reid_port() if port is None else port
    self.port = int(resolved_port)
    self.api_key = get_reid_api_key() if api_key is None else api_key
    self.use_tls = get_reid_use_tls() if use_tls is None else use_tls
    self.ca_cert = get_reid_ca_cert() if ca_cert is None else ca_cert
    self.client = None
    self.connected = False
    self.lock = threading.Lock()
    self._schema_lock = threading.Lock()
    self._schema_ready = False
    return

  def _usesInnerProductMetric(self, metric=None):
    """Return True when descriptor metric is Inner Product."""
    if metric is None:
      metric = self.similarity_metric
    return str(metric).strip().upper() == "IP"

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

  def _qdrantDistance(self, metric=None):
    """Map descriptor metric to Qdrant distance function."""
    if self._usesInnerProductMetric(metric):
      return models.Distance.DOT
    return models.Distance.EUCLID

  @staticmethod
  def _metricFromQdrantDistance(distance):
    """Map Qdrant distance function back to descriptor metric name."""
    if distance == models.Distance.DOT:
      return "IP"
    return "L2"

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
    client_kwargs = {
      "host": self.hostname,
      "port": self.port,
      "api_key": self.api_key,
      "https": self.use_tls,
      "prefer_grpc": False,
      "check_compatibility": False,
    }
    if self.use_tls and self.ca_cert:
      client_kwargs["verify"] = self.ca_cert
    return QdrantClient(**client_kwargs)

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
    self.client.create_collection(
      collection_name=collection_name,
      vectors_config=models.VectorParams(
        size=dimensions, distance=self._qdrantDistance(metric)),
    )
    self._ensurePayloadIndexes(collection_name)

  def _ensurePayloadIndexes(self, collection_name):
    """Ensure payload indexes used for UUID filter and latest-persist lookup."""
    self._ensureClient()
    try:
      collection = self.client.get_collection(collection_name)
      existing = set((collection.payload_schema or {}).keys())
    except Exception as e:
      log.debug(
        f"_ensurePayloadIndexes: Could not read payload schema for "
        f"'{collection_name}': {e}")
      existing = set()

    desired = (
      ("uuid", models.PayloadSchemaType.KEYWORD),
      ("persist_timestamp", models.PayloadSchemaType.FLOAT),
    )
    for field_name, field_schema in desired:
      if field_name in existing:
        continue
      try:
        self.client.create_payload_index(
          collection_name=collection_name,
          field_name=field_name,
          field_schema=field_schema,
          wait=True,
        )
      except Exception as e:
        log.warning(
          f"_ensurePayloadIndexes: Failed to create index '{field_name}' "
          f"on '{collection_name}': {e}")

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
      self._ensurePayloadIndexes(self.set_name)
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
    self._ensurePayloadIndexes(self.set_name)
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

  def _scrollMatchingPoints(self, collection_name, query_filter, page_size=100):
    """Scroll all points matching filter. Fallback when ordered scroll is unavailable."""
    self._ensureClient()
    points = []
    offset = None
    while True:
      batch, offset = self.client.scroll(
        collection_name=collection_name,
        scroll_filter=query_filter,
        limit=page_size,
        offset=offset,
        with_payload=["persist", "persist_timestamp"],
        with_vectors=False,
      )
      points.extend(batch)
      if offset is None:
        break
    return points

  def _latestPersistFromPoints(self, points, uuid_value):
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

  def getPersistedAttributes(self, uuid_value, set_name=SCHEMA_NAME):
    """
    Retrieve the most recent persist attributes stored for a given object UUID.

    Prefers ordered scroll by persist_timestamp DESC (O(1) in history length).
    Falls back to a full filtered scroll when ordered lookup is unavailable
    (for example collections created before payload indexes existed).
    """
    query_filter = self._buildQdrantFilter({"uuid": ["==", f"{uuid_value}"]})
    try:
      points, _ = self.client.scroll(
        collection_name=set_name,
        scroll_filter=query_filter,
        limit=1,
        with_payload=["persist", "persist_timestamp"],
        with_vectors=False,
        order_by=models.OrderBy(
          key="persist_timestamp",
          direction=models.Direction.DESC,
        ),
      )
    except Exception as e:
      log.debug(
        f"[Qdrant] getPersistedAttributes: Ordered scroll failed for "
        f"uuid={uuid_value}, falling back to full scroll: {e}")
      try:
        points = self._scrollMatchingPoints(set_name, query_filter)
      except Exception as fallback_error:
        log.debug(
          f"[Qdrant] getPersistedAttributes: Query failed for uuid={uuid_value}: "
          f"{fallback_error}")
        return {}

    if not points:
      log.debug(f"[Qdrant] getPersistedAttributes: No entry found for uuid={uuid_value}")
      return {}

    return self._latestPersistFromPoints(points, uuid_value)

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
      schema_metric = self._metricFromQdrantDistance(
        collection.config.params.vectors.distance)
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
