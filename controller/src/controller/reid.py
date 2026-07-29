# SPDX-FileCopyrightText: (C) 2024 - 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import threading

import numpy as np

from controller.reid_constants import (
  SCHEMA_NAME,
  SIMILARITY_METRIC,
  is_within_inner_product_range,
)
from controller.reid_constraints import build_query_constraints
from controller.reid_env import get_reid_confidence_threshold
from scene_common import log

class ReIDDatabase(ABC):
  def __init__(self, set_name=SCHEMA_NAME, similarity_metric=SIMILARITY_METRIC,
               dimensions=None, confidence_threshold=None):
    """Establish the backend-agnostic state shared by every ReID adapter.

    Subclasses must call this before using any inherited helper, since
    similarity scoring, schema lifecycle, and TIER 1 constraint building
    read these attributes.
    """
    self.set_name = set_name
    self.similarity_metric = similarity_metric
    self.dimensions = dimensions
    self.confidence_threshold = (
      get_reid_confidence_threshold() if confidence_threshold is None
      else confidence_threshold)
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

    if self._usesInnerProductMetric() and not is_within_inner_product_range(value):
      return False

    return True

  def _buildQueryConstraints(self, object_type, **constraints):
    """Build TIER 1 metadata filtering constraints for this adapter."""
    return build_query_constraints(
      object_type,
      confidence_threshold=self.confidence_threshold,
      **constraints)

  def prepareReidDict(self, embedding_vector, dimensions=None,
                        normalize_embeddings=False):
    """Prepare a normalized/validated ReID payload from arbitrary vector shapes.

    Supports vectors shaped as (N,), (1, N), or any array-like object by
    flattening to 1D. If dimensions is None, dimensions are inferred from the
    flattened vector length.
    """
    if embedding_vector is None:
      log.warning("prepareReidDict: Empty embedding vector, skipping this vector")
      return None

    vec_array = np.asarray(embedding_vector, dtype="float32").reshape(-1)
    inferred_dimensions = int(vec_array.shape[0])
    expected_dimensions = inferred_dimensions if dimensions is None else int(dimensions)

    if inferred_dimensions != expected_dimensions:
      log.warning(
        f"prepareReidDict: Expected vector shape ({expected_dimensions},) but got {vec_array.shape}, skipping this vector")
      return None

    if not np.all(np.isfinite(vec_array)):
      log.warning("prepareReidDict: Vector contains non-finite values, skipping this vector")
      return None

    if normalize_embeddings:
      norm = np.linalg.norm(vec_array)
      if not np.isfinite(norm) or norm == 0.0:
        log.warning(f"prepareReidDict: Invalid vector norm ({norm}), skipping this vector")
        return None
      vec_array = vec_array / norm

    return {
      "embedded_vector": vec_array.astype("float32", copy=False),
      "dimensions": expected_dimensions,
    }

  def prepareReidVector(self, reid_vector, dimensions,
                           normalize_embeddings=False):
    """Backward-compatible wrapper returning only the prepared vector."""
    prepared_reid = self.prepareReidDict(
      reid_vector,
      dimensions,
      normalize_embeddings=normalize_embeddings)
    if prepared_reid is None:
      return None
    return prepared_reid["embedded_vector"]

  @abstractmethod
  def _schemaResourceLabel(self):
    """Human-readable name for this backend's schema resource (for logs/errors)."""
    return

  @abstractmethod
  def _tryCreateSchema(self, dimensions, metric):
    """
    Attempt to create the schema resource for self.set_name.

    @param   dimensions  Embedding dimensionality
    @param   metric      Backend similarity metric (e.g. 'L2', 'IP')
    @return  bool        True if the schema was newly created; False if it
                         already existed. Raise on unrecoverable failure.
    """
    return

  @abstractmethod
  def _readSchemaMarker(self):
    """
    Read the schema marker for self.set_name.

    @return  (exists, dimensions, metric)  (False, None, None) when missing.
    """
    return

  @abstractmethod
  def _persistSchemaMarker(self, dimensions, metric):
    """Write the schema marker for self.set_name (unconditional)."""
    return

  def _writeSchemaMarker(self, dimensions, metric, skip_exists_check=False):
    """Write schema marker, optionally skipping a prior existence probe."""
    if not skip_exists_check:
      marker_exists, _, _ = self._readSchemaMarker()
      if marker_exists:
        log.debug(
          f"_writeSchemaMarker: Marker already exists for '{self.set_name}', skipping write")
        return
    self._persistSchemaMarker(dimensions, metric)

  def _afterSchemaVerified(self):
    """Optional hook after an existing schema is verified (e.g. ensure indexes)."""
    return

  def ensureSchemaInner(self, requested_dimensions, expected_metric, caller):
    """
    Core attempt-first schema setup shared by connect() and ensureSchema().

    Attempt creation first; verify against the schema marker when the resource
    already exists. Backends differ only in create/marker I/O hooks.
    """
    label = self._schemaResourceLabel()
    created = self._tryCreateSchema(requested_dimensions, expected_metric)
    if created:
      log.info(
        f"{caller}: Created {label} '{self.set_name}' "
        f"({requested_dimensions}D, {expected_metric})")
      self._writeSchemaMarker(requested_dimensions, expected_metric, skip_exists_check=True)
      self.dimensions = requested_dimensions
      return

    log.debug(
      f"{caller}: '{self.set_name}' already exists; "
      "verifying against schema marker.")
    marker_exists, marker_dimensions, marker_metric = self._readSchemaMarker()

    if not marker_exists:
      # Backward-compat: resource exists but marker is missing. Probe native
      # metadata once (safe here because create already indicated existence)
      # before writing a marker that other controllers will treat as authoritative.
      schema_exists, schema_dimensions, schema_metric = self.findSchemaMetadata(self.set_name)
      if not schema_exists or schema_dimensions is None or schema_metric is None:
        raise RuntimeError(
          f"{caller}: '{self.set_name}' exists but no schema marker found, and {label} "
          f"metadata could not be read for verification. Recreate the {label} to continue.")
      if str(schema_metric).strip().upper() != expected_metric:
        raise RuntimeError(
          f"{caller}: '{self.set_name}' uses metric {schema_metric}, expected {expected_metric}. "
          f"Recreate the {label} with matching metric.")
      if schema_dimensions != requested_dimensions:
        raise RuntimeError(
          f"{caller}: '{self.set_name}' has {schema_dimensions} dimensions, "
          f"expected {requested_dimensions}. "
          f"Recreate the {label} with matching dimensions.")
      log.warning(
        f"{caller}: '{self.set_name}' exists but no schema marker found; "
        "writing marker for future instances.")
      self._writeSchemaMarker(requested_dimensions, expected_metric, skip_exists_check=True)
      self._afterSchemaVerified()
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
        f"Recreate the {label} with matching metric.")
    if marker_dimensions != requested_dimensions:
      raise RuntimeError(
        f"{caller}: '{self.set_name}' has {marker_dimensions} dimensions, "
        f"expected {requested_dimensions}. "
        f"Recreate the {label} with matching dimensions.")

    log.info(
      f"{caller}: Verified existing {label} '{self.set_name}' "
      f"against schema marker ({marker_dimensions}D, {marker_metric})")
    self._afterSchemaVerified()
    self.dimensions = requested_dimensions

  def ensureSchema(self, dimensions):
    """Ensure ReID schema exists and matches the requested dimensions/metric."""
    with self._schema_lock:
      requested_dimensions = int(dimensions)
      if self._schema_ready:
        if int(self.dimensions) != requested_dimensions:
          label = self._schemaResourceLabel()
          raise ValueError(
            f"ReID schema already initialized with {self.dimensions} dimensions; "
            f"incoming vector has {requested_dimensions} dimensions. "
            f"Restart the controller and flush the {label} to change dimensions.")
        return
      self.ensureSchemaInner(
        requested_dimensions,
        str(self.similarity_metric).strip().upper(),
        "ensureSchema")
      self._schema_ready = True

  def findSchema(self, set_name):
    """Return True when a schema with the given name exists."""
    schema_exists, _ = self.findSchemaDetails(set_name)
    return schema_exists

  def findSchemaDetails(self, set_name):
    """Return (exists, dimensions) for the named schema."""
    schema_exists, schema_dimensions, _ = self.findSchemaMetadata(set_name)
    return schema_exists, schema_dimensions

  @abstractmethod
  def findSchemaMetadata(self, set_name):
    """
    Return native schema metadata for the named set/collection.

    @param   set_name  Name of the schema resource
    @return  (exists, dimensions, metric)
    """
    return

  @abstractmethod
  def connect(self, hostname):
    """
    Connect to the database using the specified hostname

    @param   hostname  Hostname of the database
    @return  None
    """
    return

  @abstractmethod
  def addSchema(self, set_name, similarity_metric, dimensions):
    """
    Add a schema to the database for storing the Re-ID vectors

    @param   set_name           Name of the schema to add
    @param   similarity_metric  Metric for computing the similary scores of the Re-ID vectors
    @param   dimensions         Dimensions of the Re-ID vectors to store
    @return  None
    """
    return

  @abstractmethod
  def addEntry(self, uuid, rvid, object_type, reid_vectors, set_name, persist=None, **metadata):
    """
    Adds entries to the database for the Re-ID vectors with optional metadata

    @param   uuid         Unique ID for the object
    @param   rvid         ID of the object from the motion tracker
    @param   object_type  Class of the object (Person, Vehicle, etc.)
    @param   reid_vectors Re-ID embeddings produced by a detection model
    @param   set_name     Name of the set to add the new entry to
    @param   persist      Optional dict of persistent attributes to store alongside vectors
    @param   metadata     Optional semantic attributes (age, gender, color, etc.)
    @return  None
    """
    return

  @abstractmethod
  def getPersistedAttributes(self, uuid, set_name=None):
    """
    Retrieve the most recently stored persist attributes for a given UUID.

    @param   uuid      The object UUID to look up
    @param   set_name  Optional name of the descriptor set to query
    @return  dict      Deserialized persist attributes, or empty dict if not found
    """
    return

  @abstractmethod
  def findMatches(self, object_type, reid_vectors, set_name, k_neighbors, **constraints):
    """
    Search the database for entries with the closest similarity scores to the given vector
    using 2-tier hybrid search: TIER 1 (metadata filtering) + TIER 2 (vector similarity)

    @param   object_type  Class of the source of the reid vector (Person, Vehicle, etc.)
    @param   reid_vectors Re-ID embeddings produced by a detection model
    @param   set_name     Name of the set to find similarity scores
    @param   k_neighbors  Number of similar entries to return
    @param   constraints  Optional metadata filters (age, gender, color, etc.)
    @return  iterable     Entries with the closest similarity scores
    """
    return
