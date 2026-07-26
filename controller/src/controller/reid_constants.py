# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for ReID vector database adapters."""

SCHEMA_NAME = "reid_vector"
K_NEIGHBORS = 1
SIMILARITY_METRIC = "L2"
# Tolerance applied to the theoretical [-1, 1] IP score bounds to absorb
# float32 rounding errors from vector normalization and inner-product computation.
COSINE_SIMILARITY_TOLERANCE = 1e-6
SCHEMA_MARKER_COLLECTION = "_reid_schema_markers"
