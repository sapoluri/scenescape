# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared ReID environment variable resolution.

Connection and tuning knobs use REID_* names and are backend-agnostic.
Only REID_DATABASE selects which adapter runs. Legacy VDMS_* / QDRANT_* names
remain as one-release fallbacks.
"""

import os

from scene_common import log

DEFAULT_DATABASE = "VDMS"
DEFAULT_HOSTNAME = "reid.scenescape.intel.com"
DEFAULT_PORT = 55555
DEFAULT_USE_TLS = True
DEFAULT_CONFIDENCE_THRESHOLD = 0.8
DEFAULT_CA_CERT = "/run/secrets/certs/scenescape-ca.pem"
DEFAULT_CLIENT_CERT = "/run/secrets/certs/scenescape-reid.crt"
DEFAULT_CLIENT_KEY = "/run/secrets/certs/scenescape-reid.key"

_LEGACY_WARNED = set()


def _warn_legacy(legacy_name, canonical_name):
  if legacy_name in _LEGACY_WARNED:
    return
  _LEGACY_WARNED.add(legacy_name)
  log.warning(
    f"{legacy_name} is deprecated; use {canonical_name} instead "
    f"(REID_DATABASE selects the vector backend)")


def _env_first(*names, default=None):
  """Return the first set environment value among names, else default."""
  for name in names:
    value = os.getenv(name)
    if value is not None and str(value).strip() != "":
      return value
  return default


def get_reid_database():
  """Return selected ReID backend name (uppercase)."""
  return str(_env_first("REID_DATABASE", default=DEFAULT_DATABASE)).strip().upper()


def get_reid_hostname():
  """Return shared ReID database hostname."""
  value = os.getenv("REID_HOSTNAME")
  if value is not None and str(value).strip() != "":
    return value.strip()

  for legacy_name in ("VDMS_HOSTNAME", "QDRANT_HOSTNAME"):
    legacy = os.getenv(legacy_name)
    if legacy is not None and str(legacy).strip() != "":
      _warn_legacy(legacy_name, "REID_HOSTNAME")
      return legacy.strip()

  return DEFAULT_HOSTNAME


def get_reid_port():
  """Return shared ReID database port."""
  value = os.getenv("REID_PORT")
  if value is not None and str(value).strip() != "":
    return int(value)

  legacy = os.getenv("QDRANT_PORT")
  if legacy is not None and str(legacy).strip() != "":
    _warn_legacy("QDRANT_PORT", "REID_PORT")
    return int(legacy)

  return int(DEFAULT_PORT)


def get_reid_use_tls():
  """Return whether TLS should be used for the ReID database connection."""
  value = os.getenv("REID_USE_TLS")
  if value is not None and str(value).strip() != "":
    return str(value).strip().lower() in ("1", "true", "yes", "on")

  legacy = os.getenv("QDRANT_USE_TLS")
  if legacy is not None and str(legacy).strip() != "":
    _warn_legacy("QDRANT_USE_TLS", "REID_USE_TLS")
    return str(legacy).strip().lower() in ("1", "true", "yes", "on")

  return bool(DEFAULT_USE_TLS)


def get_reid_api_key():
  """Return optional ReID API key (used by backends that support it)."""
  value = os.getenv("REID_API_KEY")
  if value is not None and str(value).strip() != "":
    return value

  legacy = os.getenv("QDRANT_API_KEY")
  if legacy is not None and str(legacy).strip() != "":
    _warn_legacy("QDRANT_API_KEY", "REID_API_KEY")
    return legacy
  return None


def get_reid_confidence_threshold():
  """Return TIER 1 metadata confidence threshold."""
  value = _env_first(
    "REID_CONFIDENCE_THRESHOLD",
    "QDRANT_CONFIDENCE_THRESHOLD",
    "VDMS_CONFIDENCE_THRESHOLD",
    default=None,
  )
  if value is None:
    return float(DEFAULT_CONFIDENCE_THRESHOLD)

  if os.getenv("REID_CONFIDENCE_THRESHOLD") is None:
    if os.getenv("QDRANT_CONFIDENCE_THRESHOLD") is not None:
      _warn_legacy("QDRANT_CONFIDENCE_THRESHOLD", "REID_CONFIDENCE_THRESHOLD")
    elif os.getenv("VDMS_CONFIDENCE_THRESHOLD") is not None:
      _warn_legacy("VDMS_CONFIDENCE_THRESHOLD", "REID_CONFIDENCE_THRESHOLD")
  return float(value)


def get_reid_ca_cert():
  """Return CA certificate path for TLS backends."""
  value = os.getenv("REID_CA_CERT")
  if value is not None and str(value).strip() != "":
    return value
  legacy = os.getenv("VDMS_CA_CERT")
  if legacy is not None and str(legacy).strip() != "":
    _warn_legacy("VDMS_CA_CERT", "REID_CA_CERT")
    return legacy
  return DEFAULT_CA_CERT


def get_reid_client_cert():
  """Return client certificate path for mTLS backends."""
  value = os.getenv("REID_CLIENT_CERT")
  if value is not None and str(value).strip() != "":
    return value
  legacy = os.getenv("VDMS_CLIENT_CERT")
  if legacy is not None and str(legacy).strip() != "":
    _warn_legacy("VDMS_CLIENT_CERT", "REID_CLIENT_CERT")
    return legacy
  return DEFAULT_CLIENT_CERT


def get_reid_client_key():
  """Return client key path for mTLS backends."""
  value = os.getenv("REID_CLIENT_KEY")
  if value is not None and str(value).strip() != "":
    return value
  legacy = os.getenv("VDMS_CLIENT_KEY")
  if legacy is not None and str(legacy).strip() != "":
    _warn_legacy("VDMS_CLIENT_KEY", "REID_CLIENT_KEY")
    return legacy
  return DEFAULT_CLIENT_KEY
