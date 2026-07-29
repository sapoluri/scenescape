# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared ReID environment variable resolution.

Connection and tuning knobs use REID_* names and are backend-agnostic.
Only REID_DATABASE selects which adapter runs.
"""

import os

DEFAULT_DATABASE = "VDMS"
DEFAULT_HOSTNAME = "reid.scenescape.intel.com"
DEFAULT_PORT = 55555
DEFAULT_USE_TLS = True
DEFAULT_CONFIDENCE_THRESHOLD = 0.8
DEFAULT_CA_CERT = "/run/secrets/certs/scenescape-ca.pem"
DEFAULT_CLIENT_CERT = "/run/secrets/certs/scenescape-reid.crt"
DEFAULT_CLIENT_KEY = "/run/secrets/certs/scenescape-reid.key"

_TRUE_VALUES = ("1", "true", "yes", "on")


def _env_value(name, default=None):
  """Return the trimmed value of name, or default when unset or blank."""
  value = os.getenv(name)
  if value is None or str(value).strip() == "":
    return default
  return str(value).strip()


def get_reid_database():
  """Return selected ReID backend name (uppercase)."""
  return _env_value("REID_DATABASE", DEFAULT_DATABASE).upper()


def get_reid_hostname():
  """Return shared ReID database hostname."""
  return _env_value("REID_HOSTNAME", DEFAULT_HOSTNAME)


def get_reid_port():
  """Return shared ReID database port."""
  return int(_env_value("REID_PORT", DEFAULT_PORT))


def get_reid_use_tls():
  """Return whether TLS should be used for the ReID database connection."""
  value = _env_value("REID_USE_TLS")
  if value is None:
    return bool(DEFAULT_USE_TLS)
  return value.lower() in _TRUE_VALUES


def get_reid_api_key():
  """Return optional ReID API key (used by backends that support it)."""
  return _env_value("REID_API_KEY")


def get_reid_confidence_threshold():
  """Return TIER 1 metadata confidence threshold."""
  return float(_env_value("REID_CONFIDENCE_THRESHOLD", DEFAULT_CONFIDENCE_THRESHOLD))


def get_reid_ca_cert():
  """Return CA certificate path for TLS backends."""
  return _env_value("REID_CA_CERT", DEFAULT_CA_CERT)


def get_reid_client_cert():
  """Return client certificate path for mTLS backends."""
  return _env_value("REID_CLIENT_CERT", DEFAULT_CLIENT_CERT)


def get_reid_client_key():
  """Return client key path for mTLS backends."""
  return _env_value("REID_CLIENT_KEY", DEFAULT_CLIENT_KEY)
