#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shared ReID environment variable resolution."""

import pytest

from controller import reid_env

_REID_ENV_NAMES = (
  "REID_DATABASE", "REID_HOSTNAME", "REID_PORT", "REID_USE_TLS", "REID_API_KEY",
  "REID_CONFIDENCE_THRESHOLD", "REID_CA_CERT", "REID_CLIENT_CERT", "REID_CLIENT_KEY",
)

_RETIRED_ENV_NAMES = (
  "VDMS_HOSTNAME", "VDMS_PORT", "VDMS_USE_TLS", "VDMS_CONFIDENCE_THRESHOLD",
  "VDMS_CA_CERT", "VDMS_CLIENT_CERT", "VDMS_CLIENT_KEY",
  "QDRANT_HOSTNAME", "QDRANT_PORT", "QDRANT_USE_TLS", "QDRANT_API_KEY",
  "QDRANT_CONFIDENCE_THRESHOLD",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
  """Isolate from developer shell env so defaults are deterministic."""
  for name in _REID_ENV_NAMES + _RETIRED_ENV_NAMES:
    monkeypatch.delenv(name, raising=False)
  yield


class TestReidEnvDefaults:
  def test_default_database_is_vdms(self):
    assert reid_env.get_reid_database() == "VDMS"

  def test_shared_connection_defaults(self):
    assert reid_env.get_reid_hostname() == "reid.scenescape.intel.com"
    assert reid_env.get_reid_port() == 55555
    assert reid_env.get_reid_use_tls() is True
    assert reid_env.get_reid_client_cert().endswith("scenescape-reid.crt")
    assert reid_env.get_reid_client_key().endswith("scenescape-reid.key")

  def test_confidence_threshold_default(self):
    assert reid_env.get_reid_confidence_threshold() == 0.8

  def test_api_key_defaults_to_none(self):
    assert reid_env.get_reid_api_key() is None


class TestReidEnvCanonicalNames:
  def test_reid_hostname_override(self, monkeypatch):
    monkeypatch.setenv("REID_HOSTNAME", "custom.example.com")
    assert reid_env.get_reid_hostname() == "custom.example.com"

  def test_reid_database_does_not_change_connection_defaults(self, monkeypatch):
    monkeypatch.setenv("REID_DATABASE", "QDRANT")
    assert reid_env.get_reid_database() == "QDRANT"
    assert reid_env.get_reid_hostname() == "reid.scenescape.intel.com"
    assert reid_env.get_reid_port() == 55555
    assert reid_env.get_reid_use_tls() is True

  def test_reid_confidence_threshold(self, monkeypatch):
    monkeypatch.setenv("REID_CONFIDENCE_THRESHOLD", "0.91")
    assert reid_env.get_reid_confidence_threshold() == 0.91

  def test_reid_use_tls_accepts_false_values(self, monkeypatch):
    monkeypatch.setenv("REID_USE_TLS", "false")
    assert reid_env.get_reid_use_tls() is False

  def test_values_are_trimmed(self, monkeypatch):
    monkeypatch.setenv("REID_HOSTNAME", "  spaced.example.com  ")
    monkeypatch.setenv("REID_PORT", " 6543 ")
    assert reid_env.get_reid_hostname() == "spaced.example.com"
    assert reid_env.get_reid_port() == 6543

  def test_blank_values_fall_back_to_defaults(self, monkeypatch):
    monkeypatch.setenv("REID_HOSTNAME", "   ")
    monkeypatch.setenv("REID_PORT", "")
    assert reid_env.get_reid_hostname() == "reid.scenescape.intel.com"
    assert reid_env.get_reid_port() == 55555


class TestRetiredBackendSpecificNames:
  """Backend-prefixed names were removed; only REID_* is honored."""

  def test_legacy_hostname_names_are_ignored(self, monkeypatch):
    monkeypatch.setenv("VDMS_HOSTNAME", "legacy-vdms.example.com")
    monkeypatch.setenv("QDRANT_HOSTNAME", "legacy-qdrant.example.com")
    assert reid_env.get_reid_hostname() == "reid.scenescape.intel.com"

  def test_legacy_port_and_tls_names_are_ignored(self, monkeypatch):
    monkeypatch.setenv("QDRANT_PORT", "6334")
    monkeypatch.setenv("QDRANT_USE_TLS", "false")
    assert reid_env.get_reid_port() == 55555
    assert reid_env.get_reid_use_tls() is True

  def test_legacy_confidence_and_cert_names_are_ignored(self, monkeypatch):
    monkeypatch.setenv("VDMS_CONFIDENCE_THRESHOLD", "0.75")
    monkeypatch.setenv("VDMS_CA_CERT", "/tmp/legacy-ca.pem")
    monkeypatch.setenv("QDRANT_API_KEY", "legacy-key")
    assert reid_env.get_reid_confidence_threshold() == 0.8
    assert reid_env.get_reid_ca_cert() == "/run/secrets/certs/scenescape-ca.pem"
    assert reid_env.get_reid_api_key() is None
