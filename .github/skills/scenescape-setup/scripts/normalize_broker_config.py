# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Normalize SceneScape mosquitto config for plaintext MQTT on listener 1883.

This keeps listener 1884 TLS websocket settings intact while removing TLS settings
from listener 1883, which DLStreamer and controller clients use in deployment setup.
"""

from pathlib import Path


TLS_LINES = {
  "keyfile /mosquitto/secrets/certs/scenescape-broker.key",
  "certfile /mosquitto/secrets/certs/scenescape-broker.crt",
  "tls_version tlsv1.3",
}


def normalize_config(config_path: Path) -> None:
  """Remove TLS directives under listener 1883 in mosquitto config."""
  lines = config_path.read_text(encoding="utf-8").splitlines()
  listener = None
  output = []

  for line in lines:
    stripped = line.strip()
    if stripped.startswith("listener "):
      parts = stripped.split()
      listener = parts[1] if len(parts) > 1 else None

    if listener == "1883" and stripped in TLS_LINES:
      continue

    output.append(line)

  config_path.write_text("\n".join(output) + "\n", encoding="utf-8")


def main() -> None:
  config_path = Path("dlstreamer-pipeline-server/mosquitto/mosquitto-secure.conf")
  normalize_config(config_path)


if __name__ == "__main__":
  main()