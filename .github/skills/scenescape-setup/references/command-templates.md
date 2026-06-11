<!--
SPDX-FileCopyrightText: (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Runtime Command Templates

Use these reusable commands in setup verification steps.

## Determine SceneScape Network Name

```bash
NET_NAME=$(docker network ls --format '{{.Name}}' | grep '_scenescape$' | head -1)
```

## RTSP Gate Check

```bash
docker run --rm --network "$NET_NAME" \
  linuxserver/ffmpeg:version-8.1-cli \
  -nostdin -v error -rtsp_transport tcp \
  -i '<rtsp_url>' \
  -t 5 -f null -
echo "EXIT:$?"
```

## MQTT Subscribe (Plaintext 1883)

```bash
docker run --rm --network <project>_scenescape eclipse-mosquitto:2 \
  mosquitto_sub -h broker.scenescape.intel.com -p 1883 \
  -t '<topic>' -C 1 -W 120
```

## MQTT Publish (Plaintext 1883)

```bash
docker run --rm --network <project>_scenescape eclipse-mosquitto:2 \
  mosquitto_pub -h broker.scenescape.intel.com -p 1883 \
  -t '<topic>' -m '<payload>'
```

## MQTT Subscribe (TLS 1883)

```bash
docker run --rm --network <project>_scenescape eclipse-mosquitto:2 \
  mosquitto_sub -h broker.scenescape.intel.com -p 1883 \
  --cafile <deploy_dir>/secrets/certs/scenescape-ca.pem \
  --insecure \
  -t '<topic>' -C 1 -W 120
```

## MQTT Publish (TLS 1883)

```bash
docker run --rm --network <project>_scenescape eclipse-mosquitto:2 \
  mosquitto_pub -h broker.scenescape.intel.com -p 1883 \
  --cafile <deploy_dir>/secrets/certs/scenescape-ca.pem \
  --insecure \
  -t '<topic>' -m '<payload>'
```
