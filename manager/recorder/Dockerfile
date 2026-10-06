# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY recorder.py .

# Local disk volume for .rrd files (shared with the manager for the
# recordings REST endpoint).
VOLUME /data/recordings

ENV RECORDER_MQTT_BROKER=localhost \
    RECORDER_MQTT_PORT=1883 \
    RECORDER_STORAGE_DIR=/data/recordings \
    RECORDER_RETENTION_DAYS=1

CMD ["python", "recorder.py"]
