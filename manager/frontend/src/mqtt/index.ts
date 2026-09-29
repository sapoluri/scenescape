// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * React MQTT helpers — client ownership remains in sscape.js (`ssMqttClient`).
 * Scene-detail React code should call `ensureMqttScene` via `lib/legacyBridge`.
 */
export { ensureMqttScene } from "../lib/legacyBridge";
