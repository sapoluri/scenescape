<!--
SPDX-FileCopyrightText: (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Functional and Integration Tests

**Location**: `tests/functional/` (integration-style flows also use this tree or `tests/system/`)

**Infrastructure**: `scenescape_env` in `tests/conftest.py` reads module-level `SCENESCAPE_SPEC`, starts the `ServiceProfile` stack, injects `params`, restores DB on teardown (unless `@pytest.mark.preserve_db`).

**Profiles**: `tests/utils/profiles.py`

**Real examples**: `tests/functional/test_roi_mqtt.py`, other modules under `tests/functional/`

## Required module header

```python
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.utils.spec import FuncTestSpec, AUTH_CONTROLLER
from tests.utils.profiles import FULL_STACK

SCENESCAPE_SPEC = FuncTestSpec(
  profile=FULL_STACK,
  auth=AUTH_CONTROLLER,
)

# Optional: Zephyr IDs when using --env-profiles
SCENESCAPE_ENV_MATRIX = {
  "full_stack": "NEX-T10404",
}

TEST_NAME = "NEX-T10404"

@pytest.mark.basic_acceptance
def test_roi_create(scenescape_env, demo_scene, request, record_xml_attribute):
  test_name = getattr(request.node, '_scenescape_test_name', TEST_NAME)
  # ... exercise MQTT/REST against live stack ...
  assert True
```

## FuncTestSpec fields

Defined in `tests/utils/spec.py`:

| Field | Purpose |
|-------|---------|
| `profile` | `ServiceProfile` from `tests/utils/profiles.py` |
| `auth` | `AUTH_CONTROLLER` or `AUTH_BROWSER` |
| `require_password` | Default `True` |
| `test_name` | Optional NEX ID for XML reporting |
| `extra_args` | Extra `--key value` pairs for params |
| `exampledb` | Override baseline DB (e.g. `calibrationdb.tar.bz2`) |

## Multi-profile runs

```bash
pytest tests/functional/test_roi_mqtt.py --env-profiles=full_stack,full_stack_with_mapping
```

## Integration-style tests

Same `SCENESCAPE_SPEC` + live services pattern. Use when asserting cross-service flows (MQTT → controller → REST/DB). Prefer existing helpers (`SceneObjectMqtt`, `RESTClient`) over duplicating boilerplate.
