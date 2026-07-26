<!--
SPDX-FileCopyrightText: (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Scenescape — AI agents

**Do not add project policy here.**

- **All tools:** [`.github/copilot-instructions.md`](.github/copilot-instructions.md)
- **Cursor:** [`.cursor/rules/scenescape.mdc`](.cursor/rules/scenescape.mdc) (loaded automatically; skills, service `Agents.md`, and IDE workflow)

## Cursor Cloud specific instructions

Non-obvious startup/run caveats for the cloud VM. Standard build/test/run commands live in
[`docs/user-guide/get-started.md`](docs/user-guide/get-started.md), [`tests/README.md`](tests/README.md),
and the root `Makefile`; only the gotchas are captured here.

### Docker daemon (must start each session)

- This is a Docker Compose / `make` project; **everything runs in Docker**. The VM has no systemd,
  so `systemctl start docker` fails. Start the daemon manually and leave it running, e.g. in a tmux
  session: `sudo dockerd > /tmp/dockerd.log 2>&1 &`.
- The daemon is preconfigured in `/etc/docker/daemon.json` to use `fuse-overlayfs` with the
  containerd snapshotter disabled (required because this is Docker-in-Docker on this VM). Do not
  switch it back to `overlay2`/containerd-snapshotter or image builds will fail.
- Run `docker` without sudo via `sudo chmod 666 /var/run/docker.sock` after starting the daemon
  (docker-group membership needs a fresh login shell, which the persistent shell here doesn't get).

### Building and running

- Build core images (one-time per image change, ~10-15 min): `SUPASS=<pw> make build-core`.
  Images persist in the VM snapshot, so this is usually already done — only rebuild after code
  changes (`make rebuild-<service>` / `make build-core`).
- Run the demo stack: `SUPASS=<pw> make demo` (Compose profile `controller`, ~11 containers).
  Web UI: `https://localhost` (self-signed cert → accept the warning). Login `admin` / `$SUPASS`.
  Stop with `make demo-close`. Live object tracking on the scene map proves the full pipeline
  (video → DL Streamer → MQTT → controller → UI) is working.

### Tests and the host pytest venv

- Unit tests run on the host in `tests/.venv` (SQLite/Django, no containers): `make run_unit_tests`.
  Other suites (functional/ui) are Docker-based; see `tests/README.md`.
- `make setup-pytest` builds the `fast_geometry` and `robot_vision` C++ extensions into
  `tests/.venv`. **Gotcha:** `fast_geometry`'s Makefile names the compiled `.so` from
  `python3-config --extension-suffix`; if `python3-config` is missing the lib is installed as a
  bare `fast_geometry` file and `sscape_tests` collection then dies with
  `RuntimeError: populate() isn't reentrant` (a swallowed `ModuleNotFoundError:
  fast_geometry.fast_geometry`). The fix is the `python3-dev` package (provides `python3-config`);
  it is installed in the snapshot, so `setup-pytest` builds the extension correctly.

### Lint

- Required CI linters that must pass: `make prettier-check` and `make indent-check`.
  `prettier`/`eslint` come from `node_modules` (`npm install` using `.github/resources/package.json`).
- Optional linters (non-gating in CI, expected to report pre-existing findings): `make lint-python`
  (`pylint`/`flake8`), `make lint-cpp` (`cpplint`), `make lint-shell` (`shellcheck`),
  `make lint-dockerfiles` (`hadolint`). Python/C++ linters are installed with `pipx`, so ensure
  `~/.local/bin` is on `PATH` before running them.
