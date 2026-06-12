# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Generate deployment files (skill steps 2–6)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def skill_dir_from_arg(value: Path) -> Path:
  path = value.resolve()
  if not (path / "references" / "docker-compose-template.md").is_file():
    raise ValueError(f"Not a scenescape-setup skill directory: {path}")
  return path


def download_dlstreamer_pipeline_server(deploy_dir: Path) -> None:
  target = deploy_dir / "dlstreamer-pipeline-server" / "queuing-config.json"
  if target.is_file():
    return

  tmp = deploy_dir / "_scenescape-tmp"
  if tmp.exists():
    shutil.rmtree(tmp)

  subprocess.run(
    [
      "git", "clone", "--filter=blob:none", "--sparse",
      "https://github.com/open-edge-platform/scenescape.git",
      str(tmp),
    ],
    check=True,
  )
  subprocess.run(
    ["git", "sparse-checkout", "set", "dlstreamer-pipeline-server"],
    cwd=tmp,
    check=True,
  )
  shutil.copytree(tmp / "dlstreamer-pipeline-server", deploy_dir / "dlstreamer-pipeline-server")
  shutil.rmtree(tmp)


def copy_skill_assets(skill_dir: Path, deploy_dir: Path) -> None:
  scripts_dst = deploy_dir / "scripts"
  scripts_dst.mkdir(parents=True, exist_ok=True)

  for pattern in ("*.py", "*.sh"):
    for src in (skill_dir / "scripts").glob(pattern):
      shutil.copy2(src, scripts_dst / src.name)

  for name in ("generate_secrets.sh", "openssl.cnf"):
    shutil.copy2(skill_dir / "references" / name, deploy_dir / name)

  subprocess.run(["chmod", "+x", *map(str, scripts_dst.glob("*.sh"))], check=False)


def generate_docker_compose(skill_dir: Path, deploy_dir: Path) -> None:
  template = skill_dir / "references" / "docker-compose-template.md"
  secrets_dir = deploy_dir / "secrets"
  awk_cmd = (
    "awk '/^```yaml$/ {flag=1; next} /^```$/ && flag {exit} flag {print}' "
    f"\"{template}\" | sed \"s|\\${{SECRETSDIR}}|{secrets_dir}|g\""
  )
  compose = subprocess.run(
    ["bash", "-c", awk_cmd],
    check=True,
    capture_output=True,
    text=True,
  )
  (deploy_dir / "docker-compose.yml").write_text(compose.stdout, encoding="utf-8")


def copy_static_configs(skill_dir: Path, deploy_dir: Path) -> None:
  for name in ("tracker-config.json", "reid-config.json"):
    shutil.copy2(skill_dir / "references" / name, deploy_dir / name)


def generate_secrets_and_env(
  deploy_dir: Path,
  skill_dir: Path,
  no_proxy_hosts: list[str],
) -> None:
  subprocess.run(["bash", "generate_secrets.sh"], cwd=deploy_dir, check=True)

  cmd = [
    sys.executable,
    str(skill_dir / "scripts" / "write_deployment_env.py"),
    "--deploy-dir", str(deploy_dir),
  ]
  for host in no_proxy_hosts:
    cmd.extend(["--append-no-proxy", host])

  subprocess.run(cmd, check=True)


def main() -> None:
  parser = argparse.ArgumentParser(description="Bootstrap SceneScape deployment files (steps 2–6)")
  parser.add_argument("--deploy-dir", required=True, type=Path)
  parser.add_argument("--skill-dir", required=True, type=Path)
  parser.add_argument("--camera-ids", required=True, nargs="+", metavar="CAMERA_ID")
  parser.add_argument("--streams", required=True, nargs="+", metavar="RTSP_URL")
  args = parser.parse_args()

  deploy_dir = args.deploy_dir.resolve()
  skill_dir = skill_dir_from_arg(args.skill_dir)
  deploy_dir.mkdir(parents=True, exist_ok=True)

  download_dlstreamer_pipeline_server(deploy_dir)
  copy_skill_assets(skill_dir, deploy_dir)
  generate_docker_compose(skill_dir, deploy_dir)

  adapt_script = Path(__file__).resolve().parent / "adapt_pipeline_config.py"
  adapt = subprocess.run(
    [
      sys.executable, str(adapt_script),
      "--deploy-dir", str(deploy_dir),
      "--camera-ids", *args.camera_ids,
      "--streams", *args.streams,
    ],
    check=True,
    capture_output=True,
    text=True,
  )
  no_proxy_hosts = []
  for line in adapt.stdout.splitlines():
    if line.startswith("no_proxy_hosts="):
      no_proxy_hosts = [h for h in line.split("=", 1)[1].split(",") if h]
  copy_static_configs(skill_dir, deploy_dir)
  generate_secrets_and_env(deploy_dir, skill_dir, no_proxy_hosts)

  print(f"Bootstrap complete: {deploy_dir}")


if __name__ == "__main__":
  main()
