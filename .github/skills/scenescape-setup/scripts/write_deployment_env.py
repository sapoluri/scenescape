# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Write deployment .env from generated SceneScape secrets.

Usage:
  python write_deployment_env.py --deploy-dir <deploy_dir>
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


def read_database_password(secrets_py: Path) -> str:
  text = secrets_py.read_text(encoding="utf-8")
  match = re.search(r"DATABASE_PASSWORD='([^']+)'", text)
  if not match:
    raise ValueError(f"DATABASE_PASSWORD not found in {secrets_py}")
  return match.group(1)


def main() -> None:
  parser = argparse.ArgumentParser(description="Write deployment .env from generated secrets")
  parser.add_argument("--deploy-dir", required=True, type=Path)
  args = parser.parse_args()

  deploy_dir = args.deploy_dir
  secrets_dir = deploy_dir / "secrets"

  database_password = read_database_password(secrets_dir / "django" / "secrets.py")
  supass = (secrets_dir / "supass").read_text(encoding="utf-8").strip()

  env_text = "\n".join(
    [
      f"SECRETSDIR={secrets_dir}",
      f"DATABASE_PASSWORD={database_password}",
      f"SUPASS={supass}",
      f"http_proxy={os.getenv('http_proxy', '')}",
      f"https_proxy={os.getenv('https_proxy', '')}",
      f"no_proxy={os.getenv('no_proxy', '')}",
      "",
    ]
  )
  (deploy_dir / ".env").write_text(env_text, encoding="utf-8")


if __name__ == "__main__":
  main()