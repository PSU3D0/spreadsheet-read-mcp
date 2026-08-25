#!/usr/bin/env python3
"""Validate MCP registry metadata that is cheap to check in CI.

This intentionally avoids third-party Python dependencies. The official registry
schema is still the source of truth; this script catches local drift before we
publish or submit registry PRs.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_JSON = REPO_ROOT / "server.json"
DOCKERFILES = [REPO_ROOT / "Dockerfile", REPO_ROOT / "Dockerfile.full"]
EXPECTED_SERVER_NAME = "io.github.PSU3D0/agent-spreadsheet"
EXPECTED_REPO = "https://github.com/PSU3D0/agent-spreadsheet"
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


def fail(message: str) -> None:
    print(f"registry metadata validation failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    try:
        data = json.loads(SERVER_JSON.read_text())
    except Exception as exc:  # noqa: BLE001 - print actionable CI error
        fail(f"could not parse {SERVER_JSON}: {exc}")

    if data.get("name") != EXPECTED_SERVER_NAME:
        fail(f"server.json name must be {EXPECTED_SERVER_NAME!r}")

    if data.get("repository", {}).get("url") != EXPECTED_REPO:
        fail(f"server.json repository.url must be {EXPECTED_REPO!r}")

    version = data.get("version")
    if not isinstance(version, str) or not VERSION_RE.match(version):
        fail("server.json version must be a concrete semver-like version")

    description = data.get("description")
    if not isinstance(description, str) or not (1 <= len(description) <= 100):
        fail("server.json description must be 1..100 characters for official registry schema")

    packages = data.get("packages")
    if not isinstance(packages, list) or not packages:
        fail("server.json must include at least one package")

    for package in packages:
        if package.get("registryType") != "oci":
            fail("agent-spreadsheet registry package must use registryType=oci")
        identifier = package.get("identifier", "")
        if not identifier.startswith("ghcr.io/psu3d0/agent-spreadsheet:"):
            fail("OCI identifier must point at ghcr.io/psu3d0/agent-spreadsheet:<tag>")
        if package.get("transport", {}).get("type") != "stdio":
            fail("registry package transport must be stdio")

    label = f'LABEL io.modelcontextprotocol.server.name="{EXPECTED_SERVER_NAME}"'
    for dockerfile in DOCKERFILES:
        text = dockerfile.read_text()
        if label not in text:
            fail(f"{dockerfile.relative_to(REPO_ROOT)} missing MCP registry label {label!r}")

    print("registry metadata validation passed")


if __name__ == "__main__":
    main()
