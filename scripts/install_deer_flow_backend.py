#!/usr/bin/env python3
"""Install Deer-flow backend dependencies."""

import subprocess
import sys
import tomllib
from pathlib import Path

# Read from environment or use default
deer_flow_dir = Path("ext/deer-flow")
backend = deer_flow_dir / "backend"

if not backend.exists():
    print(f"Error: Deer-flow backend not found at {backend}")
    sys.exit(1)

pyproject = backend / "pyproject.toml"
if not pyproject.exists():
    print(f"Error: pyproject.toml not found at {pyproject}")
    sys.exit(1)

with open(pyproject, "rb") as f:
    data = tomllib.load(f)

deps = data.get("project", {}).get("dependencies", [])
harness = backend / "packages" / "harness"

cmd = ["uv", "pip", "install"]
if harness.exists():
    cmd += ["-e", str(harness)]
cmd += deps

result = subprocess.run(cmd)
sys.exit(result.returncode)
