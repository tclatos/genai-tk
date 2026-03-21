"""Patch the agent-sandbox wheel to remove its spurious volcengine-python-sdk dependency.

The published agent-sandbox package declares volcengine-python-sdk as a hard
dependency in its wheel metadata, but never imports it in any of its code.
This script downloads the wheel and removes that entry from METADATA and RECORD.

Usage:
    uv run scripts/patch_agent_sandbox_wheel.py [--version 0.0.26]

Output:
    vendor/agent_sandbox-<version>-py2.py3-none-any.whl
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

DIST_NAME = "agent-sandbox"
DEFAULT_VERSION = "0.0.26"
VENDOR_DIR = Path(__file__).parent.parent / "vendor"
SPURIOUS_DEP = "volcengine-python-sdk"


def patch_wheel(src: Path, dst: Path) -> None:
    """Copy *src* wheel to *dst*, removing the spurious dependency from METADATA and RECORD."""
    pkg_base = src.stem.replace("-py2.py3-none-any", "").replace("-py3-none-any", "")
    # Handle both naming conventions
    with zipfile.ZipFile(src) as z:
        dist_info = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        dist_info_dir = dist_info.rsplit("/", 1)[0]

    metadata_path = f"{dist_info_dir}/METADATA"
    record_path = f"{dist_info_dir}/RECORD"

    with zipfile.ZipFile(src) as z:
        raw_meta = z.read(metadata_path).decode("utf-8")

    patched_lines = [
        l for l in raw_meta.splitlines(keepends=True) if not l.startswith(f"Requires-Dist: {SPURIOUS_DEP}")
    ]
    patched_meta = "".join(patched_lines).encode("utf-8")

    digest = hashlib.sha256(patched_meta).digest()
    b64hash = "sha256=" + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    new_size = len(patched_meta)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for filename in zin.namelist():
            data = zin.read(filename)
            info = zin.getinfo(filename)
            if filename == metadata_path:
                data = patched_meta
            elif filename == record_path:
                lines = data.decode("utf-8").splitlines(keepends=True)
                new_lines = []
                for line in lines:
                    if line.startswith(metadata_path + ","):
                        line = f"{metadata_path},{b64hash},{new_size}\n"
                    new_lines.append(line)
                data = "".join(new_lines).encode("utf-8")
            zout.writestr(info, data)

    # Verify
    with zipfile.ZipFile(dst) as z:
        content = z.read(metadata_path).decode()
        remaining = [l for l in content.splitlines() if l.startswith("Requires-Dist")]
    assert not any(SPURIOUS_DEP in r for r in remaining), "Patch failed — dep still present!"
    print(f"Patched wheel written to: {dst}")
    print(f"Remaining deps: {remaining}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_VERSION, help="agent-sandbox version to patch")
    args = parser.parse_args()

    version = args.version
    wheel_name = f"agent_sandbox-{version}-py2.py3-none-any.whl"
    dst = VENDOR_DIR / wheel_name

    if dst.exists():
        print(f"Patched wheel already exists: {dst}")
        sys.exit(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        print(f"Downloading {DIST_NAME}=={version} ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "download", f"{DIST_NAME}=={version}", "--no-deps", "-d", tmpdir],
            check=True,
        )
        src = next(tmp.glob("agent_sandbox-*.whl"))
        patch_wheel(src, dst)


if __name__ == "__main__":
    main()
