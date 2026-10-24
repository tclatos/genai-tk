"""Verify Docker itself excludes credentials before COPY can reach them."""

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.docker]

SECRET_PATHS = (
    ".env",
    "nested/.env.production",
    ".aws/credentials",
    ".ssh/id_ed25519",
    ".streamlit/secrets.toml",
    "nested/client.key",
    "nested/client.pem",
    "nested/service-credentials.json",
    ".npmrc",
    ".netrc",
)
PUBLIC_FILES = {
    "app.py": "print('public application')\n",
    "skills/runtime/example/SKILL.md": "# Public runtime skill\n",
    "skills/runtime/example/references/workflow.md": "# Public skill reference\n",
}


@pytest.fixture(scope="module")
def exported_context(tmp_path_factory: pytest.TempPathFactory) -> Path:
    workspace = tmp_path_factory.mktemp("docker-context-security")
    context = workspace / "context"
    context.mkdir()
    repository = Path(__file__).resolve().parents[3]
    shutil.copyfile(repository / ".dockerignore", context / ".dockerignore")
    for relative_path in SECRET_PATHS:
        secret = context / relative_path
        secret.parent.mkdir(parents=True, exist_ok=True)
        secret.write_text("synthetic-secret-sentinel\n", encoding="utf-8")
    for relative_path, content in PUBLIC_FILES.items():
        public_file = context / relative_path
        public_file.parent.mkdir(parents=True, exist_ok=True)
        public_file.write_text(content, encoding="utf-8")
    (context / "Dockerfile").write_text("FROM scratch\nCOPY . /context/\n", encoding="utf-8")
    output = workspace / "output"
    subprocess.run(
        [
            "docker",
            "buildx",
            "build",
            "--network=none",
            "--progress=plain",
            "--output",
            f"type=local,dest={output}",
            str(context),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    return output / "context"


@pytest.mark.parametrize("relative_path", SECRET_PATHS)
def test_secrets_never_enter_build_context(exported_context: Path, relative_path: str) -> None:
    assert not (exported_context / relative_path).exists()


@pytest.mark.parametrize("relative_path", PUBLIC_FILES)
def test_application_assets_still_enter_build_context(exported_context: Path, relative_path: str) -> None:
    assert (exported_context / relative_path).read_text(encoding="utf-8") == PUBLIC_FILES[relative_path]
