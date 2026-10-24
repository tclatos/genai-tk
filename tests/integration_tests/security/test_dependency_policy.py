"""Guard the reviewed lock against vulnerable dependency paths returning."""

import tomllib
from pathlib import Path

import pytest
from packaging.version import Version


@pytest.fixture
def locked_versions() -> dict[str, Version]:
    lock_path = Path(__file__).resolve().parents[3] / "uv.lock"
    lock = tomllib.loads(lock_path.read_text())
    return {package["name"]: Version(package["version"]) for package in lock["package"]}


@pytest.mark.integration
@pytest.mark.parametrize("package", ["chromadb", "langchain-chroma", "diskcache"])
def test_retired_dependencies_are_absent(locked_versions: dict[str, Version], package: str) -> None:
    assert package not in locked_versions


@pytest.mark.integration
@pytest.mark.parametrize(
    ("package", "minimum"),
    [
        ("pillow", "13.0.0"),
        ("fastembed", "0.9.1"),
        ("streamlit", "1.66.1"),
        ("docling", "2.142.1"),
        ("docling-slim", "2.142.1"),
        ("docling-ibm-models", "4.0.4"),
    ],
)
def test_pillow_compatible_dependency_floors(locked_versions: dict[str, Version], package: str, minimum: str) -> None:
    assert locked_versions[package] >= Version(minimum)
