---
name: optional-features
description: How to add, update, or remove optional feature extras in genai-tk — covering pyproject.toml, features.py registry, import guards, and tests.
tags: [packaging, optional-deps, uv, imports, testing]
version: "1.0"
---

# Optional Features in genai-tk

genai-tk ships a lean core and exposes heavyweight components as **opt-in extras**.
This skill explains the full lifecycle: adding a feature, gating its imports,
writing guarded tests, and keeping the registry in sync.

## Key Files

| File | Role |
|------|------|
| `pyproject.toml` → `[project.optional-dependencies]` | User-facing PyPI extras (`uv sync --extra <name>`) |
| `pyproject.toml` → `[dependency-groups]` | Developer-only groups (`dev`, `evals`) — NOT on PyPI |
| `genai_tk/config_mgmt/features.py` | `FEATURES` registry + `require_feature()` + `is_available()` |
| `tests/conftest.py` | `pytest_collection_modifyitems` — skips tests when feature absent |

---

## 1. Adding a New Optional Feature

### 1a. Add the extra to `pyproject.toml`

```toml
[project.optional-dependencies]
my-feature = [
    "some-package>=1.0",
    "another-package>=2.0",
]
# Keep `all` up to date:
all = ["genai-tk[harnessing,browser,nlp,postgres,streamlit,baml,chromadb,my-feature]"]
```

Rule: `[project.optional-dependencies]` is for **end users** (published to PyPI).
`[dependency-groups]` is for **contributors only** (local dev tools like `ruff`, `pytest`).

### 1b. Register in `features.py`

Open `genai_tk/config_mgmt/features.py` and add an entry to `FEATURES`:

```python
"my-feature": FeatureInfo(
    description="Short human description shown in error messages",
    packages=["some-package", "another-package"],      # PyPI names
    check_modules=["some_package", "another_package"], # importable module names
    install_cmd='uv sync --extra my-feature  # or: uv add "genai-tk[my-feature]"',
),
```

**`packages`** = PyPI package names (used in error messages).
**`check_modules`** = Python module names you'd pass to `importlib.util.find_spec()`.

### 1c. Verify

```bash
uv run python -c "from genai_tk.config_mgmt.features import missing_features; print(missing_features())"
```

The new feature should appear in the output if not yet installed.

---

## 2. Gating Imports at Call Sites

### At module top-level (entry points, CLI shells, webapp)

If the entire module only makes sense when a feature is installed, guard at the top:

```python
from genai_tk.config_mgmt.features import require_feature

require_feature("my-feature", context="cli agents my-command")

from some_package import SomeClass   # safe: require_feature raises before this line
```

### Inside a class method or function (lazy — preferred)

```python
def create(self, ...):
    from genai_tk.config_mgmt.features import require_feature  # noqa: PLC0415
    require_feature("my-feature", context="MyClass.create")
    from some_package import SomeClass  # noqa: PLC0415
    ...
```

### Error message produced

When the feature is missing the user sees:

```
ImportError: Optional feature 'my-feature' (required by: MyClass.create) is not installed.
  Description : Short human description shown in error messages
  Packages    : some-package, another-package
  Install with: uv sync --extra my-feature  # or: uv add "genai-tk[my-feature]"
```

---

## 3. Writing Feature-Gated Tests

### Per-test skip (recommended for individual tests)

Use the `@pytest.mark.requires_feature` marker — registered automatically by `tests/conftest.py`:

```python
import pytest

@pytest.mark.requires_feature("my-feature")
def test_my_feature_does_something():
    from some_package import SomeClass
    result = SomeClass().run()
    assert result is not None
```

### Whole-module skip (required when the module-level import chain depends on a feature)

When a test file imports a production module that calls `require_feature()` at its
own top level (e.g. `aio_backend.py`), the `@pytest.mark.requires_feature` marker
is too late — the import fails before any marker can be applied.
Use `pytest.skip(allow_module_level=True)` instead:

```python
import pytest
from genai_tk.config_mgmt.features import is_available

if not is_available("harnessing"):
    pytest.skip(
        "Optional feature 'harnessing' not installed — run: uv sync --extra harnessing",
        allow_module_level=True,
    )

# Only reached when harnessing is installed:
from deepagents.backends.protocol import SandboxBackendProtocol  # noqa: E402
from genai_tk.agents.sandbox.aio_backend import AioSandboxBackend  # noqa: E402
```

The test is **skipped** (not failed) when the feature is absent, with a message:

```
SKIPPED — Optional feature 'harnessing' not installed — run: uv sync --extra harnessing
```

---

## 4. Removing or Renaming a Feature

1. Remove the key from `FEATURES` in `features.py`.
2. Remove or rename the extra in `pyproject.toml → [project.optional-dependencies]`.
3. Update the `all` extra accordingly.
4. Search for `require_feature("old-name"` and `is_available("old-name"` with grep and update.
5. Update test markers: `@pytest.mark.requires_feature("old-name")`.

---

## 5. Checking Feature State at Runtime

```python
from genai_tk.config_mgmt.features import available_features, missing_features, is_available

print(available_features())   # ['baml', 'browser', 'chromadb', ...]
print(missing_features())     # ['harnessing', 'nlp', ...]
print(is_available("baml"))   # True / False
```

---

## 6. Installing Features

```bash
# Install one feature
uv sync --extra browser

# Install multiple features
uv sync --extra harnessing --extra browser

# Install everything
uv sync --extra all

# During project init
cli init --extra harnessing --extra browser

# In a downstream project using genai-tk
uv add "genai-tk[harnessing,browser]"
```

---

## 7. Current Feature Registry

| Feature | Packages | Notes |
|---------|----------|-------|
| `harnessing` | smolagents, deepagents, deepagents-cli, agent-sandbox, opensandbox, deerflow-harness | Heavy — includes Docker sandbox |
| `browser` | playwright | Run `uv run playwright install chromium` after install |
| `nlp` | spacy, en-core-web-sm, en-core-web-lg | ~500 MB including models |
| `postgres` | langchain-postgres, psycopg, psycopg2-binary | Requires PostgreSQL server |
| `streamlit` | streamlit | Web UI — not needed for CLI-only use |
| `baml` | baml-cli, baml-lib | Run `uv run baml-cli init --dest baml_src` after install |
| `chromadb` | chromadb, langchain-chroma | Local vector DB |

---

## 8. Sync Check (manual)

There is no automated sync script yet. To verify consistency:

```bash
# List extras defined in pyproject.toml
python -c "
import tomllib
p = tomllib.loads(open('pyproject.toml').read())
print(list(p['project']['optional-dependencies'].keys()))
"

# List features registered in features.py
uv run python -c "
from genai_tk.config_mgmt.features import FEATURES
print(list(FEATURES.keys()))
"
```

Both lists should match (minus the `all` aggregate extra).
