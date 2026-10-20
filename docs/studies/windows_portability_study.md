# Windows Portability & Feasibility Study (Native Non-Linux Execution)

**Status**: Proposal / Exploratory Study  
**Target Environment**: Windows 11 / Windows Server (Native x64, PowerShell / cmd, non-WSL)  
**Authors**: GenAI Toolkit Architecture Team  
**Date**: October 2026  

---

## 1. Executive Summary

This study evaluates running **`genai-tk`** natively under **Windows** (without requiring WSL or Linux VMs). It analyzes compatibility across the codebase, pinpoints exact breaking points, recommends strategies to limit development friction, and defines a phased delivery plan.

### Compatibility Heatmap

| Subsystem / Layer | Compatibility | Critical Gaps / Blockers on Native Windows |
| :--- | :---: | :--- |
| **Core Framework** (LLM/Embeddings, LCEL, Prompts) | **100%** | Standard Python 3.12+ (Pydantic v2, LangChain, LiteLLM, OmegaConf). Works out of the box. |
| **CLI & Scaffolding** (`cli`, Typer, Rich, Textual) | **95%** | Minor path formatting in CLI prompts; Textual/Rich have full Windows VT100 support. |
| **Workflow Engine** (Prefect, `prefect-yaml`) | **85%** | Multiprocessing uses `spawn` instead of `fork`; requires strict top-level module guards. |
| **Document Pipeline** (Markdownize, LibreOffice) | **70%** | LibreOffice detection (`soffice.exe` search path); Docling C++ wheel compatibility. |
| **Build & Dev Tooling** (`justfile`, `tk.just`, `uv`) | **50%** | POSIX shell recipes (`curl \| sh`, `command -v`, `rm -rf`, `find`) break on PowerShell/cmd. |
| **Agent Sandboxing & CodeAct** (Docker/OpenSandbox) | **25%** | Docker volume mount syntax (`C:\` vs `/c/`), named pipe vs unix socket, POSIX permissions. |

---

## 2. In-Depth Gap Analysis: What Will Not Work

### 2.1. Build & Task Runner (`tk.just` / `justfile`)

The repository's shared recipes in `tk.just` rely on Linux shell utilities and POSIX syntax:

1. **POSIX Tooling**:
   - `check-uv`: Uses `command -v uv >/dev/null 2>&1` and `curl -LsSf https://astral.sh/uv/install.sh | sh`. Neither `command -v` nor piping to `sh` exists on standard Windows shells.
   - `clean`: Relies on `find . \( -name "*.py[co]" ... \) -exec rm -rf {} + 2>/dev/null || true`. `find` and `rm -rf` are unavailable without MSYS2/Git Bash.
2. **Environment Variable Injection**:
   - Recipes like `@webapp: PYTHONPATH="{{ dev_pythonpath }}" uv run streamlit run ...` fail because Windows cmd/PowerShell do not support inline POSIX variable prefixes.

### 2.2. Filesystem Paths & Separator Conventions

1. **Hardcoded `/tmp` References**:
   - Several scratchpad utilities, test runners, and caching routines default to `/tmp` instead of Python's portable `tempfile.gettempdir()`. On Windows, `/tmp` resolves to non-existent root locations like `C:\tmp`.
2. **Path Separators in URLs & Mounts**:
   - Windows uses backslashes (`\`). While Python's `pathlib.Path` handles these internally, passing raw string paths to Docker volume mounts or vector store URIs causes parsing errors unless normalized to `.as_posix()`.
3. **Case Sensitivity & Gitwildmatch**:
   - NTFS is case-preserving but case-insensitive. `pathspec` globbing and hash manifests must account for case normalization to prevent cache misses or duplicate indexing.

### 2.3. Agent Sandboxing & Code Execution (`genai_tk/agents/sandbox/`)

1. **Docker Desktop Volume Mounts**:
   - Mounting directories into containers (e.g. `./skills:/skills:ro`, `./workspace:/workspace:rw`) fails if absolute host paths containing drive letters (such as `C:\Users\...`) are passed without converting them to Docker Desktop mount syntax (`/c/Users/...`).
2. **Resource Limits & Kernel Flags**:
   - The Linux sandbox isolation in `genai-tk` relies on Linux kernel primitives (e.g., `ulimit core=0` in [Design 06](../design/06_sandbox_and_codeact.md)). Passing these parameters directly to native Windows processes or non-Linux containers fails.
3. **IPC & Sockets**:
   - OpenSandbox server bindings using Unix domain sockets (`/var/run/docker.sock`) must fall back to Windows named pipes (`npipe:////./pipe/docker_engine`).

### 2.4. Document Processing & External Binaries

1. **LibreOffice Detection**:
   - In `office2pdf_flow.py`, `is_libreoffice_available()` checks `shutil.which("libreoffice")` or `shutil.which("soffice")`.
   - On Windows, LibreOffice is rarely in system `PATH`; it is installed at `C:\Program Files\LibreOffice\program\soffice.exe`.
2. **Subprocess Invocation**:
   - Subprocesses spawning LibreOffice without `shell=False` or without specifying `.exe` may fail to resolve the binary on Windows.

### 2.5. Python Process & Async Runtime Semantics

1. **Multiprocessing `spawn` vs `fork`**:
   - Linux uses `fork` by default; Windows uses `spawn`. With `spawn`, child processes re-import the main module from scratch.
   - Any script, Prefect task, or CLI tool lacking an `if __name__ == '__main__':` block will trigger recursive process creation and crash with `RuntimeError`.
2. **Asyncio Proactor Loop**:
   - Windows defaults to `asyncio.ProactorEventLoop`. Some third-party subprocess transports or signal handlers (`SIGTERM`, `SIGINT`) behave differently than POSIX signals.

---

## 3. Recommendations to Limit Portability Effort

To achieve Windows support without creating a divergent codebase or burdensome maintenance overhead, the following principles should be adopted:

### 3.1. Abstract Task Execution into Python CLI Commands
Instead of maintaining dual Bash/PowerShell recipes in `justfile`, move complex shell logic into `cli` commands:
- Replace `check-uv` and `clean` with pure Python logic (e.g. `uv run cli dev clean`).
- In `justfile`, set the shell conditionally or use `uv run python -m ...` directly:
  ```just
  # Cross-platform clean recipe via Python
  clean:
      uv run python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
  ```

### 3.2. Enforce Path Portability via `pathlib.Path` & `.as_posix()`
- Mandate `tempfile.gettempdir()` for all temporary directories.
- Always apply `.resolve().as_posix()` when building Docker volume mappings or URI strings.
- Never concatenate paths with string addition (`+ "/" +`); use the `/` operator on `Path` instances.

### 3.3. Standardize Binary Resolution
Encapsulate external tool discovery in dedicated helper functions:
```python
def find_libreoffice_binary() -> Path | None:
    if which := shutil.which("soffice") or shutil.which("libreoffice"):
        return Path(which)
    if sys.platform == "win32":
        default_win_path = Path("C:/Program Files/LibreOffice/program/soffice.exe")
        if default_win_path.exists():
            return default_win_path
    return None
```

### 3.4. Rely on Docker Desktop (WSL2 Backend) for Container Sandboxing
Do not attempt to run agents inside Windows Server containers. Instead, support **Docker Desktop on Windows with the WSL2 backend**:
- The agent harness runs in native Windows Python.
- Containerized tools and sandboxes run in standard Linux container images hosted by Docker Desktop.
- Ensure volume bind-mount paths are translated cleanly across the host/container boundary.

---

## 4. Phased Portability Roadmap

```text
Phase 1: CI & Environment Scaffolding
  ├── Add Windows GitHub Actions runner (windows-latest)
  ├── Portable uv installation docs (PowerShell)
  └── Clean justfile recipes for cross-platform execution

Phase 2: Path & Runtime Normalization
  ├── Eliminate hardcoded /tmp and POSIX path separators
  ├── Normalize Docker volume mount paths via .as_posix()
  └── Audit 'if __name__ == "__main__":' multiprocessing guards

Phase 3: Document Conversion & Binary Discovery
  ├── Enhance LibreOffice discovery for Windows standard paths
  └── Validate C++ wheels (docling, baml-py, zvec) on win_amd64

Phase 4: Agent Sandboxing & Browser Validation
  ├── Verify Docker Desktop named-pipe / daemon communication
  └── Validate Playwright browser installation on Windows
```

### Phase 1: CI & Environment Scaffolding
- **Windows Runner in CI**: Add a Windows test job (`windows-latest`) running unit tests (`pytest tests/unit_tests/`).
- **PowerShell Setup Documentation**: Document Windows setup using `irm https://astral.sh/uv/install.ps1 | iex`.
- **Refactor `tk.just`**: Ensure core recipes (`fmt`, `lint`, `test-unit`) run cleanly on Windows.

### Phase 2: Path & Runtime Normalization
- Replace all hardcoded `/tmp` occurrences with `tempfile.gettempdir()`.
- Audit file opening and path manipulation across `genai_tk/config_mgmt/paths.py` and `genai_tk/workflow/`.
- Ensure all CLI entry points are guarded against multiprocessing `spawn` loops.

### Phase 3: Converters & Optional Extras
- Update `office2pdf_flow.py` to auto-detect `soffice.exe` in `C:\Program Files\LibreOffice\`.
- Test package installations on `win_amd64` for all optional extras (`[nlp]`, `[baml]`, `[docling]`).
- Document prerequisites for C++ compilers (Visual Studio Build Tools) if source builds are needed.

### Phase 4: Sandboxing & Agent Validation
- Test `SandboxManager.is_docker_available()` against Docker Desktop on Windows.
- Verify file exchange between native Windows filesystem and Linux sandbox containers.
- Verify headless browser execution via `playwright install chromium` on Windows.

---

## 5. Decision & Recommendation

1. **Target Architecture**: Support native Windows Python running directly on host, paired with Docker Desktop for container sandboxing.
2. **Estimated Porting Effort**: **Low to Moderate** (~2–3 engineering days). The core framework is pure Python and Pydantic v2; changes are primarily isolated to build scripts, path resolution, and binary discovery.
3. **Alternative**: If near-zero engineering effort is required, running `genai-tk` inside **WSL2** remains the fastest option for Windows developers today. However, native support is viable and straightforward to implement following the phased roadmap above.
