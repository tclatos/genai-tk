# Deer-flow Setup Simplification - Change Summary

**Date**: February 17, 2026  
**Objective**: Simplify deer-flow integration by making `DEER_FLOW_PATH` environment variable mandatory

## Overview

The deer-flow integration has been simplified by removing fallback path logic and making the `DEER_FLOW_PATH` environment variable mandatory. This makes the dependency explicit, reduces code complexity, and aligns with best practices for external tool integration.

## Changes Made

### 1. Simplified Path Setup (`_path_setup.py`)

**File**: `genai_tk/extra/agents/deer_flow/_path_setup.py`

**Before** (~100 lines):
- Searched 4 locations: DEER_FLOW_PATH, ext/deer-flow, ../deer-flow, ~/ext_prj/deer-flow
- Complex fallback logic
- Hardcoded development paths

**After** (~60 lines):
- **Only uses DEER_FLOW_PATH environment variable**
- Clear error message when not set
- No fallback logic
- Simple and maintainable

**Key Function Changes**:
```python
# Before
def get_deer_flow_backend_path() -> Path:
    # Search 4 locations...
    candidates = [env_path, ext/deer-flow, ../deer-flow, ~/ext_prj/deer-flow]
    # Try each candidate...

# After  
def get_deer_flow_backend_path() -> Path:
    env_path = os.environ.get("DEER_FLOW_PATH")
    if not env_path:
        raise EnvironmentError("DEER_FLOW_PATH environment variable is not set...")
    return Path(env_path) / "backend"
```

### 2. Updated Dependency Configuration (`pyproject.toml`)

**File**: `pyproject.toml`

**Before**:
```toml
deer-flow = [
    "deer-flow @ file:///home/tcl/ext_prj/deer-flow/backend",
]
```

**After**:
```toml
# NOTE: Deer-flow must be installed manually after setting DEER_FLOW_PATH
#   export DEER_FLOW_PATH=/path/to/deer-flow
#   uv pip install -e $DEER_FLOW_PATH/backend
deer-flow = []
```

**Rationale**: Package managers (uv/pip) don't support environment variable expansion in `file://` URLs. Manual installation is clearer and more portable across different environments.

### 3. Enhanced Documentation (`docs/Deer_Flow_Integration.md`)

**Updated Sections**:

1. **Overview** - Added note about mandatory `DEER_FLOW_PATH`
2. **Installation** - Added "Quick Start" section with setup script
3. **Path Setup** - Simplified to only mention `DEER_FLOW_PATH`
4. **Troubleshooting** - Updated error messages and solutions

**Key Additions**:
- Setup script usage instructions
- Clear environment variable requirements
- Step-by-step manual setup guide
- Updated error messages with solutions

### 4. Created Setup Script (`scripts/setup_deerflow.sh`)

**New File**: `scripts/setup_deerflow.sh` (executable, ~150 lines)

**Features**:
- Interactive installation with 3 options:
  1. `~/ext_prj/deer-flow` (recommended)
  2. `./ext/deer-flow` (in current project)
  3. Custom path
- Clones deer-flow repository if needed
- Installs deer-flow package with dependencies
- Adds `DEER_FLOW_PATH` to shell profile (~/.bashrc, ~/.zshrc, etc.)
- Tests installation
- Provides clear success/failure feedback

**Usage**:
```bash
bash scripts/setup_deerflow.sh
```

### 5. Updated README (`README.md`)

**New Section**: "Agent Frameworks" with deer-flow integration

**Content**:
- Quick setup instructions
- CLI usage examples
- Link to full documentation

## Benefits

### 1. **Simplicity (KISS Principle)**
- Reduced `_path_setup.py` by ~40% (100 lines → 60 lines)
- Removed complex fallback logic
- Single source of truth for deer-flow location

### 2. **Explicit Dependencies**
- No hidden fallback paths
- Clear error messages when not configured
- Users know exactly what's required

### 3. **Portability**
- Works on any machine with `DEER_FLOW_PATH` set
- No hardcoded paths (e.g., `/home/tcl/...`)
- Environment-agnostic

### 4. **Better Developer Experience**
- Automated setup script for quick start
- Clear documentation
- Helpful error messages with solutions

### 5. **Maintainability**
- Less code to maintain
- Clearer intent
- Easier to debug

## Migration Guide

### For Existing Users

If you previously had deer-flow working, you need to set the environment variable:

```bash
# Option 1: Use setup script (recommended)
bash scripts/setup_deerflow.sh

# Option 2: Manual setup
export DEER_FLOW_PATH=/home/tcl/ext_prj/deer-flow  # or your path
echo 'export DEER_FLOW_PATH=/home/tcl/ext_prj/deer-flow' >> ~/.bashrc
source ~/.bashrc

# Verify  
echo $DEER_FLOW_PATH
cli agents deerflow --list
```

### For New Users

Follow the Quick Start in the documentation:

1. Run setup script: `bash scripts/setup_deerflow.sh`
2. Reload shell: `source ~/.bashrc`
3. Test: `cli agents deerflow --list`

## Testing

### Verification Steps

1. **Path resolution works**:
   ```bash
   DEER_FLOW_PATH=/path/to/deer-flow python -c "
   from genai_tk.extra.agents.deer_flow._path_setup import get_deer_flow_backend_path
   print(get_deer_flow_backend_path())
   "
   ```

2. **Error handling works**:
   ```bash
   env -u DEER_FLOW_PATH python -c "
   from genai_tk.extra.agents.deer_flow._path_setup import get_deer_flow_backend_path
   get_deer_flow_backend_path()
   "
   # Should show clear error message
   ```

3. **Agent works end-to-end**:
   ```bash
   DEER_FLOW_PATH=/path/to/deer-flow cli agents deerflow --list
   ```

### Test Results

✅ Path resolution with `DEER_FLOW_PATH` set - **PASSED**  
✅ Error handling without `DEER_FLOW_PATH` - **PASSED**  
✅ Deer-flow agent CLI (`--list`, `--chat`) - **PASSED**  
✅ Setup script syntax validation - **PASSED**

## Files Modified

1. `genai_tk/extra/agents/deer_flow/_path_setup.py` - Simplified to use only DEER_FLOW_PATH
2. `pyproject.toml` - Updated deer-flow dependency group documentation
3. `docs/Deer_Flow_Integration.md` - Updated installation and troubleshooting sections
4. `README.md` - Added Agent Frameworks section

## Files Created

1. `scripts/setup_deerflow.sh` - Automated setup script (executable)
2. `docs/DEER_FLOW_SETUP_CHANGES.md` - This document

## Breaking Changes

⚠️ **Breaking Change**: Users must now set `DEER_FLOW_PATH` environment variable

**Before**: Deer-flow would be auto-discovered from multiple locations  
**After**: Deer-flow location must be explicitly set via `DEER_FLOW_PATH`

**Migration Path**: Use the setup script or set the variable manually (see Migration Guide above)

## Future Improvements

Potential enhancements:

1. **XDG Base Directory Support**: Support `~/.local/share/deer-flow` as standard location
2. **Version Checking**: Validate deer-flow version compatibility
3. **Auto-Update**: Option to auto-update deer-flow repository
4. **Multiple Environments**: Support switching between dev/prod deer-flow installations

## References

- **Main Documentation**: [docs/Deer_Flow_Integration.md](Deer_Flow_Integration.md)
- **Setup Script**: [scripts/setup_deerflow.sh](../scripts/setup_deerflow.sh)
- **Deer-flow Repository**: https://github.com/bytedance/deer-flow
- **Issue/PR**: (if applicable)

## Questions & Support

If you encounter issues:

1. Check `DEER_FLOW_PATH` is set: `echo $DEER_FLOW_PATH`
2. Verify deer-flow exists: `ls $DEER_FLOW_PATH/backend/src`
3. Run setup script: `bash scripts/setup_deerflow.sh`
4. See troubleshooting in [Deer_Flow_Integration.md](Deer_Flow_Integration.md)

---

**Reviewed by**: AI Agent  
**Approved by**: (Pending user review)  
**Status**: ✅ Implemented and tested
