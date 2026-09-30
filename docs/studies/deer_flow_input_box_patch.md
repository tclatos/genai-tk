# Patch: deer-flow frontend stale model fallback

**File:** `frontend/src/components/workspace/input-box.tsx`
**Function:** `InputBox` component — `selectedModel` useMemo

## Problem

DeerFlow stores the last-used model name in `localStorage` under the key
`deerflow.local-settings`.  When the backend config changes (e.g. after running
`cli agents deerflow --web`), the model list served by the backend no longer
contains the old model name.  The original code only auto-selected the first
model when `context.model_name` was empty; a non-empty but *stale* name was
passed unchanged to the backend, causing:

```
ValueError: Model gpt-4 not found in config
```

Queries silently produced no output.

## Fix

Unify the "no model set" and "saved model not found" cases into a single
fallback that auto-selects `models[0]` and updates the context (which
overwrites localStorage).

### Before

```tsx
const selectedModel = useMemo(() => {
  if (!context.model_name && models.length > 0) {
    const model = models[0]!;
    setTimeout(() => {
      onContextChange?.({
        ...context,
        model_name: model.name,
        mode: model.supports_thinking ? "pro" : "flash",
      });
    }, 0);
    return model;
  }
  return models.find((m) => m.name === context.model_name);
}, [context, models, onContextChange]);
```

### After

```tsx
const selectedModel = useMemo(() => {
  const found = models.find((m) => m.name === context.model_name);
  // Auto-select the first model when none is set OR the saved model no longer exists
  if (!found && models.length > 0) {
    const model = models[0]!;
    setTimeout(() => {
      onContextChange?.({
        ...context,
        model_name: model.name,
        mode: model.supports_thinking ? "pro" : "flash",
      });
    }, 0);
    return model;
  }
  return found;
}, [context, models, onContextChange]);
```

## How to re-apply

If the `deer-flow` repo is reset or updated over this change, apply the diff
above manually, or run from the repo root:

```bash
cd /home/tcl/ext_prj/deer-flow
git diff HEAD frontend/src/components/workspace/input-box.tsx
```

to confirm the patch is present, and re-edit the file if it is not.
