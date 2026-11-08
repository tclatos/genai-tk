---
name: webapp
description: Add or modify the genai-tk Streamlit webapp, pages, navigation, UI components, and genai_tk:// page references.
---

# GenAI Toolkit Webapp

## Read First

- `docs/webapp.md`
- `genai_tk/ui/streamlit/apps/streamlit.py`
- `genai_tk/ui/streamlit/components/`
- `genai_tk/ui/streamlit/pages/`
- `config/webapp.yaml`

## Code Map

| Concern | Paths |
|---|---|
| App entry point | `genai_tk/ui/streamlit/apps/streamlit.py` |
| Built-in pages | `genai_tk/ui/streamlit/pages/` |
| Shared UI components | `genai_tk/ui/streamlit/components/` |
| Navigation and app config | `config/webapp.yaml`, `genai_tk/ui/config/webapp.py` |
| Streamlit utilities | `genai_tk/ui/streamlit/utils/` |

## Add A Page

1. Place built-in pages under `genai_tk/ui/streamlit/pages/...` or project pages under the generated package.
2. Register the page in `config/webapp.yaml` navigation.
3. Use stable Streamlit session state keys prefixed by the page feature.
4. Reuse `genai_tk.ui.streamlit.components` for LLM selectors, agent layout, traces, and message rendering.
5. Keep page imports light so app startup stays responsive.

## Commands

```bash
just webapp
uv run python -m streamlit run genai_tk/ui/streamlit/apps/streamlit.py
```

## Avoid

- Do not store credentials in Streamlit session state.
- Do not block app import with live LLM calls or browser startup.
- Do not duplicate UI components when a reusable component exists under `ui/streamlit/components/`.
