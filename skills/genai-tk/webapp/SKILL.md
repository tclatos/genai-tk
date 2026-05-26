---
name: genai-tk-webapp
description: Add or modify the genai-tk Streamlit webapp, pages, navigation, UI components, and genai_tk:// page references.
---

# GenAI Toolkit Webapp

## Read First

- `docs/webapp.md`
- `genai_tk/webapp/main/streamlit.py`
- `genai_tk/webapp/ui_components/`
- `genai_tk/webapp/pages/`
- `config/webapp.yaml`

## Code Map

| Concern | Paths |
|---|---|
| App entry point | `genai_tk/webapp/main/streamlit.py` |
| Built-in pages | `genai_tk/webapp/pages/` |
| Shared UI components | `genai_tk/webapp/ui_components/` |
| Navigation and app config | `config/webapp.yaml` |
| Streamlit utilities | `genai_tk/utils/streamlit/` |

## Add A Page

1. Place built-in pages under `genai_tk/webapp/pages/...` or project pages under the generated package.
2. Register the page in `config/webapp.yaml` navigation.
3. Use stable Streamlit session state keys prefixed by the page feature.
4. Reuse `genai_tk.webapp.ui_components` for LLM selectors, agent layout, traces, and message rendering.
5. Keep page imports light so app startup stays responsive.

## Commands

```bash
make webapp
uv run streamlit run genai_tk/webapp/main/streamlit.py
```

## Avoid

- Do not store credentials in Streamlit session state.
- Do not block app import with live LLM calls or browser startup.
- Do not duplicate UI components when a reusable component exists under `ui_components/`.
