---
name: add-webapp-page
description: Step-by-step procedure to create a new Streamlit page in a genai-tk project and register it in the webapp navigation.
---

# Add a Webapp Page

Follow these steps to add a new Streamlit page to a genai-tk project.

## Prerequisites
- The project was initialized with `cli init` (has `config/webapp.yaml`)
- The project has a `<package>/webapp/pages/` directory

## Step 1: Create the Streamlit Page

Create a new file in `<package>/webapp/pages/demos/my_page.py`:

```python
"""My custom Streamlit page."""

import streamlit as st
from streamlit import session_state as sss

from genai_tk.core.llm_factory import get_llm
from genai_tk.webapp.ui_components.llm_selector import render_llm_selector

st.set_page_config(page_title="My Page", page_icon="🔧", layout="wide")
st.title("🔧 My Page")
st.caption("Description of what this page does")

# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    llm_id = render_llm_selector(key="my_page_llm")

# ── Main content ────────────────────────────────────────────────────────
if "my_messages" not in sss:
    sss.my_messages = []

# Display existing messages
for msg in sss.my_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    sss.my_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            llm = get_llm(llm_id or "default")
            result = llm.invoke(prompt)
            answer = result.content
            st.markdown(answer)

    sss.my_messages.append({"role": "assistant", "content": answer})
```

## Step 2: Register in webapp.yaml

Edit `config/webapp.yaml` to add the page to navigation:

```yaml
ui:
  pages_dir: <package_name>/webapp/pages
  navigation:
    demos:
      - demos/my_page.py
      # ... other pages ...
    settings:
      - settings/configuration.py
```

## Step 3: Verify

```bash
make webapp
# Navigate to the new page in the sidebar
```

## Key Patterns

- Use `st.set_page_config()` at the top of each page
- Use `render_llm_selector()` for model selection in sidebar
- Use `streamlit.session_state` (aliased as `sss`) for state persistence
- Use unique keys for session state variables to avoid conflicts between pages
- For agent pages, use `langgraph.prebuilt.create_react_agent()` with tools
- For chat UI, use `st.chat_message()` and `st.chat_input()`
- Pages can reference built-in genai-tk pages with `genai_tk://` prefix in navigation

## Available UI Components (from genai_tk)

```python
from genai_tk.webapp.ui_components.llm_selector import render_llm_selector
from genai_tk.webapp.ui_components.agent_layout import render_agent_sidebar
from genai_tk.webapp.ui_components.message_renderer import render_message_with_mermaid
from genai_tk.webapp.ui_components.trace_middleware import TraceMiddleware
```
