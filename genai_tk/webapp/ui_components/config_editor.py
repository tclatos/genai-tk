"""Reusable configuration editor dialog for Streamlit applications.

Requires the optional ``streamlit-monaco`` package.  Install it with:
    uv add streamlit-monaco
"""

from pathlib import Path

import streamlit as st
import yaml


@st.dialog("Edit Configuration", width="large")
def edit_config_dialog(config_path: str | Path) -> None:
    """Open a dialog to edit a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        st.error(f"Configuration file not found: {config_path}")
        return

    try:
        from streamlit_monaco import st_monaco
    except ImportError:
        st.warning(
            "Install `streamlit-monaco` to enable the visual editor:\n"
            "```\nuv add streamlit-monaco\n```"
        )
        with open(config_path, encoding="utf-8") as f:
            st.code(f.read(), language="yaml")
        return

    try:
        with open(config_path, encoding="utf-8") as f:
            current_content = f.read()

        edited_content = st_monaco(
            value=current_content, height="400px", language="yaml", theme="vs-dark", minimap=False, lineNumbers=True
        )

        col1, col2, _ = st.columns([1, 1, 2])

        if col1.button("💾 Save", width="stretch"):
            try:
                yaml.safe_load(edited_content)
                with open(config_path, "w", encoding="utf-8") as f:
                    f.write(edited_content)
                st.success("Configuration saved successfully!")
                st.rerun()
            except yaml.YAMLError as e:
                st.error(f"Invalid YAML: {e}")
            except Exception as e:
                st.error(f"Error saving file: {e}")

        if col2.button("❌ Cancel", width="stretch"):
            st.rerun()

    except Exception as e:
        st.error(f"Error loading configuration: {e}")
