"""genai-tk built-in Streamlit webapp entry point.

Provides a minimal web UI for testing agent demos:
- 🦌 DeerFlow Agent
- 🤖 Deep Agent

Run with:
    make webapp
or:
    uv run streamlit run genai_tk/webapp/main/streamlit.py

Configuration keys (all optional):
    ui.app_name   — browser tab title and sidebar header
    ui.logo       — path to logo image (relative to CWD or absolute)

Additional pages can be added by configuring ``ui.navigation`` and
``ui.pages_dir`` in your project's config (same format as genai-blueprint).
"""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from loguru import logger

from genai_tk.utils.basic_auth import authenticate, load_auth_config
from genai_tk.utils.config_mngr import global_config
from genai_tk.utils.logger_factory import setup_logging

load_dotenv()
setup_logging()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

_app_name = global_config().get_str("ui.app_name", default=Path.cwd().name)

st.set_page_config(
    page_title=_app_name,
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Authentication (optional — only enforced when auth is enabled in config)
# ---------------------------------------------------------------------------

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

try:
    auth_config = load_auth_config()
except Exception:
    auth_config = None

if auth_config is not None and auth_config.enabled and not st.session_state.authenticated:
    st.title("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid username or password")
    st.stop()

# ---------------------------------------------------------------------------
# Logo (optional)
# ---------------------------------------------------------------------------

_logo_path: str | None = global_config().get_str("ui.logo", default=None)
if _logo_path:
    _logo_resolved = Path(_logo_path) if Path(_logo_path).is_absolute() else Path.cwd() / _logo_path
    if _logo_resolved.exists():
        st.logo(str(_logo_resolved), size="medium")
    else:
        logger.warning("Logo not found: {}", _logo_resolved)

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

# Check whether the project has customised the navigation via config
try:
    _nav_config: dict | None = global_config().get_dict("ui.navigation")
except Exception:
    _nav_config = None

_pages_dir_str: str | None = global_config().get_str("ui.pages_dir", default=None)

if _nav_config and _pages_dir_str:
    # Project-level navigation: load pages from the configured directory + structure
    _pages_dir = Path(_pages_dir_str)

    def _file_name_to_page_name(file_name: str) -> str:
        """Convert a file name to a formatted page title."""
        # Well-known overrides for built-in genai-tk pages
        _KNOWN_TITLES = {
            "deer_flow_agent": "🦌 DeerFlow Agent",
            "reAct_agent": "🤖 ReAct Agent",
            "smolagents_streamlit": "🤖 SmolAgents",
        }
        try:
            name_only = file_name.split("/")[-1].rsplit(".", 1)[0]
            if name_only in _KNOWN_TITLES:
                return _KNOWN_TITLES[name_only]
            parts = name_only.split("_")
            start = 1 if (parts and parts[0].isdigit()) else 0
            words = []
            for w in parts[start:]:
                if not w:
                    continue
                if any(c.isupper() for c in w[1:]):
                    words.append(w[0].upper() + w[1:])
                elif w == w.upper() and len(w) > 1:
                    words.append(w)
                else:
                    words.append(w.capitalize())
            return " ".join(words) if words else file_name
        except Exception:
            return file_name

    pages: dict[str, list] = {}
    for section_name, page_files in _nav_config.items():
        section_pages = []
        for page_file in page_files:
            # Resolve page path:
            #   genai_tk://path  → installed genai_tk/webapp/path
            #   /absolute/path   → as-is
            #   relative/path    → relative to pages_dir
            if page_file.startswith("genai_tk://"):
                rel = page_file[len("genai_tk://") :]
                from importlib.resources import files as _pkg_files

                try:
                    page_path = Path(str(_pkg_files("genai_tk") / "webapp" / "pages" / rel))
                except Exception:
                    logger.warning("Could not resolve genai_tk package path: {}", page_file)
                    continue
            elif Path(page_file).is_absolute():
                page_path = Path(page_file)
            else:
                page_path = _pages_dir / page_file

            if page_path.exists():
                section_pages.append(st.Page(page=page_path, title=_file_name_to_page_name(page_file)))
            else:
                logger.warning("Page not found: {}", page_path)
        if section_pages:
            pages[section_name.title()] = section_pages

    pg = st.navigation(pages, position="top")
    pg.run()

else:
    # Default: built-in demo pages only
    _here = Path(__file__).parent.parent  # genai_tk/webapp/
    _demos_dir = _here / "pages" / "demos"

    pages = {
        "Agent Demos": [
            st.Page(page=_demos_dir / "deer_flow_agent.py", title="🦌 DeerFlow Agent"),
            st.Page(page=_demos_dir / "reAct_agent.py", title="🤖 ReAct Agent"),
        ]
    }

    pg = st.navigation(pages, position="top")
    pg.run()
