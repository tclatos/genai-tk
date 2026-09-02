"""NeMo Relay observability bootstrap + local trajectory store (Phase 1 write-side).

Wires the NeMo Relay runtime into the genai-tk agent flow so agent runs emit the
canonical Agent Trajectory Observability Format (ATOF) event stream to a local
store — the first-class, agent-readable trajectory record.

Two capture modes:

- **Per-session store** (default, ``setup_nemo_relay()`` with no ``atof_path``):
  each top-level agent scope (a run) is written under
  ``<data_root>/trajectories/<root_uuid>/events.jsonl`` with a sibling
  ``meta.json`` summary, and one line per run is appended to
  ``<data_root>/trajectories/index.jsonl``. Events are grouped to their root
  scope by walking the ``parent_uuid`` chain. This is the source-of-truth store
  consumed by ``genai_tk.utils.trajectory_store`` and the ``cli trajectory``
  command group.
- **Single-file** (``setup_nemo_relay(atof_path=...)``): appends every event
  to one JSONL file. Used by the Phase-0 smoke tests.

Usage is automatic: :func:`genai_tk.utils.tracing.setup_monitoring` calls
:func:`setup_nemo_relay` once at startup, and the harness / agent invoke config
attach :func:`get_relay_callback_handler` so the LangGraph run hierarchy maps to
Relay agent scopes. The deep-agent factory additionally injects
``NemoRelayDeepAgentsMiddleware`` via ``add_nemo_relay_integration`` so model
and tool calls are wrapped as Relay ``llm`` / ``tool`` scopes.
"""

from __future__ import annotations

import atexit
import json
import threading
from pathlib import Path
from typing import Any

from loguru import logger

_SUBSCRIBER_NAME = "genai-tk-atof"


def _default_store_dir() -> Path:
    """Resolve the default per-session trajectory store root from config."""
    try:
        from genai_tk.config_mgmt.config_mngr import paths_config

        return Path(paths_config().data_root) / "trajectories"
    except Exception:
        return Path("data/trajectories")


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(tz=timezone.utc).isoformat()


# ── Per-session store state ──────────────────────────────────────────────────


class _SessionRun:
    """Aggregates for one top-level agent scope (one run)."""

    def __init__(self, root_uuid: str, root_name: str, started_at: str) -> None:
        self.root_uuid = root_uuid
        self.root_name = root_name
        self.started_at = started_at
        self.ended_at: str | None = None
        self.status: str = "ok"
        self.n_scopes = 0
        self.n_llm = 0
        self.n_tool = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.tools: list[str] = []
        self.skills: list[str] = []

    def meta(self, run_dir: Path) -> dict[str, Any]:
        """Return the ``meta.json`` summary dict for this run."""
        return {
            "run_id": self.root_uuid,
            "profile": self.root_name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "n_scopes": self.n_scopes,
            "n_llm_calls": self.n_llm,
            "n_tool_calls": self.n_tool,
            "total_prompt_tokens": self.prompt_tokens,
            "total_completion_tokens": self.completion_tokens,
            "tools": sorted(set(self.tools)),
            "skills_loaded": sorted(set(self.skills)),
            "events_path": str(run_dir / "events.jsonl"),
        }

    def index_line(self, run_dir: Path) -> dict[str, Any]:
        """Return one ``index.jsonl`` line for this run."""
        m = self.meta(run_dir)
        # Keep the index line compact — drop the events_path (derivable).
        return {
            "run_id": m["run_id"],
            "profile": m["profile"],
            "started_at": m["started_at"],
            "ended_at": m["ended_at"],
            "status": m["status"],
            "n_llm_calls": m["n_llm_calls"],
            "n_tool_calls": m["n_tool_calls"],
            "total_prompt_tokens": m["total_prompt_tokens"],
            "total_completion_tokens": m["total_completion_tokens"],
            "tools": m["tools"],
            "skills_loaded": m["skills_loaded"],
        }


class _StoreState:
    """Per-session store capture state."""

    def __init__(self, store_dir: Path) -> None:
        self.store_dir = store_dir
        self._lock = threading.Lock()
        # uuid -> parent_uuid (ancestry within the current run)
        self.parents: dict[str, str | None] = {}
        # uuid -> root_uuid (cache)
        self.roots: dict[str, str] = {}
        self.current: _SessionRun | None = None
        self.current_file: Any = None
        self.current_dir: Path | None = None

    def _root_of(self, uuid: str, parent_uuid: str | None) -> str:
        """Resolve the root scope uuid for an event by walking the parent chain."""
        if uuid in self.roots:
            return self.roots[uuid]
        if parent_uuid is None:
            root = uuid
        else:
            root = self.roots.get(parent_uuid, parent_uuid)
            # Walk up until we find an ancestor with no parent.
            while True:
                p = self.parents.get(root)
                if p is None or root == p:
                    break
                root = p
        self.roots[uuid] = root
        return root

    def _close_current(self) -> None:
        """Finalize the current run: write meta.json + index line, close file."""
        if self.current is None:
            return
        try:
            if self.current_dir is not None:
                (self.current_dir / "meta.json").write_text(
                    json.dumps(self.current.meta(self.current_dir), indent=2), encoding="utf-8"
                )
                with (self.store_dir / "index.jsonl").open("a", encoding="utf-8") as idx:
                    idx.write(json.dumps(self.current.index_line(self.current_dir)) + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"NeMo Relay store finalize failed: {exc}")
        if self.current_file is not None:
            try:
                self.current_file.close()
            except Exception:  # noqa: BLE001
                pass
        self.current_file = None
        self.current = None
        self.current_dir = None

    def write(self, event: dict[str, Any], raw: str) -> None:
        """Route one event to its run's events.jsonl, opening/rotating as needed."""
        with self._lock:
            uuid = event.get("uuid")
            parent = event.get("parent_uuid")
            if not isinstance(uuid, str):
                return
            self.parents[uuid] = parent
            root = self._root_of(uuid, parent)

            if self.current is None or self.current.root_uuid != root:
                self._close_current()
                run_dir = self.store_dir / root
                run_dir.mkdir(parents=True, exist_ok=True)
                self.current_dir = run_dir
                self.current_file = (run_dir / "events.jsonl").open("a", encoding="utf-8")
                # Root scope name + start ts from the root event itself.
                name = str(event.get("name") or "agent")
                ts = str(event.get("timestamp") or _now_iso())
                self.current = _SessionRun(root, name, ts)

            # Aggregate from this event.
            kind = event.get("kind")
            cat = event.get("category")
            sc = event.get("scope_category")
            if kind == "scope" and sc == "start":
                self.current.n_scopes += 1
                if cat == "llm":
                    self.current.n_llm += 1
                elif cat == "tool":
                    self.current.n_tool += 1
                    if isinstance(event.get("name"), str):
                        self.current.tools.append(event["name"])
            elif kind == "scope" and sc == "end":
                if cat == "llm":
                    cp = event.get("category_profile") or {}
                    ar = cp.get("annotated_response") or {}
                    usage = ar.get("usage") or {}
                    if isinstance(usage, dict):
                        self.current.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                        self.current.completion_tokens += int(usage.get("completion_tokens") or 0)
                    # If the scope end carries ERROR status, mark the run errored.
                    meta = event.get("metadata") or {}
                    if isinstance(meta, dict) and meta.get("otel.status_code") == "ERROR":
                        self.current.status = "error"
                elif cat == "agent" and self.current.root_uuid == uuid:
                    # Last agent-end for the root wins (middleware before_agent
                    # hooks push/pop agent scopes with the same uuid first).
                    self.current.ended_at = str(event.get("timestamp") or _now_iso())
                    meta = event.get("metadata") or {}
                    if isinstance(meta, dict) and meta.get("otel.status_code") == "ERROR":
                        self.current.status = "error"
            elif kind == "mark" and event.get("name") == "skill.load":
                data = event.get("data") or {}
                if isinstance(data, dict) and isinstance(data.get("skill_name"), str):
                    self.current.skills.append(data["skill_name"])

            try:
                self.current_file.write(raw + "\n")
                self.current_file.flush()
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"NeMo Relay store write failed: {exc}")

    def flush(self) -> None:
        """Flush the current run's file (keeps the run open)."""
        with self._lock:
            if self.current_file is not None:
                try:
                    self.current_file.flush()
                except Exception:  # noqa: BLE101
                    pass

    def close(self) -> None:
        """Finalize and close the current run."""
        with self._lock:
            self._close_current()


# ── Single-file state (spike / tests) ────────────────────────────────────────


class _FileState:
    """Single-file capture state (Phase-0 spike mode)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._file = path.open("a", encoding="utf-8")

    def write(self, raw: str) -> None:
        with self._lock:
            try:
                self._file.write(raw + "\n")
                self._file.flush()
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"NeMo Relay ATOF write failed: {exc}")

    def close(self) -> None:
        with self._lock:
            try:
                self._file.close()
            except Exception:  # noqa: BLE101
                pass


# ── Singleton facade ─────────────────────────────────────────────────────────


class _RelayState:
    """Active capture state — either a per-session store or a single file."""

    def __init__(self) -> None:
        self.active = False
        self.store: _StoreState | None = None
        self.file: _FileState | None = None
        self.path: Path | None = None  # single-file path (test mode)

    def on_event(self, event: Any) -> None:
        try:
            raw = event.to_json()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"NeMo Relay subscriber callback failed: {exc}")
            return
        try:
            parsed = json.loads(raw)
        except Exception:  # noqa: BLE001
            parsed = {}
        if self.store is not None:
            self.store.write(parsed, raw)
        elif self.file is not None:
            self.file.write(raw)


_state = _RelayState()
_setup_lock = threading.Lock()


def is_nemo_relay_available() -> bool:
    """Return True if the ``nemo_relay`` package is importable."""
    try:
        import nemo_relay  # noqa: F401

        return True
    except ImportError:
        return False


def setup_nemo_relay(*, atof_path: Path | None = None, store_dir: Path | None = None) -> bool:
    """Register the ATOF subscriber and atexit flush.

    Idempotent. With ``atof_path`` → single-file mode (spike/tests). Without it
    → per-session store mode under ``store_dir`` (default
    ``<data_root>/trajectories``). No-op when ``nemo_relay`` is not installed.

    Args:
        atof_path: Explicit single JSONL output path (test mode).
        store_dir: Explicit per-session store root (defaults to config data_root).

    Returns:
        True if the subscriber is active after the call.
    """
    with _setup_lock:
        if _state.active:
            return True
        if not is_nemo_relay_available():
            logger.debug("NeMo Relay not installed — skipping ATOF subscriber setup")
            return False

        import nemo_relay

        if atof_path is not None:
            path = Path(atof_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            _state.file = _FileState(path)
            _state.path = path
            where = f"file {path}"
        else:
            sdir = Path(store_dir) if store_dir is not None else _default_store_dir()
            sdir.mkdir(parents=True, exist_ok=True)
            _state.store = _StoreState(sdir)
            where = f"store {sdir}"

        try:
            nemo_relay.subscribers.register(_SUBSCRIBER_NAME, _state.on_event)
        except RuntimeError as exc:
            if "already exists" in str(exc).lower():
                logger.debug(f"NeMo Relay subscriber {_SUBSCRIBER_NAME} already registered: {exc}")
            else:
                raise
        atexit.register(_atexit_flush)
        _state.active = True
        logger.info(f"NeMo Relay ATOF subscriber active → {where}")
        return True


def _atexit_flush() -> None:
    if not _state.active:
        return
    try:
        import nemo_relay

        try:
            nemo_relay.subscribers.flush()
        except RuntimeError:
            pass
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"NeMo Relay atexit flush failed: {exc}")
    if _state.store is not None:
        _state.store.close()
    elif _state.file is not None:
        _state.file.close()


def flush_nemo_relay() -> None:
    """Synchronously flush queued ATOF events (no running asyncio loop only)."""
    if not _state.active:
        return
    try:
        import nemo_relay

        nemo_relay.subscribers.flush()
    except RuntimeError:
        logger.debug("NeMo Relay sync flush skipped (asyncio loop running); use flush_nemo_relay_async()")
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"NeMo Relay flush failed: {exc}")
    if _state.store is not None:
        _state.store.flush()


async def flush_nemo_relay_async() -> None:
    """Asynchronously flush queued ATOF events from an asyncio task."""
    if not _state.active:
        return
    try:
        import nemo_relay

        await nemo_relay.subscribers.flush_async()
    except Exception as exc:  # noqa: BLE101
        logger.debug(f"NeMo Relay async flush failed: {exc}")
    if _state.store is not None:
        _state.store.flush()


def get_relay_callback_handler() -> Any | None:
    """Return a Relay LangGraph/LangChain callback handler, or None if unavailable.

    Maps the compiled-graph run hierarchy to Relay agent scopes (``on_chain_*``
    → ``scope.push/pop``) and emits human-in-the-loop interrupt/resume marks for
    Deep Agents runs. Prefers the Deep Agents handler; falls back to the plain
    LangChain handler when the ``deepagents`` extra is not installed.
    """
    if not is_nemo_relay_available():
        return None
    try:
        from nemo_relay.integrations.deepagents import NemoRelayDeepAgentsCallbackHandler

        return NemoRelayDeepAgentsCallbackHandler()
    except ImportError:
        try:
            from nemo_relay.integrations.langchain.callbacks import NemoRelayCallbackHandler

            return NemoRelayCallbackHandler()
        except ImportError:
            logger.debug("NeMo Relay callback handler integration unavailable")
            return None


def reset_nemo_relay() -> None:
    """Tear down the subscriber and reset state (used in tests)."""
    with _setup_lock:
        if not _state.active:
            return
        try:
            import nemo_relay

            nemo_relay.subscribers.deregister(_SUBSCRIBER_NAME)
        except Exception:  # noqa: BLE001
            pass
        if _state.store is not None:
            _state.store.close()
        elif _state.file is not None:
            _state.file.close()
        _state.active = False
        _state.store = None
        _state.file = None
        _state.path = None
