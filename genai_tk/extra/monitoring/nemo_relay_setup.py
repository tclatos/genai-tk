"""NeMo Relay observability bootstrap + local trajectory store (Phase 1 write-side).

Wires the NeMo Relay runtime into the genai-tk agent flow so agent runs emit the
canonical Agent Trajectory Observability Format (ATOF) event stream to a local
store — the first-class, agent-readable trajectory record.

Two capture modes:

- **Per-session store** (default, ``setup_nemo_relay()`` with no ``atof_path``):
  each top-level agent scope (a run) is written under
  ``<data_root>/trajectories/<root_uuid>/events.jsonl`` with a sibling
  ``meta.json`` summary, and exactly one line per run is upserted into
  ``<data_root>/trajectories/index.jsonl``. Events are grouped to their root
  scope by walking the ``parent_uuid`` chain, and every run keeps a single
  aggregate for the process, so ``meta.json`` / ``index.jsonl`` / ``events.jsonl``
  always report the same call and token counts. This is the source-of-truth store
  consumed by ``genai_tk.extra.monitoring.trajectory_store`` and the ``cli trajectory``
  command group.
- **Single-file** (``setup_nemo_relay(atof_path=...)``): appends every event
  to one JSONL file. Used by the Phase-0 smoke tests.

Usage is automatic: :func:`genai_tk.extra.monitoring.tracing.setup_monitoring` calls
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
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

_SUBSCRIBER_NAME = "genai-tk-atof"

# At most this many runs keep their events file open at once. A session normally
# has one run, so this only matters when several agents share a process; beyond
# it the least-recently-used run is finalized (meta + index line) and reopened on
# demand, which never resets its aggregate.
_MAX_OPEN_RUNS = 8


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
        self.deepagent_skills: list[str] = []  # source dirs from the aggregated config mark

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
            "skills_loaded": sorted(set(self.skills)) if self.skills else sorted(set(self.deepagent_skills)),
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
    """Per-session store capture state.

    Events are grouped into runs by walking the ``parent_uuid`` chain, and every
    run keeps **one** aggregate for the whole process — a run is never split.
    Two properties make the stored numbers match ``events.jsonl``:

    - **Late parents.** Subscribers are called from several threads, so a child
      scope's event can arrive before its parent's first event. A parent that is
      not known yet is attributed to the run that is already open instead of
      being mistaken for a new root. NeMo Relay's decision-model evaluator
      scopes are emitted from worker threads and carry a parent id that never
      appears as an event uuid at all, which is exactly this case; a top-level
      ``agent`` scope start still defines a run.
    - **One index line per run.** ``index.jsonl`` is upserted, so re-finalizing a
      run rewrites its line instead of appending a second, partial one. An Alko
      classification was rotated mid-flight and reported 5 + 9 LLM calls and
      116 668 + 465 731 prompt tokens in two lines for one 14-call run, so any
      dashboard reading a single line understated the run by 60%.
    """

    def __init__(self, store_dir: Path) -> None:
        self.store_dir = store_dir
        self._lock = threading.Lock()
        # uuid -> root_uuid (the run a scope belongs to)
        self.roots: dict[str, str] = {}
        # root_uuid -> aggregate / run directory / open events file
        self.runs: dict[str, _SessionRun] = {}
        self.run_dirs: dict[str, Path] = {}
        self.files: dict[str, Any] = {}
        # root uuids by recency of use (most recently used last)
        self.order: list[str] = []

    def _root_of(self, event: dict[str, Any]) -> str:
        """Resolve which run a scope event belongs to, tolerating a late parent.

        A scope's run is inherited from its parent's own resolution, which is
        cached when that parent's first event is processed. Walking the raw
        parent chain instead would let a *child* of an unattributed scope define a
        run of its own — the exact failure that split the Alko run in two.
        """
        uuid = str(event["uuid"])
        parent_uuid = event.get("parent_uuid")
        cached = self.roots.get(uuid)
        if cached is not None:
            return cached

        is_run_start = (
            event.get("kind") == "scope" and event.get("category") == "agent" and event.get("scope_category") == "start"
        )
        if parent_uuid is None:
            root = uuid
        elif parent_uuid in self.roots:
            root = self.roots[parent_uuid]
        elif is_run_start or not self.order:
            # A top-level agent scope whose own parent is the LangGraph trace
            # root: that parent is never instrumented, so the run is named after
            # it (and it is the first event of a session).
            root = str(parent_uuid)
        else:
            # The parent scope's own event has not been delivered (or never will
            # be). Attribute this event to the run that is already open rather
            # than inventing a run for the parent.
            root = self.order[-1]
            logger.debug(
                "trajectory store: unknown parent {} on {} — attributing to run {}",
                parent_uuid,
                event.get("name"),
                root,
            )
        self.roots[uuid] = root
        return root

    def _touch(self, root: str) -> None:
        """Mark *root* as the most recently used run."""
        if root in self.order:
            self.order.remove(root)
        self.order.append(root)

    def _run_for(self, root: str, event: dict[str, Any]) -> _SessionRun:
        """Return the aggregate for *root*, creating the run the first time it is seen."""
        run = self.runs.get(root)
        if run is None:
            run_dir = self.store_dir / root
            run_dir.mkdir(parents=True, exist_ok=True)
            run = _SessionRun(root, str(event.get("name") or "agent"), str(event.get("timestamp") or _now_iso()))
            self.runs[root] = run
            self.run_dirs[root] = run_dir
        return run

    def _file_for(self, root: str) -> Any:
        """Return *root*'s open events file, capping how many handles stay open."""
        handle = self.files.get(root)
        if handle is not None:
            return handle
        handle = (self.run_dirs[root] / "events.jsonl").open("a", encoding="utf-8")
        self.files[root] = handle
        while len(self.files) > _MAX_OPEN_RUNS:
            stale = next((r for r in self.order if r in self.files and r != root), None)
            if stale is None:
                break
            self._finalize(stale)
        return handle

    def _finalize(self, root: str) -> None:
        """Write *root*'s meta.json, upsert its index line and close its file."""
        run = self.runs.get(root)
        run_dir = self.run_dirs.get(root)
        if run is not None and run_dir is not None:
            if run.ended_at is None:
                # No event carried a timestamp; fall back to wall-clock now so
                # meta.json / index lines never carry a null ended_at.
                run.ended_at = _now_iso()
            try:
                (run_dir / "meta.json").write_text(json.dumps(run.meta(run_dir), indent=2), encoding="utf-8")
                self._upsert_index_line(run.index_line(run_dir))
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"NeMo Relay store finalize failed: {exc}")
        handle = self.files.pop(root, None)
        if handle is not None:
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass
        if root in self.order:
            self.order.remove(root)

    def _upsert_index_line(self, line: dict[str, Any]) -> None:
        """Write *line* into ``index.jsonl``, replacing any existing line for that run."""
        index = self.store_dir / "index.jsonl"
        run_id = line.get("run_id")
        kept: list[str] = []
        if index.exists():
            for existing in index.read_text(encoding="utf-8").splitlines():
                if not existing.strip():
                    continue
                try:
                    if json.loads(existing).get("run_id") == run_id:
                        continue
                except Exception:  # noqa: BLE001
                    pass
                kept.append(existing)
        kept.append(json.dumps(line))
        tmp = index.with_name(index.name + ".tmp")
        tmp.write_text("\n".join(kept) + "\n", encoding="utf-8")
        tmp.replace(index)

    def write(self, event: dict[str, Any], raw: str) -> None:
        """Route one event to its run's events.jsonl, aggregating as it goes."""
        with self._lock:
            uuid = event.get("uuid")
            if not isinstance(uuid, str):
                return
            root = self._root_of(event)
            run = self._run_for(root, event)
            self._touch(root)
            handle = self._file_for(root)

            # The run's window ends at its last event; an agent scope end is the
            # usual last event, but the trace root is not always instrumented.
            ts = event.get("timestamp")
            if isinstance(ts, str) and ts:
                run.ended_at = ts

            # Aggregate from this event.
            kind = event.get("kind")
            cat = event.get("category")
            sc = event.get("scope_category")
            if kind == "scope" and sc == "start":
                run.n_scopes += 1
                if cat == "llm":
                    run.n_llm += 1
                elif cat == "tool":
                    run.n_tool += 1
                    if isinstance(event.get("name"), str):
                        run.tools.append(event["name"])
            elif kind == "scope" and sc == "end":
                if cat == "llm":
                    cp = event.get("category_profile") or {}
                    ar = cp.get("annotated_response") or {}
                    usage = ar.get("usage") or {}
                    if isinstance(usage, dict):
                        run.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                        run.completion_tokens += int(usage.get("completion_tokens") or 0)
                # If a scope end carries ERROR status, mark the run errored.
                meta = event.get("metadata") or {}
                if isinstance(meta, dict) and meta.get("otel.status_code") == "ERROR":
                    run.status = "error"
            elif kind == "mark":
                if event.get("name") == "skill.load":
                    data = event.get("data") or {}
                    if isinstance(data, dict) and isinstance(data.get("skill_name"), str):
                        run.skills.append(data["skill_name"])
                else:
                    mmeta = event.get("metadata") or {}
                    if isinstance(mmeta, dict) and mmeta.get("deepagents_kind") == "skill":
                        # DeepAgents 'Skills Configured' mark only carries the
                        # skill *source directories* — fallback for the index
                        # when the run has no proper ``skill.load`` marks.
                        data = event.get("data") or {}
                        skills = data.get("skills") if isinstance(data, dict) else None
                        if isinstance(skills, list):
                            run.deepagent_skills.extend(s for s in skills if isinstance(s, str))

            try:
                handle.write(raw + "\n")
                handle.flush()
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"NeMo Relay store write failed: {exc}")

    def flush(self) -> None:
        """Flush every open run's file (keeps the runs open)."""
        with self._lock:
            for handle in list(self.files.values()):
                try:
                    handle.flush()
                except Exception:  # noqa: BLE101
                    pass

    def close(self) -> None:
        """Finalize and close every run."""
        with self._lock:
            for root in list(self.runs):
                self._finalize(root)


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
    """Active capture state — either a per-session store or a single file, plus optional OTLP."""

    def __init__(self) -> None:
        self.active = False
        self.store: _StoreState | None = None
        self.file: _FileState | None = None
        self.path: Path | None = None  # single-file path (test mode)
        self.otel_subscriber: Any | None = None
        self.otel_subscriber_name: str | None = None
        self.otel_endpoint: str | None = None
        self.listeners: list[Callable[[dict[str, Any]], None]] = []

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

        # Notify active event listeners (e.g. AUIP bridge, streaming adapters)
        for listener in list(self.listeners):
            try:
                listener(parsed)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Relay event listener failed: {exc}")


_state = _RelayState()
_setup_lock = threading.Lock()


def add_relay_event_listener(listener: Callable[[dict[str, Any]], None]) -> None:
    """Register a listener for live NeMo Relay lifecycle events."""
    with _setup_lock:
        if listener not in _state.listeners:
            _state.listeners.append(listener)


def remove_relay_event_listener(listener: Callable[[dict[str, Any]], None]) -> None:
    """Remove a previously registered live NeMo Relay event listener."""
    with _setup_lock:
        if listener in _state.listeners:
            _state.listeners.remove(listener)


def is_nemo_relay_available() -> bool:
    """Return True if the ``nemo_relay`` package is importable."""
    try:
        import nemo_relay  # noqa: F401

        return True
    except ImportError:
        return False


def setup_nemo_relay(
    *,
    atof_path: Path | None = None,
    store_dir: Path | None = None,
    enable_otlp: bool | None = None,
    otlp_endpoint: str | None = None,
    otlp_headers: dict[str, str] | None = None,
    service_name: str | None = None,
) -> bool:
    """Register the ATOF subscriber and atexit flush, plus optional OTLP export.

    Idempotent. With ``atof_path`` → single-file mode (spike/tests). Without it
    → per-session store mode under ``store_dir`` (default
    ``<data_root>/trajectories``).
    When ``enable_otlp`` is True or when Langfuse/OTEL is active in monitoring config,
    also configures NeMo Relay's native OpenTelemetrySubscriber to stream OTLP spans
    (e.g. to Langfuse at /api/public/otel/v1/traces).

    Args:
        atof_path: Explicit single JSONL output path (test mode).
        store_dir: Explicit per-session store root (defaults to config data_root).
        enable_otlp: Force enable/disable Relay OTLP exporter (defaults to checking config).
        otlp_endpoint: Optional override endpoint for OTLP export.
        otlp_headers: Optional headers (e.g. Authorization) for OTLP export.
        service_name: Service name attribute on exported OTLP spans.

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

        # Check and configure native OpenTelemetrySubscriber export (e.g. to Langfuse)
        _setup_relay_otlp(
            enable_otlp=enable_otlp,
            otlp_endpoint=otlp_endpoint,
            otlp_headers=otlp_headers,
            service_name=service_name,
        )

        atexit.register(_atexit_flush)
        _state.active = True
        logger.info(f"NeMo Relay ATOF subscriber active → {where}")
        return True


def _setup_relay_otlp(
    *,
    enable_otlp: bool | None = None,
    otlp_endpoint: str | None = None,
    otlp_headers: dict[str, str] | None = None,
    service_name: str | None = None,
) -> None:
    """Configure NeMo Relay's native OpenTelemetrySubscriber if requested or configured."""
    import base64
    import os

    import nemo_relay

    should_export = enable_otlp
    endpoint = otlp_endpoint
    headers: dict[str, str] = dict(otlp_headers or {})
    resolved_service = service_name

    if should_export is None or endpoint is None:
        try:
            from genai_tk.extra.monitoring.tracing import monitoring_config

            m_cfg = monitoring_config()
            if should_export is None:
                should_export = m_cfg.is_active("langfuse") or m_cfg.is_active("otel")

            if endpoint is None and m_cfg.is_active("langfuse"):
                lf = m_cfg.langfuse
                effective_host = (
                    lf.host
                    if lf.host and lf.host != "http://localhost:3000"
                    else os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST") or lf.host
                )
                endpoint = f"{effective_host.rstrip('/')}/api/public/otel/v1/traces"
                pk = lf.public_key or os.environ.get("LANGFUSE_PUBLIC_KEY", "")
                sk = lf.secret_key or os.environ.get("LANGFUSE_SECRET_KEY", "")
                if pk and sk and "Authorization" not in headers:
                    token = base64.b64encode(f"{pk}:{sk}".encode()).decode()
                    headers["Authorization"] = f"Basic {token}"
            elif endpoint is None and m_cfg.is_active("otel"):
                endpoint = m_cfg.otel.endpoint
                if not endpoint.endswith("/v1/traces"):
                    endpoint = f"{endpoint.rstrip('/')}/v1/traces"
                if m_cfg.otel.headers:
                    headers.update(m_cfg.otel.headers)

            if not resolved_service:
                resolved_service = f"genai-tk-{m_cfg.project}"
        except Exception as exc:
            logger.debug(f"Failed to inspect monitoring_config for Relay OTLP: {exc}")

    if not should_export or not endpoint:
        return

    # NeMo Relay requires explicit per-endpoint headers and disallows process-global
    # OTEL_EXPORTER_OTLP_HEADERS to prevent credential leaks across endpoints.
    # Temporarily shelter the env var during subscriber instantiation.
    sheltered_headers = os.environ.pop("OTEL_EXPORTER_OTLP_HEADERS", None)
    try:
        # Use OpenInference convention which Langfuse natively maps
        otel_cfg = nemo_relay.OpenTelemetryConfig("openinference", endpoint)
        otel_cfg.service_name = resolved_service or "genai-tk"
        for k, v in headers.items():
            otel_cfg.set_header(k, v)

        otel_sub = nemo_relay.OpenTelemetrySubscriber(otel_cfg)
        sub_name = f"{_SUBSCRIBER_NAME}-otlp"
        otel_sub.register(sub_name)
        _state.otel_subscriber = otel_sub
        _state.otel_subscriber_name = sub_name
        _state.otel_endpoint = endpoint
        logger.info(f"NeMo Relay OTLP exporter active → {endpoint}")
    except Exception as exc:
        logger.warning(f"NeMo Relay OTLP subscriber setup failed: {exc}")
    finally:
        if sheltered_headers is not None:
            os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = sheltered_headers


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
    if _state.otel_subscriber is not None:
        try:
            _state.otel_subscriber.force_flush()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"NeMo Relay OTLP atexit flush failed: {exc}")
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
    if _state.otel_subscriber is not None:
        try:
            _state.otel_subscriber.force_flush()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"NeMo Relay OTLP sync flush failed: {exc}")
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
    if _state.otel_subscriber is not None:
        try:
            _state.otel_subscriber.force_flush()
        except Exception as exc:  # noqa: BLE101
            logger.debug(f"NeMo Relay OTLP async flush failed: {exc}")
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
        if _state.otel_subscriber is not None:
            try:
                sub_name = _state.otel_subscriber_name or f"{_SUBSCRIBER_NAME}-otlp"
                _state.otel_subscriber.deregister(sub_name)
            except Exception:  # noqa: BLE001
                pass
            _state.otel_subscriber = None
            _state.otel_subscriber_name = None
            _state.otel_endpoint = None

        if _state.store is not None:
            _state.store.close()
        elif _state.file is not None:
            _state.file.close()
        _state.active = False
        _state.store = None
        _state.file = None
        _state.path = None


def is_relay_otlp_active() -> bool:
    """Return True if the NeMo Relay OTLP exporter subscriber is active."""
    return _state.otel_subscriber is not None


def get_relay_otlp_endpoint() -> str | None:
    """Return the active NeMo Relay OTLP exporter endpoint, if configured."""
    return _state.otel_endpoint


def emit_test_probe(
    probe_name: str = "monitoring-test-probe",
    *,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Emit a test agent scope and mark via NeMo Relay and flush all subscribers.

    Useful for CLI connectivity verification (e.g. ``cli monitoring test``).
    Returns True if the probe was emitted and flushed without unhandled errors.
    """
    if not is_nemo_relay_available():
        return False

    import nemo_relay

    # Ensure relay is set up
    setup_nemo_relay()

    try:
        with nemo_relay.scope.scope(
            name=probe_name,
            scope_type=nemo_relay.ScopeType.Agent,
            metadata={"test_probe": True, **(metadata or {})},
        ) as handle:
            nemo_relay.scope.event(
                "probe.ping",
                handle=handle,
                data={"status": "ping", "ts": _now_iso()},
                metadata={"source": "cli.monitoring.test"},
            )

        flush_nemo_relay()
        return True
    except Exception as exc:
        logger.warning(f"NeMo Relay test probe failed: {exc}")
        return False
