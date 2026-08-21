"""Read layer over the local trajectory store (Phase 2/3 foundation).

Parses the ATOF JSONL events written by :mod:`genai_tk.utils.nemo_relay_setup`
into typed :class:`Trajectory` objects and provides the read operations used
by the ``cli trajectory`` command group and by store-based evals:

- :meth:`TrajectoryStore.list_runs` — enumerate runs from ``index.jsonl``.
- :meth:`TrajectoryStore.get` — build a :class:`Trajectory` for one run.
- :meth:`TrajectoryStore.messages` — OpenAI-format message projection
  (consumed by ``agentevals``/``openevals``).
- :meth:`TrajectoryStore.skills` / :meth:`TrajectoryStore.stats` /
  :meth:`TrajectoryStore.diff` / :meth:`TrajectoryStore.prune`.

The store root defaults to ``<data_root>/trajectories`` (matching the write
side). All read methods are best-effort and never raise on malformed events —
they skip unparseable lines with a debug log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


def _default_store_dir() -> Path:
    """Return the default trajectory store root from config."""
    try:
        from genai_tk.config_mgmt.config_mngr import paths_config

        return Path(paths_config().data_root) / "trajectories"
    except Exception:
        return Path("data/trajectories")


# ── Data models ──────────────────────────────────────────────────────────────


@dataclass
class RunSummary:
    """One row of ``cli trajectory list`` / ``index.jsonl``."""

    run_id: str
    profile: str
    started_at: str
    ended_at: str | None
    status: str
    n_llm_calls: int
    n_tool_calls: int
    total_prompt_tokens: int
    total_completion_tokens: int
    tools: list[str]
    skills_loaded: list[str]


@dataclass
class LlmCall:
    """One LLM scope (a model call)."""

    uuid: str
    model: str | None
    usage: dict[str, Any] | None
    tool_calls: list[dict[str, Any]]
    message: str | None
    started_at: str
    ended_at: str | None


@dataclass
class ToolCall:
    """One tool scope (a tool invocation)."""

    uuid: str
    name: str
    args: dict[str, Any]
    result: str | None
    tool_call_id: str | None
    started_at: str
    ended_at: str | None


@dataclass
class SkillLoad:
    """One ``skill.load`` mark."""

    skill_name: str
    source: str | None
    tool_name: str | None
    timestamp: str


@dataclass
class Trajectory:
    """A full run reconstructed from ATOF events."""

    run_id: str
    profile: str
    started_at: str
    ended_at: str | None
    status: str
    events: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: list[LlmCall] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    skill_loads: list[SkillLoad] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    @property
    def tool_names(self) -> list[str]:
        """Tool names called, in order of first occurrence."""
        seen: set[str] = set()
        names: list[str] = []
        for tc in self.tool_calls:
            if tc.name not in seen:
                seen.add(tc.name)
                names.append(tc.name)
        return names

    @property
    def skill_names(self) -> list[str]:
        """Skill names loaded, in order of first occurrence."""
        seen: set[str] = set()
        names: list[str] = []
        for sl in self.skill_loads:
            if sl.skill_name not in seen:
                seen.add(sl.skill_name)
                names.append(sl.skill_name)
        return names


# ── Store ────────────────────────────────────────────────────────────────────


class TrajectoryStore:
    """Read layer over the per-session trajectory store.

    Args:
        root: Store root directory (default ``<data_root>/trajectories``).
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else _default_store_dir()

    # ── listing ──────────────────────────────────────────────────────────────

    def list_runs(
        self,
        *,
        profile: str | None = None,
        since: datetime | None = None,
        status: str | None = None,
    ) -> list[RunSummary]:
        """Return run summaries from ``index.jsonl``, filtered by the given args."""
        summaries = self._read_index()
        out: list[RunSummary] = []
        for s in summaries:
            if profile is not None and s.profile != profile:
                continue
            if status is not None and s.status != status:
                continue
            if since is not None:
                started = _parse_iso(s.started_at)
                if started is None or started < since:
                    continue
            out.append(s)
        # Most recent first.
        out.sort(key=lambda s: s.started_at, reverse=True)
        return out

    def _read_index(self) -> list[RunSummary]:
        """Parse ``index.jsonl`` into summaries (empty if absent)."""
        idx = self.root / "index.jsonl"
        if not idx.exists():
            return []
        out: list[RunSummary] = []
        for line in idx.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                logger.debug("trajectory_store: skipping unparseable index line")
                continue
            out.append(
                RunSummary(
                    run_id=str(d.get("run_id") or ""),
                    profile=str(d.get("profile") or ""),
                    started_at=str(d.get("started_at") or ""),
                    ended_at=d.get("ended_at"),
                    status=str(d.get("status") or "ok"),
                    n_llm_calls=int(d.get("n_llm_calls") or 0),
                    n_tool_calls=int(d.get("n_tool_calls") or 0),
                    total_prompt_tokens=int(d.get("total_prompt_tokens") or 0),
                    total_completion_tokens=int(d.get("total_completion_tokens") or 0),
                    tools=list(d.get("tools") or []),
                    skills_loaded=list(d.get("skills_loaded") or []),
                )
            )
        return out

    # ── single run ───────────────────────────────────────────────────────────

    def get(self, run_id: str) -> Trajectory | None:
        """Build a :class:`Trajectory` for ``run_id`` from its ``events.jsonl``."""
        run_dir = self.root / run_id
        events_path = run_dir / "events.jsonl"
        if not events_path.exists():
            return None
        events = self._read_events(events_path)
        if not events:
            return None
        meta = self._read_meta(run_dir) or {}
        return self._build_trajectory(run_id, events, meta)

    def _read_events(self, path: Path) -> list[dict[str, Any]]:
        """Parse a JSONL events file, skipping unparseable lines."""
        out: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                logger.debug("trajectory_store: skipping unparseable event line")
        return out

    def _read_meta(self, run_dir: Path) -> dict[str, Any] | None:
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    def _build_trajectory(self, run_id: str, events: list[dict[str, Any]], meta: dict[str, Any]) -> Trajectory:
        """Reconstruct llm/tool scopes and skill.load marks from ATOF events."""
        # Index scope start events by uuid so we can pair them with ends.
        starts: dict[str, dict[str, Any]] = {}
        for e in events:
            if e.get("kind") == "scope" and e.get("scope_category") == "start":
                uuid = e.get("uuid")
                if isinstance(uuid, str):
                    starts[uuid] = e

        llm_calls: list[LlmCall] = []
        tool_calls: list[ToolCall] = []
        skill_loads: list[SkillLoad] = []
        prompt_tokens = 0
        completion_tokens = 0
        status = str(meta.get("status") or "ok")

        for e in events:
            kind = e.get("kind")
            cat = e.get("category")
            sc = e.get("scope_category")
            uuid = e.get("uuid")
            ts = str(e.get("timestamp") or "")

            if kind == "scope" and sc == "end" and isinstance(uuid, str):
                start = starts.get(uuid, {})
                if cat == "llm":
                    cp = e.get("category_profile") or {}
                    ar = cp.get("annotated_response") or {}
                    usage = ar.get("usage") if isinstance(ar, dict) else None
                    tcs = ar.get("tool_calls") if isinstance(ar, dict) else None
                    model = ar.get("model") if isinstance(ar, dict) else None
                    msg = ar.get("message") if isinstance(ar, dict) else None
                    if isinstance(usage, dict):
                        prompt_tokens += int(usage.get("prompt_tokens") or 0)
                        completion_tokens += int(usage.get("completion_tokens") or 0)
                    mmeta = e.get("metadata") or {}
                    if isinstance(mmeta, dict) and mmeta.get("otel.status_code") == "ERROR":
                        status = "error"
                    llm_calls.append(
                        LlmCall(
                            uuid=uuid,
                            model=model if isinstance(model, str) else None,
                            usage=usage if isinstance(usage, dict) else None,
                            tool_calls=list(tcs) if isinstance(tcs, list) else [],
                            message=msg if isinstance(msg, str) else None,
                            started_at=str(start.get("timestamp") or ts),
                            ended_at=ts,
                        )
                    )
                elif cat == "tool":
                    # Tool start `data` is the args dict directly (e.g.
                    # {"message": "hello"}). Tool end `data` is a serialized
                    # ToolMessage whose content is nested under data.data.content.
                    edata = e.get("data") or {}
                    nested = edata.get("data") if isinstance(edata, dict) else None
                    if not isinstance(nested, dict):
                        nested = {}
                    result = _stringify(nested.get("content"))
                    tool_call_id = nested.get("tool_call_id")
                    sdata = start.get("data") or {}
                    args = sdata if isinstance(sdata, dict) else {}
                    tool_calls.append(
                        ToolCall(
                            uuid=uuid,
                            name=str(start.get("name") or e.get("name") or ""),
                            args=args,
                            result=result,
                            tool_call_id=tool_call_id if isinstance(tool_call_id, str) else None,
                            started_at=str(start.get("timestamp") or ts),
                            ended_at=ts,
                        )
                    )

            elif kind == "mark" and e.get("name") == "skill.load":
                data = e.get("data") or {}
                mmeta = e.get("metadata") or {}
                skill_loads.append(
                    SkillLoad(
                        skill_name=str(data.get("skill_name") or "") if isinstance(data, dict) else "",
                        source=mmeta.get("skill_load_source") if isinstance(mmeta, dict) else None,
                        tool_name=mmeta.get("tool_name") if isinstance(mmeta, dict) else None,
                        timestamp=ts,
                    )
                )

        # Order llm/tool calls by start time for a sensible timeline.
        llm_calls.sort(key=lambda c: c.started_at)
        tool_calls.sort(key=lambda c: c.started_at)

        return Trajectory(
            run_id=run_id,
            profile=str(meta.get("profile") or ""),
            started_at=str(meta.get("started_at") or ""),
            ended_at=meta.get("ended_at"),
            status=status,
            events=events,
            llm_calls=llm_calls,
            tool_calls=tool_calls,
            skill_loads=skill_loads,
            total_prompt_tokens=prompt_tokens,
            total_completion_tokens=completion_tokens,
        )

    # ── projections ──────────────────────────────────────────────────────────

    def messages(self, run_id: str) -> list[dict[str, Any]]:
        """Project a run to OpenAI-format messages (user/assistant/tool/assistant).

        Used by store-based evals (``agentevals``/``openevals``) and by
        ``cli trajectory show --format messages``. Best-effort over the ATOF
        llm/tool scopes; falls back to ``meta`` when events lack a user message.
        """
        traj = self.get(run_id)
        if traj is None:
            return []

        out: list[dict[str, Any]] = []

        # User message: prefer the root agent scope start input, else meta.
        user_msg = self._root_user_message(traj)
        if user_msg:
            out.append({"role": "user", "content": user_msg})

        # Interleave assistant (llm) and tool messages by start time.
        items: list[tuple[str, str, dict[str, Any]]] = []
        for lc in traj.llm_calls:
            entry: dict[str, Any] = {"role": "assistant", "content": lc.message or ""}
            tcs = _openai_tool_calls(lc.tool_calls)
            if tcs:
                entry["tool_calls"] = tcs
            items.append((lc.started_at, "assistant", entry))
        for tc in traj.tool_calls:
            items.append((tc.started_at, "tool", {"role": "tool", "content": tc.result or ""}))
        items.sort(key=lambda it: it[0])

        # Drop trailing empty assistant entries (no content, no tool_calls).
        for _, role, entry in items:
            if role == "assistant" and not entry.get("content") and not entry.get("tool_calls"):
                continue
            out.append(entry)

        return out

    def _root_user_message(self, traj: Trajectory) -> str | None:
        """Extract the user message from the root agent scope start input.

        The outermost scope is an *implicit* root that emits no start event of
        its own; the first *explicit* agent scope (e.g. named after the
        profile) carries the user ``messages`` input in its ``data``. Several
        agent scope starts may reuse the same uuid (middleware ``before_agent``
        hooks) and also carry a ``messages`` list — prefer the first whose
        ``messages`` is a plain string (the raw user query).
        """
        for e in traj.events:
            if e.get("kind") != "scope" or e.get("scope_category") != "start":
                continue
            if e.get("category") != "agent":
                continue
            data = e.get("data") or {}
            if not isinstance(data, dict):
                continue
            msgs = data.get("messages")
            if isinstance(msgs, str):
                return msgs
        # Fallback: first agent-start whose messages is a list with a user msg.
        for e in traj.events:
            if e.get("kind") != "scope" or e.get("scope_category") != "start":
                continue
            if e.get("category") != "agent":
                continue
            data = e.get("data") or {}
            if not isinstance(data, dict):
                continue
            msgs = data.get("messages")
            if isinstance(msgs, list):
                for m in msgs:
                    if isinstance(m, dict) and m.get("role") == "user":
                        return str(m.get("content") or "")
                for m in msgs:
                    if isinstance(m, dict) and m.get("type") in ("human", "user"):
                        return str(m.get("content") or "")
        return None

    def skills(self, run_id: str) -> list[SkillLoad]:
        """Return the ``skill.load`` marks for a run."""
        traj = self.get(run_id)
        return traj.skill_loads if traj is not None else []

    # ── aggregate stats ───────────────────────────────────────────────────────

    def stats(self, *, since: datetime | None = None) -> dict[str, Any]:
        """Aggregate stats across runs (counts, tokens, cost, tool/skill freq)."""
        runs = self.list_runs(since=since)
        tool_freq: dict[str, int] = {}
        skill_freq: dict[str, int] = {}
        n_failed = 0
        latencies: list[float] = []
        for s in runs:
            if s.status == "error":
                n_failed += 1
            for t in s.tools:
                tool_freq[t] = tool_freq.get(t, 0) + 1
            for sk in s.skills_loaded:
                skill_freq[sk] = skill_freq.get(sk, 0) + 1
            latency = _latency_seconds(s.started_at, s.ended_at)
            if latency is not None:
                latencies.append(latency)
        latencies.sort()
        return {
            "n_runs": len(runs),
            "n_failed": n_failed,
            "total_prompt_tokens": sum(s.total_prompt_tokens for s in runs),
            "total_completion_tokens": sum(s.total_completion_tokens for s in runs),
            "tool_frequency": dict(sorted(tool_freq.items(), key=lambda kv: kv[1], reverse=True)),
            "skill_load_frequency": dict(sorted(skill_freq.items(), key=lambda kv: kv[1], reverse=True)),
            "latency_p50": _percentile(latencies, 0.5),
            "latency_p95": _percentile(latencies, 0.95),
        }

    # ── diff ─────────────────────────────────────────────────────────────────

    def diff(self, run_id_a: str, run_id_b: str) -> dict[str, Any]:
        """Structural diff of two runs (tools, skills, step counts, status)."""
        a = self.get(run_id_a)
        b = self.get(run_id_b)
        if a is None or b is None:
            return {"error": "one or both runs not found", "a": run_id_a, "b": run_id_b}
        return {
            "a": {"run_id": a.run_id, "profile": a.profile, "status": a.status},
            "b": {"run_id": b.run_id, "profile": b.profile, "status": b.status},
            "tools_only_in_a": sorted(set(a.tool_names) - set(b.tool_names)),
            "tools_only_in_b": sorted(set(b.tool_names) - set(a.tool_names)),
            "skills_only_in_a": sorted(set(a.skill_names) - set(b.skill_names)),
            "skills_only_in_b": sorted(set(b.skill_names) - set(a.skill_names)),
            "n_llm_calls": {"a": len(a.llm_calls), "b": len(b.llm_calls)},
            "n_tool_calls": {"a": len(a.tool_calls), "b": len(b.tool_calls)},
            "tokens": {
                "a": {"prompt": a.total_prompt_tokens, "completion": a.total_completion_tokens},
                "b": {"prompt": b.total_prompt_tokens, "completion": b.total_completion_tokens},
            },
        }

    # ── retention ────────────────────────────────────────────────────────────

    def prune(
        self,
        *,
        keep_last: int | None = None,
        older_than_days: int | None = None,
    ) -> int:
        """Delete run dirs not matching the retention policy. Returns count removed."""
        runs = self.list_runs()
        to_keep: set[str] = set()
        if keep_last is not None:
            for s in runs[:keep_last]:
                to_keep.add(s.run_id)
        cutoff: datetime | None = None
        if older_than_days is not None:
            cutoff = datetime.now(tz=timezone.utc).timestamp() - older_than_days * 86400
        removed = 0
        for s in runs:
            if s.run_id in to_keep:
                continue
            if cutoff is not None:
                started = _parse_iso(s.started_at)
                if started is not None and started.timestamp() > cutoff:
                    continue
            run_dir = self.root / s.run_id
            if run_dir.exists():
                _rm_tree(run_dir)
                removed += 1
        # Rebuild index.jsonl without removed runs.
        self._rebuild_index(
            {s.run_id for s in runs} - {s.run_id for s in runs if s.run_id in to_keep or (cutoff is not None)}
        )
        return removed

    def _rebuild_index(self, removed: set[str]) -> None:
        """Rewrite ``index.jsonl`` excluding ``removed`` run ids."""
        idx = self.root / "index.jsonl"
        if not idx.exists():
            return
        kept: list[str] = []
        for line in idx.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            rid = str(d.get("run_id") or "")
            if rid not in removed:
                kept.append(line)
        idx.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _openai_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert annotated tool_calls to OpenAI ``tool_calls`` shape."""
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        name = str(tc.get("name") or "")
        args = tc.get("arguments")
        out.append(
            {
                "id": str(tc.get("id") or ""),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args) if not isinstance(args, str) else args,
                },
            }
        )
    return out


def _stringify(value: Any) -> str | None:
    """Best-effort stringify of a tool result value."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except Exception:  # noqa: BLE101
        return str(value)


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE101
        return None


def _latency_seconds(started_at: str | None, ended_at: str | None) -> float | None:
    s = _parse_iso(started_at)
    e = _parse_iso(ended_at)
    if s is None or e is None:
        return None
    return (e - s).total_seconds()


def _percentile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * q
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _rm_tree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
