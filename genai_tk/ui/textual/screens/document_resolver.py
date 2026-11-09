"""Generic resolver and launcher for Document and MarkdownSection pop-up modals."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from loguru import logger
from textual.app import App

from genai_tk.ui.textual.screens.document_modal import DocumentViewerModal


def _find_default_kg_db(opportunity: str | None = None) -> str | None:
    """Find candidate Kùzu / Ladybug database paths for the given opportunity or workspace."""
    candidates: list[Path] = []
    if opportunity:
        candidates.extend(
            [
                Path(f"data/kg/{opportunity}_tree.db"),
                Path(f"data/kg/{opportunity}.db"),
            ]
        )
    candidates.extend(
        [
            Path("data/kg/rfq_tree.db"),
            Path("data/kg/document_graph.db"),
            Path("data/kg/val_clair_tree.db"),
            Path("data/kg/val-clair_tree.db"),
        ]
    )

    try:
        from genai_graph.agent import resolve_db_path

        p = resolve_db_path()
        if p and Path(p).exists():
            return str(Path(p).resolve())
    except Exception:
        pass

    for c in candidates:
        if (
            c.exists()
            and (c / "catalog.kz").exists()
            or (c / "metadata.kz").exists()
            or (c.is_dir() and any(c.iterdir()))
        ):
            return str(c.resolve())
    return None


def resolve_document_or_section(
    target: str,
    db_path: str | None = None,
    opportunity: str | None = None,
) -> tuple[str, str, str | None]:
    """Resolve a document or section identifier/URI to (title, content, subtitle).

    Args:
        target: Target identifier or URI (e.g. 'section://file_...::13', 'doc://Alko_RFP.pdf.md',
            'file_20d0d01e51d9c515::13', 'Appendix 3.docx', or path).
        db_path: Path to the Kùzu graph database if known.
        opportunity: Opportunity slug to help resolve default databases or folders.

    Returns:
        Tuple of (title, markdown_content, subtitle).
    """
    cleaned = target.strip()
    kind: str = "auto"

    if cleaned.startswith(("section://", "sec://")):
        kind = "section"
        cleaned = re.sub(r"^(section|sec)://", "", cleaned)
    elif cleaned.startswith(("doc://", "document://")):
        kind = "document"
        cleaned = re.sub(r"^(doc|document)://", "", cleaned)
    elif cleaned.startswith("file://"):
        kind = "file"
        cleaned = re.sub(r"^file://", "", cleaned)

    cleaned = unquote(cleaned)

    # If format is "DocName · SectionName"
    sec_hint = None
    if " · " in cleaned:
        parts = cleaned.split(" · ", 1)
        cleaned = parts[0].strip()
        sec_hint = parts[1].strip()

    # Determine kind heuristic if auto
    if kind == "auto":
        if "::" in cleaned:
            kind = "section"
        elif any(cleaned.lower().endswith(ext) for ext in (".md", ".pdf", ".docx", ".xlsx", ".pptx", ".txt")):
            kind = "document"

    # Attempt Kùzu / Ladybug resolution
    actual_db = db_path or _find_default_kg_db(opportunity)
    if actual_db and Path(actual_db).exists():
        try:
            from genai_graph.kg.backend import KuzuBackend
            from genai_graph.kg.query.document_graph_tools import (
                get_document,
                get_section_content,
                reconstruct_document,
                reconstruct_section,
            )

            backend = KuzuBackend()
            backend.connect(actual_db, read_only=True)

            if kind in ("section", "auto"):
                # Check for explicit section ID pattern file_<hash>::<seq> anywhere in cleaned
                m_sec = re.search(r"(file_[a-f0-9]+::\d+)", cleaned)
                target_sec_id = m_sec.group(1) if m_sec else cleaned

                sec_rows = get_section_content(backend, [target_sec_id])
                if sec_rows:
                    sec_row = sec_rows[0]
                    sec_title = sec_row.get("title") or target_sec_id
                    sec_text = sec_row.get("text") or ""
                    if not sec_text:
                        sec_text = reconstruct_section(backend, target_sec_id) or ""

                    doc_fn = None
                    ch = sec_row.get("content_hash")
                    if ch:
                        doc_meta = get_document(backend, ch)
                        if doc_meta:
                            doc_fn = doc_meta.get("filename")

                    doc_prefix = f"Document: {doc_fn} | " if doc_fn else ""
                    lines_info = (
                        f" (Lines {sec_row.get('line_start')}–{sec_row.get('line_end')})"
                        if sec_row.get("line_start")
                        else ""
                    )
                    sub = f"{doc_prefix}Section ID: {target_sec_id}{lines_info} | Database: {Path(actual_db).name}"
                    return f"Section: {sec_title}", sec_text, sub

                # Try direct section reconstruction or query
                text = reconstruct_section(backend, target_sec_id)
                if text:
                    return (
                        f"Section: {target_sec_id}",
                        text,
                        f"Section ID: {target_sec_id} | Database: {Path(actual_db).name}",
                    )

                # Try finding section by title if not found by ID
                if sec_hint or kind == "section":
                    search_title = sec_hint or cleaned
                    try:
                        df = backend.execute_get_as_df(
                            "MATCH (s:MarkdownSection) WHERE lower(s.title) CONTAINS lower($title) "
                            "RETURN s.title AS title, s.text AS text, s.section_id AS sec_id LIMIT 1",
                            {"title": search_title},
                            union=False,
                        )
                        if not df.empty:
                            row = df.iloc[0]
                            return (
                                str(row["title"]),
                                str(row["text"]),
                                f"Section ID: {row['sec_id']} | Matched by title",
                            )
                    except Exception:
                        pass

            if kind in ("document", "auto"):
                doc_text = reconstruct_document(backend, cleaned)
                doc_meta = get_document(backend, cleaned)
                if not doc_text:
                    clean_name = Path(cleaned).name
                    try:
                        df_doc = backend.execute_get_as_df(
                            "MATCH (d:Document) WHERE lower(d.filename) CONTAINS lower($fn) "
                            "RETURN d.filename AS filename, d.content_hash AS content_hash, d.section_count AS section_count LIMIT 1",
                            {"fn": clean_name},
                            union=False,
                        )
                        if not df_doc.empty:
                            row_d = df_doc.iloc[0]
                            doc_text = reconstruct_document(backend, str(row_d["content_hash"]))
                            doc_meta = dict(row_d)
                    except Exception:
                        pass

                if doc_text:
                    fn = (doc_meta.get("filename") if doc_meta else None) or cleaned
                    ch = (doc_meta.get("content_hash") if doc_meta else None) or "—"
                    sc = doc_meta.get("section_count", "—") if doc_meta else "—"
                    return (
                        f"Document: {fn}",
                        doc_text,
                        f"Content Hash: {ch} | Sections: {sc} | Database: {Path(actual_db).name}",
                    )
        except Exception as exc:
            logger.debug("Graph document resolution error: {}", exc)

    # Attempt local filesystem resolution
    candidate_paths = [
        Path(cleaned),
        Path("data") / cleaned,
        Path("data/uploaded_rfq") / cleaned,
    ]
    if opportunity:
        candidate_paths.extend(
            [
                Path(f"data/uploaded_rfq/{opportunity}") / cleaned,
                Path(f"data/uploaded_rfq/{opportunity.capitalize()}") / cleaned,
            ]
        )

    for p in candidate_paths:
        if p.exists() and p.is_file():
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                return p.name, content, f"Source File: {p.resolve()}"
            except Exception as exc:
                logger.debug("Failed reading local file {}: {}", p, exc)

    # Search markdown files by filename match in data
    try:
        data_dir = Path("data")
        if data_dir.exists():
            stem = Path(cleaned).stem.lower()
            for found in data_dir.rglob("*.md"):
                if stem in found.name.lower():
                    try:
                        content = found.read_text(encoding="utf-8", errors="replace")
                        return found.name, content, f"Matched file: {found.resolve()}"
                    except Exception:
                        pass
    except Exception:
        pass

    # Fallback message
    title = cleaned
    content = f"""### 📄 Referenced Resource: `{cleaned}`

No pre-rendered Markdown was found in the graph database or local storage for this document or section.

- **Target Identifier:** `{cleaned}`
- **Opportunity Context:** `{opportunity or "Default"}`
- **Database Inspected:** `{Path(actual_db).name if actual_db else "None"}`
"""
    if sec_hint:
        content += f"- **Referenced Section:** `{sec_hint}`\n"

    return title, content, "Resource Reference (Unresolved)"


def open_document_modal(
    app: App[Any],
    target: str,
    db_path: str | None = None,
    opportunity: str | None = None,
    fallback_content: str | None = None,
    title: str | None = None,
) -> None:
    """Resolve document/section target and push a DocumentViewerModal on the app screen stack."""
    resolved_title, resolved_content, subtitle = resolve_document_or_section(
        target,
        db_path=db_path,
        opportunity=opportunity,
    )
    final_content = (
        fallback_content if (fallback_content and "No pre-rendered Markdown" in resolved_content) else resolved_content
    )
    modal = DocumentViewerModal(
        title=title or resolved_title,
        content=final_content,
        subtitle=subtitle,
    )
    app.push_screen(modal)
