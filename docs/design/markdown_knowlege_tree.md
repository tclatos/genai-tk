# Markdown Knowledge Tree (`mdktree`) — Design Brief

## Goal

Build a Python library that ingests a **corpus of Markdown files** into a **graph database (Kuzu)**, extracting each document's heading hierarchy as a tree of `Section` nodes. On top of this graph, run **vectorless agentic RAG**: an LLM agent navigates the corpus by walking tree structures and reading section text — no embeddings, no vector search.

The agent can:
- List all documents in the corpus.
- Inspect any document's table of contents (TOC).
- Fetch the raw text of specific sections by ID or line range.
- Search for sections by keyword across the whole corpus.
- Follow markdown links between sections (including cross-document links).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                User / OpenAI Agents SDK                       │
└────────────────────────────┬─────────────────────────────────┘
                             │ tool calls
┌────────────────────────────┴─────────────────────────────────┐
│ MarkdownKnowledgeTreeClient                                   │
│                                                               │
│  ┌────────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │  Ingestor  │   │  Retriever   │   │   Agent Runner     │  │
│  └─────┬──────┘   └──────┬───────┘   └─────────┬──────────┘  │
│        │                 │                      │             │
│        ▼                 │                      │             │
│  ┌──────────────┐        │                      │             │
│  │  mdparser    │        │                      │             │
│  │ (markdown-   │        │                      │             │
│  │  it-py)      │        │                      │             │
│  └──────┬───────┘        │                      │             │
│         │                ▼                      ▼             │
│         │     ┌──────────────────────────────────────┐       │
│         └────►│           KuzuBackend                │       │
│               │  Document, Section, Entity, Tag      │       │
│               │  + HasChildSection, LinksTo, etc.   │       │
│               └──────────────────────────────────────┘       │
└───────────────────────────────────────────────────────────────┘
```

---

## Dependencies

```
kuzu>=0.4.0
markdown-it-py>=3.0.0
mdit-py-plugins>=0.4.0        # wikilink support (optional)
tiktoken>=0.7.0               # token counting
openai>=1.40.0                # LLM calls
openai-agents>=0.0.10         # @function_tool, Agent, Runner
```

---

## Kuzu Schema

### Node Tables

```cypher
CREATE NODE TABLE Document (
    doc_id          STRING PRIMARY KEY,   -- deterministic: sha256(source_path)[:16]
    doc_name        STRING,
    doc_description STRING,               -- LLM-generated 1-paragraph summary
    source_path     STRING,
    file_hash       STRING,               -- sha256(file_content); for dedup
    line_count      INT64,
    ingested_at     TIMESTAMP,
    tags            STRING[]
);

CREATE NODE TABLE Section (
    section_id      STRING PRIMARY KEY,   -- "{doc_id}::{line_num}" e.g. "a1b2c3::42"
    doc_id          STRING,
    title           STRING,
    level           INT16,                -- 1..6 from ATX/Setext heading
    line_num        INT64,                -- 1-indexed source line
    end_line        INT64,                -- last line before next sibling/ancestor heading
    text            STRING,               -- raw markdown slice (heading line + body)
    summary         STRING,               -- LLM summary (leaf nodes)
    prefix_summary  STRING,               -- LLM summary (internal nodes)
    token_count     INT32
);

CREATE NODE TABLE Entity (
    entity_id   STRING PRIMARY KEY,
    name        STRING,
    type        STRING                    -- person/org/concept/...
);

CREATE NODE TABLE Tag (
    name        STRING PRIMARY KEY,
    description STRING
);
```

### Relationship Tables

```cypher
CREATE REL TABLE HasRootSection  (FROM Document TO Section);
CREATE REL TABLE HasChildSection (FROM Section   TO Section);
CREATE REL TABLE LinksTo         (FROM Section   TO Section,
                                  link_text STRING,
                                  link_type STRING);   -- "inline" | "reference" | "wikilink" | "external"
CREATE REL TABLE Mentions        (FROM Section   TO Entity);
CREATE REL TABLE Tagged          (FROM Document  TO Tag);
```

### Key design decisions

- **No `node_id` field.** The `section_id` (`"{doc_id}::{line_num}"`) is the sole identifier. Document order comes from `line_num`, tree structure from `HasChildSection` edges.
- **Deterministic `doc_id`:** `sha256(absolute_source_path)[:16]` — stable across re-ingests as long as the file path doesn't change. This keeps `section_id` and `LinksTo` edges stable when content is edited but headings don't move.
- **`file_hash` for dedup:** if the file content hash hasn't changed, skip re-ingestion entirely.

---

## Markdown Parsing (`mdparser.py`)

Use `markdown-it-py` with the CommonMark preset + GFM tables + strikethrough. No regex for heading detection.

### Parser logic

1. **Initialize parser:**
   ```python
   md = MarkdownIt("commonmark", {"html": True, "linkify": True}) \
       .enable("table") \
       .enable("strikethrough")
   ```

2. **Parse with line tracking:**
   ```python
   tokens = md.parse(raw, env={"record_lines": True})
   ```
   Every block token now has `tok.map = [start_line, end_line]` (0-indexed, end-exclusive).

3. **Collect top-level headings only** (depth == 0 in the token stream — not inside blockquotes/lists):
   - For each `heading_open` token at nesting depth 0, record:
     - `level` from `tok.tag` (e.g. `"h2"` → `2`)
     - `title` from the following `inline` token's `.content`
     - `line_num` from `tok.map[0] + 1` (convert to 1-indexed)

4. **Compute `end_line` for each section:**
   - For section *k* with level *L*, `end_line` = `line_num` of the next section whose level ≤ *L*, minus 1. If no such section, `end_line` = last line of file.

5. **Extract section text:**
   - `text = "\n".join(raw_lines[line_num - 1 : end_line])`

6. **Extract links:**
   - Walk all tokens (including inline children) for `link_open` tokens.
   - `href` = `tok.attrs["href"]`
   - `link_text` = content of sibling tokens until `link_close`
   - Record `src_section_line_num` by checking which section's `[line_num, end_line]` range contains the link's line.
   - Optionally: enable `mdit_py_plugins.wikilinks` for `[[target|text]]` syntax.

### Output

```python
@dataclass
class FlatNode:
    title:    str
    level:    int
    line_num: int
    end_line: int
    text:     str

@dataclass
class LinkRecord:
    src_section_line: int
    href:             str
    link_text:        str
    link_type:        str    # "inline" | "reference" | "wikilink" | "external"

def parse_markdown(raw: str) -> tuple[list[FlatNode], list[LinkRecord]]:
    ...
```

---

## Tree Builder (`treebuilder.py`)

Stack-based nesting — a node becomes a child of the last node on the stack whose level is strictly smaller.

```python
def build_tree(flat: list[FlatNode]) -> list[FlatNode]:
    roots: list[FlatNode] = []
    stack: list[FlatNode] = []
    for node in flat:
        while stack and stack[-1].level >= node.level:
            stack.pop()
        if not stack:
            roots.append(node)
        else:
            stack[-1].children.append(node)
        stack.append(node)
    return roots
```

After building, recursively assign `section_id` to each node:
```python
section_id = f"{doc_id}::{node.line_num}"
```

---

## Ingestion Pipeline (`ingestor.py`)

```python
async def ingest_file(client, md_path: str) -> str:
    raw = Path(md_path).read_text(encoding="utf-8")
    file_hash = hashlib.sha256(raw.encode()).hexdigest()

    # 1. Dedup check
    existing = client.storage.get_doc_by_path(md_path)
    if existing and existing["file_hash"] == file_hash:
        return existing["doc_id"]

    # 2. Deterministic doc_id
    abs_path = os.path.abspath(md_path)
    doc_id = hashlib.sha256(abs_path.encode()).hexdigest()[:16]

    # 3. Parse
    flat_nodes, links = parse_markdown(raw)
    tree = build_tree(flat_nodes)
    add_token_counts(tree)  # via tiktoken

    # 4. LLM summaries (optional)
    if client.config.add_summaries:
        tree = await generate_summaries(tree, client.llm)
    doc_description = (
        await generate_doc_description(tree, client.llm)
        if client.config.add_doc_description else ""
    )

    # 5. Resolve cross-doc links
    resolved_links = resolve_link_targets(links, client.storage, abs_path)

    # 6. Persist to Kuzu
    client.storage.delete_document(doc_id)  # clean slate if re-ingesting
    client.storage.upsert_document(doc_id, doc_name, doc_description,
                                   abs_path, file_hash, raw.count("\n") + 1,
                                   client.config.default_tags)
    client.storage.upsert_tree(doc_id, tree)
    client.storage.upsert_links(resolved_links)
    return doc_id


def ingest_directory(client, dir_: str, glob: str = "**/*.md") -> list[str]:
    doc_ids = []
    for path in Path(dir_).glob(glob):
        doc_ids.append(asyncio.run(ingest_file(client, str(path))))
    return doc_ids
```

### Cross-doc link resolution

After all docs are ingested, resolve each `LinkRecord.href`:

| `href` format | Resolution |
|---|---|
| `#anchor` | Same-doc: find section whose slugified `title` matches `anchor` |
| `./other.md` or `other.md` | Resolve relative to source dir → look up `Document` by `source_path` |
| `./other.md#section` | Combine both above |
| `https://…` | `link_type = "external"`; no `LinksTo` edge (store in separate `ExternalLink` table or skip) |

---

## Retrieval Tools (`retriever.py`)

Six tools exposed to the agent:

### 1. `list_documents()`

```cypher
MATCH (d:Document)
OPTIONAL MATCH (d)-[:Tagged]->(t:Tag)
RETURN d.doc_id, d.doc_name, d.doc_description, d.line_count,
       collect(t.name) AS tags
ORDER BY d.doc_name;
```

Returns JSON array of all documents with metadata.

### 2. `get_document(doc_id)`

```cypher
MATCH (d:Document {doc_id: $doc_id})
RETURN d.*;
```

Returns metadata for a single document.

### 3. `get_document_structure(doc_id)`

```cypher
MATCH (d:Document {doc_id: $doc_id})-[:HasRootSection]->(root:Section)
MATCH path = (root)-[:HasChildSection*0..]->(s:Section)
RETURN s.section_id, s.title, s.level, s.line_num, s.prefix_summary
ORDER BY s.line_num;
```

Returns the document's TOC tree (titles + IDs + prefix summaries, no full text). This is the agent's map for deciding which sections to read.

### 4. `get_section_content(section_ids)`

Accepts comma-separated `section_id` values.

```cypher
MATCH (s:Section)
WHERE s.section_id IN split($section_ids, ',')
RETURN s.section_id, s.title, s.text, s.line_num, s.end_line;
```

Returns the raw markdown text of the requested sections.

### 5. `find_sections_by_keyword(keyword, limit=20)`

```cypher
MATCH (s:Section)
WHERE s.title CONTAINS $keyword OR s.text CONTAINS $keyword
RETURN s.doc_id, s.section_id, s.title, s.level, s.line_num
ORDER BY s.line_num
LIMIT $limit;
```

Cross-document keyword search. Pure string matching — no embeddings.

### 6. `get_linked_sections(section_id)`

```cypher
MATCH (s1:Section {section_id: $section_id})-[l:LinksTo]->(s2:Section)
RETURN s2.doc_id, s2.section_id, s2.title, l.link_text, l.link_type;
```

Follows markdown links from a section to related sections (potentially in other documents).

---

## Agent Runner (`agent.py`)

### System prompt

```text
You are MarkdownKnowledgeTree, a corpus QA assistant.

WORKFLOW:
1. Call list_documents() to learn which documents exist and what each covers.
2. Pick 1–3 candidate documents based on doc_description and tags.
3. For each candidate, call get_document_structure(doc_id) to see the TOC.
4. Reason about which subtrees likely contain the answer. If a section title
   is ambiguous, use find_sections_by_keyword(keyword) to locate candidate
   sections across the whole corpus.
5. Call get_section_content(section_ids="...") with tight ranges — never fetch
   whole documents.
6. If a section references another section via markdown link, call
   get_linked_sections(section_id) to follow it; the target may be in a
   different document.
7. After gathering evidence, synthesize a concise answer. Cite sources using
   [doc_id::line_num] format.

RULES:
- Never guess. If the tools don't return the answer, say so.
- Always cite source section_ids in the form [doc_id::line_num].
- Prefer narrow section fetches over broad ones.
```

### Agent setup

```python
from agents import Agent, Runner, function_tool

def create_agent(client: MarkdownKnowledgeTreeClient) -> Agent:
    @function_tool
    def list_documents() -> str:
        """List all documents in the corpus."""
        return client.storage.list_documents_json()

    @function_tool
    def get_document(doc_id: str) -> str:
        """Get metadata for a single document."""
        return client.storage.get_document_json(doc_id)

    @function_tool
    def get_document_structure(doc_id: str) -> str:
        """Get the TOC tree of a document (titles + IDs, no full text)."""
        return client.storage.get_document_structure_json(doc_id)

    @function_tool
    def get_section_content(section_ids: str) -> str:
        """Fetch raw text of sections. Comma-separated section_ids."""
        return client.storage.get_section_content_json(section_ids)

    @function_tool
    def find_sections_by_keyword(keyword: str, limit: int = 20) -> str:
        """Cross-document keyword search over section titles and text."""
        return client.storage.find_sections_by_keyword_json(keyword, limit)

    @function_tool
    def get_linked_sections(section_id: str) -> str:
        """Follow markdown links from a section to related sections."""
        return client.storage.get_linked_sections_json(section_id)

    return Agent(
        name="MarkdownKnowledgeTree",
        instructions=SYSTEM_PROMPT,
        model="gpt-4o-2024-11-20",
        tools=[
            list_documents,
            get_document,
            get_document_structure,
            get_section_content,
            find_sections_by_keyword,
            get_linked_sections,
        ],
    )


def query(client: MarkdownKnowledgeTreeClient, question: str) -> str:
    agent = create_agent(client)
    result = Runner.run_sync(agent, question)
    return result.final_output
```

---

## Client API (`client.py`)

```python
class MarkdownKnowledgeTreeClient:
    def __init__(self,
                 workspace: str = "data/mdktree",
                 llm_model: str = "gpt-4o-mini",
                 retrieve_model: str = "gpt-4o-2024-11-20",
                 config: Config | None = None):
        self.workspace = Path(workspace).expanduser()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.llm = llm_model
        self.retrieve_model = retrieve_model
        self.config = config or default_config()
        self.storage = KuzuBackend(self.workspace / "store.kuzu")

    # ── Ingestion ──────────────────────────────────
    def ingest_file(self, path: str) -> str: ...
    def ingest_directory(self, dir_: str, glob: str = "**/*.md") -> list[str]: ...
    def reindex_changed(self, dir_: str) -> list[str]: ...

    # ── Retrieval ──────────────────────────────────
    def list_documents(self) -> str: ...
    def get_document(self, doc_id: str) -> str: ...
    def get_document_structure(self, doc_id: str) -> str: ...
    def get_section_content(self, section_ids: str) -> str: ...
    def find_sections_by_keyword(self, keyword: str, limit: int = 20) -> str: ...
    def get_linked_sections(self, section_id: str) -> str: ...

    # ── Agent ──────────────────────────────────────
    def query(self, question: str) -> str: ...
```

---

## Example Usage (`examples/agentic_vectorless_rag_demo.py`)

```python
"""
Agentic Vectorless RAG Demo
============================
Ingests a directory of Markdown files into a Kuzu graph store,
then answers questions using an LLM agent that walks the tree
structure — no vector embeddings required.
"""
from mdktree import MarkdownKnowledgeTreeClient

if __name__ == "__main__":
    # 1. Initialize client
    client = MarkdownKnowledgeTreeClient(
        workspace="data/mdktree",
        llm_model="gpt-4o-mini",
        retrieve_model="gpt-4o-2024-11-20",
    )

    # 2. Ingest a corpus of markdown files
    print("Ingesting corpus …")
    doc_ids = client.ingest_directory("docs/", glob="**/*.md")
    print(f"  {len(doc_ids)} documents indexed.")

    # 3. Show what's in the corpus
    print("\nCorpus overview:")
    print(client.list_documents())

    # 4. Ask a question — agent walks the trees autonomously
    question = "Compare the optimization choices used in model_a and model_b."
    print(f"\nQuestion: {question}\n{'=' * 60}")
    answer = client.query(question)
    print(f"\nAnswer:\n{answer}")
```

### Expected agent trace

For the question above, the agent would:

1. **`list_documents()`** → sees `model_a.md` and `model_b.md`, both tagged `ml/paper`.
2. **`get_document_structure("model_a_hash")`** → sees TOC including `Training > Optimizer (line 45)`.
3. **`get_document_structure("model_b_hash")`** → sees TOC including `Setup > Optimizer (line 38)`.
4. **`get_section_content("model_a_hash::45,model_b_hash::38")`** → reads both optimizer sections.
5. **`get_linked_sections("model_a_hash::45")`** → discovers a link to `model_b_hash::38` (cross-doc markdown link `[compared to](model_b.md#optimizer)`).
6. Synthesizes answer with citations `[model_a_hash::45]` and `[model_b_hash::38]`.

---

## Project Layout

```
markdown-knowledge-tree/
├── pyproject.toml
├── README.md
├── mdktree/
│   ├── __init__.py
│   ├── schema.py              # Kuzu DDL strings
│   ├── mdparser.py            # markdown-it-py → FlatNode + LinkRecord
│   ├── treebuilder.py         # flat list → nested tree
│   ├── summarizer.py          # LLM summaries for nodes + docs
│   ├── ingestor.py            # end-to-end ingest pipeline
│   ├── retriever.py           # 6 tool implementations
│   ├── client.py              # high-level API
│   ├── agent.py               # OpenAI Agents SDK integration
│   ├── config.py              # Config dataclass + defaults
│   └── storage/
│       ├── __init__.py
│       ├── base.py            # StorageBackend Protocol
│       └── kuzu_backend.py    # Kuzu implementation
├── examples/
│   └── agentic_vectorless_rag_demo.py
└── tests/
    ├── test_mdparser.py
    ├── test_treebuilder.py
    ├── test_kuzu_backend.py
    └── test_ingestor.py
```

---

## Summary of Key Design Choices

| Decision | Choice | Rationale |
|---|---|---|
| Markdown parser | `markdown-it-py` (CommonMark + GFM) | Correct heading detection (Setext, code-fence safe), `tok.map` for line numbers, native link extraction |
| Storage | Kuzu (embedded graph DB) | Native graph traversal for tree walking + cross-doc links; Cypher expressiveness; no server needed |
| Section ID | `"{doc_id}::{line_num}"` | Deterministic, stable across re-ingests (if headings don't move), no separate counter field needed |
| Document ID | `sha256(source_path)[:16]` | Deterministic — stable across re-ingests, preserves all edges |
| Dedup | `sha256(file_content)` | Skip re-parsing + re-summarizing if file unchanged |
| Tree structure | `HasChildSection` edges | Graph-native; no need to reconstruct from flat rows |
| Document order | `line_num` field | From parser's `tok.map`; more precise than DFS counter |
| Retrieval | 6 agent tools | `list_documents` + `get_document` + `get_document_structure` + `get_section_content` + `find_sections_by_keyword` + `get_linked_sections` |
| Vectorless | No embeddings | LLM walks the TOC tree and decides which sections to read — same cost profile, simpler infra |