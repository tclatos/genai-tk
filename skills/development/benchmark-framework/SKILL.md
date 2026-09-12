---
name: benchmark-framework
description: Build, configure, and execute multi-dataset benchmarks (cli bench) with custom adapters, document-to-knowledge-graph pipelines, multimodal image understanding, agent harness evaluation, LLM-as-judge scoring, and TUI inspection.
tags: [benchmark, evaluation, cli, llm-as-judge, adapter, docgraph, datasets, multimodal]
version: "1.1"
author: ""
category: development
---

# Benchmark Framework Guide (`cli bench`)

The Unified Benchmark Framework (`genai_graph.bench`) provides a standardized, dataset-agnostic pipeline for evaluating autonomous agents and Document Graph retrieval systems across diverse datasets (e.g., **FinanceBench**, **OfficeQA**, **MMLongBench-Doc**).

## When to Use

- Implementing or scaffolding a new benchmark evaluation suite (e.g. SEC filings, legal contracts, technical manuals, multimodal survey reports).
- Adding or configuring benchmark profiles in `config/bench.yaml`.
- Running end-to-end evaluation runs, re-grading existing agent trajectories, or inspecting reports via CLI or TUI.
- Authoring custom `BaseBenchmarkAdapter` classes with domain-specific dataset loaders, document fetchers, and judge rubrics.
- Debugging evaluation failures, numeric equivalence issues, OCR gaps, visual chart understanding, or trajectory errors.

---

## CLI Commands Reference

Benchmark commands are registered via `genai_graph.core.commands_bench.BenchCommands`:

```bash
# 1. List configured benchmark profiles
uv run cli bench list

# 2. Run benchmark evaluation
uv run cli bench run -p mistral_glm
uv run cli bench run -p mistral_glm -q docqa_0207,docqa_0684           # run specific questions
uv run cli bench run -p mistral_glm -d doc_name_1,doc_name_2          # filter by document name
uv run cli bench run -p mistral_glm --skip fetch --skip build         # skip fetch/build when graph exists
uv run cli bench run -p mistral_glm --rerun                           # force re-running agent turns
uv run cli bench run -p mistral_glm --no-judge                        # run agent only without grading
uv run cli bench run -p mistral_glm -n 5                              # limit to first N questions

# 3. Re-grade existing runs without re-running agent
uv run cli bench grade -p mistral_glm
uv run cli bench grade -p mistral_glm --force                         # force re-grade all questions

# 4. Download dataset and documents
uv run cli bench download --docs -n 10                                # download questions + 10 PDFs
uv run cli bench download --no-docs                                   # download questions only

# 5. Generate aggregated metric summary report
uv run cli bench report -p mistral_glm
uv run cli bench report -p mistral_glm --json                         # export summary JSON

# 6. Question and trajectory inspection
uv run cli bench questions -p mistral_glm                             # tabular view
uv run cli bench questions -p mistral_glm -t                          # interactive Textual TUI
uv run cli bench questions -p mistral_glm -t --trajectory             # TUI with live trajectory replay
```

---

## 6-Stage Pipeline Architecture

```
[1. Fetch Dataset & Documents]
         │
         ▼
[2. Markdownize & OCR Ladder] (Mistral OCR with base64 images -> Docling -> MarkItDown)
         │
         ▼
[3. Hierarchical Graph Ingestion] (Ladybug DB + Sections + Image nodes + BM25 & FTS)
         │
         ▼
[4. Autonomous Agent Execution] (DeepAgent / React harness + Document Graph + VLM tools)
         │
         ▼
[5. LLM-as-Judge Evaluation] (Domain-specific rubric + Mafin 2.5 numeric equivalence + groundedness)
         │
         ▼
[6. Metrics & Summary Aggregation] (Accuracy, numeric match, groundedness, error breakdown)
```

1. **Fetch**: Retrieves questions from HuggingFace, CSV, or local files; caches raw PDFs.
2. **Markdownize**: Converts documents using profile-configured strategy (`fast`, `medium`, `best`). For multimodal decks and papers, `MistralOCRConverter` extracts images with xxHash32 hex names into `images_dir` and inserts markdown comments `<--embed-workers and now creates the FTS index by Image: ... -->`.
3. **Build Graph**: Parses markdown into `Folder ──CONTAINS──▶ Document ──HAS_SECTION──▶ MarkdownSection ──HAS_IMAGE──▶ Image` inside Ladybug DB.
4. **Run Agent**: Dispatches questions to the agent harness with graph tools (`get_folder_toc`, `get_document_toc`, `get_section_content`, `search_sections`, `search_images`, `query_image`). Records tool calls, thinking traces, and token usage into `runs.jsonl`.
5. **Judge**: Evaluates outputs against `gold_answer` and `evidence` using domain rubrics, outputting `scores.jsonl`.
6. **Aggregate**: Generates `scores_summary.json` with accuracy tiers, numeric match rates, groundedness rates, and failure taxonomy.

---

## Multimodal & Image Understanding in Document Graph

For document understanding benchmarks requiring visual chart, diagram, or line plot interpretation:

1. **OCR Configuration (`config/markdownize.yaml`)**:
   Enable `include_image_base64: true` so the OCR converter extracts images to disk and generates metadata annotations.
2. **Schema & Node Ingestion**:
   `DocumentGraphFactory` creates `Image` nodes linked from parent `MarkdownSection` via `HAS_IMAGE`:
   - `image_id`: Unique identifier (`{section_id}::{image_hash}`)
   - `path`: Local image path (e.g. `data/markdown_multi/images/442947f8.jpeg`)
   - `description`: Extracted caption / figure title
   - `size`: Byte length of image file
3. **Navigation Tools**:
   - `search_images(query, document_id, section_id, limit)`: Locate relevant figures by caption, title, or filename.
   - `query_image(image, question, model)`: Dispatch visual question answering directly to a Vision-Language Model (e.g. `glm_5.3_flash@openrouter`) passing base64 image data.

---

## Agent Harnessing & Dependency Prerequisites

When using `type: deep` agents with sandboxing and filesystem capabilities (e.g. DeepAgents SDK):
- Ensure the `harnessing` extra dependencies are installed:
  ```bash
  uv sync --extra harnessing
  ```
- This provides `deepagents`, `deerflow-harness`, and `agent-sandbox` needed for deep agent planning and execution.

---

## Creating a Custom Benchmark Adapter

Create an adapter by subclassing `BaseBenchmarkAdapter` from `genai_graph.bench.adapters.base`:

```python
from pathlib import Path
from genai_graph.bench.adapters.base import BaseBenchmarkAdapter, download_hf_file, load_hf_dataset_to_pandas
from genai_graph.bench.models import BenchQuestion


class MyBenchmarkAdapter(BaseBenchmarkAdapter):
    """Adapter for MyBenchmark dataset."""

    DATASET_NAME = "my_org/my_benchmark"

    def load_dataset(self, split: str | None = "test", cache_dir: Path | None = None) -> list[BenchQuestion]:
        df = load_hf_dataset_to_pandas(self.DATASET_NAME, split=split or "test", cache_dir=cache_dir)
        questions: list[BenchQuestion] = []
        for _, row in df.iterrows():
            questions.append(
                BenchQuestion(
                    id=str(row["id"]),
                    doc_name=str(row["doc_name"]),
                    doc_names=[str(row["doc_name"])],
                    question=str(row["question"]),
                    gold_answer=str(row["answer"]),
                    justification=str(row.get("justification", "")),
                    evidence=[str(row.get("evidence", ""))],
                    metadata={"split": split, "domain": row.get("domain", "")},
                )
            )
        return questions

    def fetch_document(self, doc_name: str, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_filename = f"{doc_name}.pdf"
        dest_path = output_dir / pdf_filename
        if dest_path.exists():
            return dest_path
        return download_hf_file(
            repo_id=self.DATASET_NAME,
            filename=f"documents/{pdf_filename}",
            repo_type="dataset",
            dest_path=dest_path,
        )

    def get_judge_rubric(self) -> str:
        return (
            "You are an expert evaluator assessing answers against ground truth.
"
            "Criteria:
"
            "1. Correctness: Evaluate factual and semantic accuracy.
"
            "2. Numeric Precision: Match financial/numeric figures accounting for rounding and scale.
"
            "3. Groundedness: Verify answer is supported by the document context.
"
            "4. Anti-Hallucination: If gold answer is 'Not answerable', verify agent properly states info is absent."
        )
```

---

## Configuring `config/bench.yaml`

```yaml
default_profile: mistral_glm
adapter: mmlongbench.adapter.MMLongBenchAdapter

bench_profiles:
  mistral_glm:
    description: "GLM 5.3 Flash agent with Mistral OCR and DeepSeek V4 Pro judge"
    markdownize_profile: best
    monitoring: null

    llms:
      agent: glm_5.3_flash@openrouter
      build: deepseek-v4-flash-0731@openrouter
      judge: DeepSeek-V4-Pro-0813@openrouter

    build:
      skip_ocr: false
      force: false
      llm: deepseek-v4-flash-0731@openrouter
      structure_strategy: auto  # auto | algo | toc_preamble | llm_full
      summaries: true
      workers: 4
      summary_min_tokens: 800
      context_safety_ratio: 0.9
      embeddings: qwen3_06b@deepinfra
      fts: true
      chunk_size_tokens: 1500

    files:
      pathspecs: ["*"]          # glob / pathspec pattern
      limit: null

    agent:
      profile: default
      folder_id: null
      concurrency: 10

    judge:
      enabled: true
      concurrency: 5
```

### Path Interpolation in `bench.yaml`
The `{profile}` placeholder in directory paths is automatically substituted at runtime:
- Raw PDFs: `data/pdfs/`
- Markdown files: `data/markdown_multi/`
- Knowledge Graph DB: `data/kg/{profile}/lbug/` or `data/kg/mmlongbench_multi.db`
- Execution Traces: `data/mmlongbench/{profile}/runs.jsonl`
- Scored Verdicts: `data/mmlongbench/{profile}/scores.jsonl`

---

## Grading Criteria & Anti-Hallucination Evaluation

The evaluator in `genai_graph.bench.judge` implements standardized equivalence rules:
- **Numeric Normalization**: Commas, currency symbols (`$`, `€`), percentages (`50%` == `0.5`), scale notations (`M` == `,000,000`), and standard rounding are treated as equivalent.
- **Unanswerable Questions (Anti-Hallucination Probes)**:
  - When gold answer is `Not answerable`, `Unanswerable`, or `N/A` (e.g. ~22.5% of MMLongBench-Doc questions), the agent is scored `CORRECT` if it explicitly concludes that the requested predicate (year, entity, demographic, or metric) is absent from the document.
- **Correctness Tiers**:
  - `correct`: Substantively complete and accurate.
  - `partial`: Correct reasoning or partial answers, but missing details or sub-questions.
  - `incorrect`: Contradicts gold answer, wrong numerical calculations, or hallucinated facts.
- **Failure Taxonomy**:
  - `missing_ocr_or_visual_chart`: Required chart, diagram, or table text was omitted or unreadable in OCR.
  - `retrieval_or_lookup_error`: Agent failed to locate the relevant document, section, or page.
  - `calculation_or_math_error`: Agent retrieved correct figures but made arithmetic or scaling mistakes.
  - `halted_or_empty_response`: Agent hit max iteration, recursion limits, timeout, or returned an empty response.

---

## Code Map

| Concern | Path |
|---------|------|
| Benchmark CLI Commands | `genai_graph/core/commands_bench.py` |
| Core Data Models | `genai_graph/bench/models.py` |
| Config & Path Interpolation | `genai_graph/bench/config.py` |
| Prefect Pipeline Flows | `genai_graph/bench/flows.py` |
| Agent Execution & Harness Streaming | `genai_graph/bench/runner.py` |
| LLM-as-Judge Evaluation | `genai_graph/bench/judge.py` |
| Metrics Aggregation | `genai_graph/bench/summary.py` |
| Interactive Textual TUI | `genai_graph/bench/tui.py` |
| Knowledge Graph Construction | `genai_graph/kg/document_graph/build.py` |
| Document Graph Navigation & VLM Tools | `genai_graph/kg/query/document_graph_tools.py` |
| Base Adapter Interface & Utilities | `genai_graph/bench/adapters/base.py` |

## References

- [Unified Benchmark Framework Docs](docs/benchmark_framework.md)
- [Empirical Evaluation & Studies](docs/benchmarks_financebench_officeqa.md)
- [Document Graph Architecture](docs/document-graph.md)
