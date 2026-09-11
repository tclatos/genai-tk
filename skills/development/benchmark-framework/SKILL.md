---
name: benchmark-framework
description: Build, configure, and execute multi-dataset benchmarks (cli bench) with custom adapters, document-to-knowledge-graph pipelines, agent harness evaluation, LLM-as-judge scoring, and TUI inspection.
tags: [benchmark, evaluation, cli, llm-as-judge, adapter, docgraph, datasets]
version: "1.0"
author: ""
category: development
---

# Benchmark Framework Guide (`cli bench`)

The Unified Benchmark Framework (`genai_graph.bench`) provides a standardized, dataset-agnostic pipeline for evaluating autonomous agents and Document Graph retrieval systems across diverse datasets (e.g., **FinanceBench**, **OfficeQA**, **MMLongBench-Doc**).

## When to Use

- Implementing or scaffolding a new benchmark evaluation suite (e.g. SEC filings, legal contracts, technical manuals).
- Adding or configuring benchmark profiles in `config/bench.yaml`.
- Running end-to-end evaluation runs, re-grading existing agent trajectories, or inspecting reports via CLI or TUI.
- Authoring custom `BaseBenchmarkAdapter` classes with domain-specific dataset loaders, document fetchers, and judge rubrics.
- Debugging evaluation failures, numeric equivalence issues, OCR gaps, or trajectory errors.

---

## CLI Commands Reference

Benchmark commands are registered via `genai_graph.core.commands_bench.BenchCommands`:

```bash
# 1. List configured benchmark profiles
uv run cli bench list

# 2. Run end-to-end benchmark (Fetch -> OCR -> KG Build -> Agent Run -> LLM Judge)
uv run cli bench run -p mistral_glm
uv run cli bench run -p mistral_glm -q financebench_001,financebench_002  # run specific questions
uv run cli bench run -p mistral_glm --dry-run                           # inspect execution plan
uv run cli bench run -p mistral_glm --no-grade                          # run agent only without grading

# 3. Re-grade existing runs without re-running agent
uv run cli bench grade -p mistral_glm
uv run cli bench grade -p mistral_glm --force                           # force re-grade all questions

# 4. Generate aggregated metric summary report
uv run cli bench report -p mistral_glm
uv run cli bench report -p mistral_glm --json                           # export summary JSON

# 5. Question and trajectory inspection
uv run cli bench questions -p mistral_glm                               # tabular view
uv run cli bench questions -p mistral_glm -t                            # interactive Textual TUI
uv run cli bench questions -p mistral_glm -t --trajectory               # TUI with live trajectory replay
```

---

## 6-Stage Pipeline Architecture

```
[1. Fetch Dataset & Documents]
         │
         ▼
[2. Markdownize & OCR Ladder] (Mistral OCR -> Docling -> MarkItDown)
         │
         ▼
[3. Hierarchical Graph Ingestion] (Ladybug DB + BM25 & Vector Indexing)
         │
         ▼
[4. Autonomous Agent Execution] (DeepAgent / React harness + trajectory recording)
         │
         ▼
[5. LLM-as-Judge Evaluation] (Mafin 2.5 numeric equivalence + groundedness)
         │
         ▼
[6. Metrics & Summary Aggregation] (Accuracy, partial, token counts, error breakdown)
```

1. **Fetch**: Retrieves questions from HuggingFace, CSV, or local files; caches raw PDFs.
2. **Markdownize**: Converts documents using profile-configured strategy (`fast`, `medium`, `best`) or points directly to `saved_markdown_dir`.
3. **Build Graph**: Parses markdown into `Folder/Document/MarkdownSection` hierarchy inside Ladybug DB.
4. **Run Agent**: Dispatches questions to the agent harness, recording tool calls, thinking traces, and token usage into `runs.jsonl`.
5. **Judge**: Evaluates outputs against `gold_answer` and `evidence` using domain rubrics, outputting `scores.jsonl`.
6. **Aggregate**: Generates `summary.json` with accuracy tiers, numeric match rates, and failure taxonomy.

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
            "You are an expert evaluator assessing answers against ground truth.\n"
            "Criteria:\n"
            "1. Correctness: Evaluate factual and semantic accuracy.\n"
            "2. Numeric Precision: Match financial/numeric figures accounting for rounding and scale.\n"
            "3. Groundedness: Verify answer is supported by the document context."
        )
```

---

## Configuring `config/bench.yaml`

```yaml
default_profile: mistral_glm
adapter: my_package.adapter.MyBenchmarkAdapter

bench_profiles:
  mistral_glm:
    llms:
      agent: glm_5.2@openrouter
      judge: DeepSeek-V4-Pro@openrouter
    build:
      skip_ocr: false
      structure_strategy: auto  # auto | algo | toc_preamble | llm_full
      summaries: true
      workers: 4
    files:
      pathspecs: ["*"]          # gitignore-style filtering pattern
      limit: null
    agent:
      profile: default
      folder_id: null
      concurrency: 10
    judge:
      concurrency: 10
    monitoring: null            # null | "langfuse" | "langsmith"
```

### Path Interpolation in `bench.yaml`
The `{profile}` placeholder in directory paths is automatically substituted at runtime:
- Raw PDFs: `data/pdfs/`
- Markdown files: `data/markdown/{profile}/` or `data/markdown_multi/{profile}/`
- Knowledge Graph DB: `data/kg/{profile}/lbug/`
- Execution Traces: `data/traces/{profile}/runs.jsonl`
- Scored Verdicts: `data/traces/{profile}/scores.jsonl`

---

## Grading Criteria & Mafin 2.5 Rules

The evaluator in `genai_graph.bench.judge` implements Mafin 2.5 equivalence rules:
- **Numeric Normalization**: Commas, currency symbols (`$`, `€`), percentages (`50%` == `0.5`), scale notations (`$5M` == `$5,000,000`), and standard rounding are treated as equivalent.
- **Correctness Tiers**:
  - `CORRECT`: Substantively complete and accurate.
  - `PARTIAL`: Correct reasoning or partial answers, but missing details or sub-questions.
  - `INCORRECT`: Contradicts gold answer, wrong numerical calculations, or hallucinated facts.
- **Failure Taxonomy**:
  - `missing_ocr`: Required table/text was omitted in the Markdownize step.
  - `retrieval`: Agent failed to retrieve relevant sections or nodes.
  - `calculation`: Agent retrieved correct figures but made arithmetic errors.
  - `halted`: Agent hit max iteration or timeout limits.

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
| Knowledge Graph Construction | `genai_graph/bench/build_graph.py` |
| Base Adapter Interface & Utilities | `genai_graph/bench/adapters/base.py` |

## References

- [Unified Benchmark Framework Docs](docs/benchmark_framework.md)
- [Empirical Evaluation & Studies](docs/benchmarks_financebench_officeqa.md)
- [Document Graph Architecture](docs/document-graph.md)
