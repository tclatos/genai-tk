# Markdownization

`markdownize_flow` converts a directory of mixed documents — PDF, Word, PowerPoint,
Excel, images, and text — into clean Markdown with a **single call**. You pick one **profile**
(`fast` / `medium` / `best` / `lighton` / `anydoc` / `llm` or a custom profile); the profile
routes files to specific converters via an ordered list of pathspecs. Low-level converter wiring
and the LibreOffice → PDF → OCR path are implementation details managed automatically.

## Quick start

```bash
# Run the built-in workflow (uses the `medium` profile by default)
uv run cli workflow run markdownize --set sources=./docs --set md_output_dir=./md

# Choose a profile
uv run cli workflow run markdownize --preset fast  --set sources=./docs --set md_output_dir=./md
uv run cli workflow run markdownize --preset best  --set sources=./docs --set md_output_dir=./md
uv run cli workflow run markdownize --preset lighton --set sources=./docs --set md_output_dir=./md
uv run cli workflow run markdownize --preset anydoc --set sources=./docs --set md_output_dir=./md
uv run cli workflow run markdownize --preset llm --set sources=./docs --set md_output_dir=./md
```

```python
from genai_tk.workflow.markdownize import markdownize_flow

markdownize_flow(sources="./docs", md_output_dir="./md", profile="medium")
```

**Supported inputs:** `.pdf`, `.doc`/`.docx`/`.docm`/`.odt`/`.rtf`, `.ppt`/`.pptx`/`.odp`/`.pps`/`.pot`,
`.xls`/`.xlsx`/`.xlsm`/`.xlsb`/`.ods`, images (`.png`/`.jpg`/`.jpeg`/`.gif`/`.bmp`/`.webp`), and
`.html`/`.htm`/`.csv`/`.json`/`.txt`/`.epub`.

## Supported Converters

The toolkit provides 7 document converter engines in `genai_tk.extra.markdownize`:

| Converter | Description | Supported Formats | Requirements |
|---|---|---|---|
| `markitdown` | Microsoft MarkItDown local parser | Office, PDF, HTML, CSV, JSON, images | Built-in |
| `messy_xls` | Deterministic spreadsheet parser handling merged headers & multi-table sheets | `.xlsx`, `.xls`, `.ods` | Built-in (`openpyxl`) |
| `edgeparse` | Local edgeparse PDF parser | `.pdf` | `edgeparse` |
| `mistral_ocr` | Mistral AI Document OCR (single & batch API) | `.pdf`, Word, PPT, OpenDocument, images | `MISTRAL_API_KEY` |
| `lighton_ocr` | LightOn AI Parse REST API (sync & async polling modes) | `.pdf`, Office, images, HTML | `LIGHTON_API_KEY` |
| `anydoc` | Firecrawl anydoc Rust parser | Word, PPT, Excel, OpenDoc, RTF, EPUB, PDF | `firecrawl-anydoc` |
| `llm` | LangChain LLM factory async batch multimodal transcription | Images, PDFs, text, code, HTML | Provider API key |

## Profiles

Built-in profiles are defined in `config/markdownize.yaml`:

| Profile | Speed | Network | What it does |
|---|---|---|---|
| `fast` | fastest | none | Everything converted locally (`markitdown`; spreadsheets via `messy_xls`). No LibreOffice, no API keys. |
| `medium` *(default)* | balanced | Mistral OCR | Office docs rendered through LibreOffice to PDF then OCR'd with Mistral; spreadsheets use `messy_xls`. |
| `best` | slowest | Mistral OCR | Everything (including spreadsheets) rendered to PDF then OCR'd for highest fidelity. |
| `lighton` | fast | LightOn API | Converted via LightOn OCR API (`LIGHTON_API_KEY`). |
| `anydoc` | fast | none | Converted locally with Firecrawl anydoc Rust engine. |
| `llm` | dynamic | LLM provider | Converted with the configured LangChain LLM model (`llm: default`). |

`default` is an alias for `medium`.

```python
from genai_tk.workflow.markdownize import get_markdownize_profile

profile = get_markdownize_profile("lighton")
markdownize_flow(sources="./docs", md_output_dir="./md", profile=profile)
```

## Customising Profiles & Selectors

You can define custom converter instances and ordered pathspec rules in your YAML config:

```yaml
markdownize_converters:
  custom_llm:
    class: genai_tk.extra.markdownize.llm_converter.LLMConverter
    params:
      llm: claude_sonnet_37@anthropic
      max_concurrency: 8

markdownize_profiles:
  my_custom_profile:
    rules:
      - pathspec: "**/*.{xlsx,xls,ods}"
        converter: messy_xls
      - pathspec: "**/scanned_*.pdf"
        converter: lighton_ocr
      - pathspec: "**/*.pdf"
        converter: custom_llm
      - pathspec: "**/*"
        converter: markitdown
```

The `via_pdf` route can be assigned to any Office format rule to convert the document to PDF via LibreOffice first, then process it through the matching PDF converter.

## Incremental Processing

A `manifest.json` in the output cache directory records each source file's content hash and the profile fingerprint. Re-runs only reprocess files that changed — or all files if you switch profiles or pass `force_stage="md"`.
