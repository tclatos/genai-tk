# Markdownization

`markdownize_flow` converts a directory of mixed documents — PDF, Word, PowerPoint,
Excel, and images — into Markdown with a **single call**. You pick one **profile**
(`fast` / `medium` / `best`); the profile decides how each kind of document is
converted. Low-level converter wiring and the LibreOffice → PDF → OCR path are
implementation details you never touch directly.

## Quick start

```bash
# Run the built-in workflow (uses the `medium` profile by default)
uv run cli workflow run markdownize --set base_dir=./docs --set output_dir=./md

# Choose a profile
uv run cli workflow run markdownize --preset fast  --set base_dir=./docs --set output_dir=./md
uv run cli workflow run markdownize --preset best  --set base_dir=./docs --set output_dir=./md
```

```python
from genai_tk.workflow.markdownize import markdownize_flow

markdownize_flow(base_dir="./docs", output_dir="./md", profile="medium")
```

**Supported inputs:** `.pdf`, `.doc`/`.docx`/`.odt`/`.rtf`, `.ppt`/`.pptx`/`.odp`,
`.xls`/`.xlsx`/`.ods`, images (`.png`/`.jpg`/`.jpeg`/`.gif`/`.bmp`), and
`.html`/`.htm`/`.csv`/`.json`.

## Profiles

A profile is the recommended way to select conversion quality. Three profiles are
built in and always available — no configuration required:

| Profile | Speed | Network | What it does |
|---------|-------|---------|--------------|
| `fast` | fastest | none | Everything converted locally (`markitdown`; spreadsheets via a deterministic parser). No LibreOffice, no API keys. |
| `medium` *(default)* | balanced | Mistral OCR | Office docs rendered through LibreOffice to PDF then OCR'd with Mistral; spreadsheets use the deterministic parser. Best general-purpose fidelity. |
| `best` | slowest | Mistral OCR | Everything (including spreadsheets) rendered to PDF then OCR'd for the highest fidelity. |

`default` is an alias for `medium`.

```python
from genai_tk.workflow.markdownize import get_markdownize_profile

profile = get_markdownize_profile("best")  # -> MarkdownizeProfile(...)
markdownize_flow(base_dir="./docs", output_dir="./md", profile="best")
```

### Requirements

- `medium` / `best` render Office documents with **LibreOffice**. If it is missing
  the flow fails fast with install instructions (`apt install libreoffice`,
  `brew install --cask libreoffice`, or `dnf install libreoffice`). Use the `fast`
  profile to convert without LibreOffice.
- `medium` / `best` OCR PDFs with **Mistral** — set `MISTRAL_API_KEY`. All PDFs
  (native and the ones produced from Office documents) are OCR'd in a **single
  batch job**, which is cheaper than per-file calls. If the batch fails, each PDF
  falls back to local `markitdown`.

## Customising profiles

Projects may override or add profiles under a `markdownize_profiles` config key.
Entries take precedence over the built-in of the same name and use the four
per-family converter fields:

```yaml
# any auto-scanned config YAML
markdownize_profiles:
  fast:
    pdf_converter: mistral   # override just the PDF backend of the built-in `fast`
  my_profile:
    ppt_converter: via_pdf   # via_pdf | markitdown
    doc_converter: markitdown
    excel_converter: messy_xls_parser   # via_pdf | markitdown | messy_xls_parser
    pdf_converter: edgeparse     # mistral | markitdown | edgeparse
```

```python
markdownize_flow(base_dir="./docs", output_dir="./md", profile="my_profile")
```

The `via_pdf` value on `ppt_converter` / `doc_converter` / `excel_converter` means
"render to PDF with LibreOffice first, then convert that PDF with `pdf_converter`".

## Incremental processing

A `manifest.json` in the output directory records each source file's content hash
and the profile fingerprint. Re-runs only reprocess files that changed — or all
files if you switch profiles (the fingerprint changes) or pass `force=True`.

## Related

- `office2pdf_flow` — standalone LibreOffice → PDF conversion, reused by the
  `via_pdf` path. See [prefect.md](prefect.md).
- [workflows.md](workflows.md) — YAML-driven orchestration that chains
  markdownize with other steps.
