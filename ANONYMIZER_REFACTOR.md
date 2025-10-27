# Custom Presidio Anonymizer Refactoring

## Problem
The original implementation in `genai_tk/extra/custom_presidio_anonymizer.py` was using the outdated `langchain-experimental` package (v0.0.42), which is incompatible with the current `presidio-anonymizer` API (v2.2.360). This caused a `TypeError` when attempting to anonymize text.

## Solution
Replaced the dependency on `langchain_experimental.data_anonymizer.PresidioReversibleAnonymizer` with a custom implementation using the current Presidio API directly.

## Key Changes

### 1. Dependencies
**Removed:**
- `langchain_experimental.data_anonymizer.PresidioReversibleAnonymizer`
- `langchain_experimental.data_anonymizer.deanonymizer_matching_strategies`

**Added:**
- `presidio_analyzer.AnalyzerEngine` - for PII detection
- `presidio_anonymizer.AnonymizerEngine` - for text anonymization
- `rapidfuzz.fuzz` - for fuzzy string matching during deanonymization

### 2. Implementation Details

#### Anonymization
- Uses `AnalyzerEngine` to detect PII entities in text
- Maintains a mapping dictionary (`_mapping`) to track original → fake value pairs
- Generates fake values using Faker library based on entity type
- Manually replaces detected entities to handle overlapping cases correctly

#### Deanonymization
- Implements custom reverse mapping from fake → original values
- Supports both exact and fuzzy matching modes
- Uses `rapidfuzz` for fuzzy string matching (threshold configurable via `fuzzy_matching_threshold` parameter)

#### Custom Recognizers
- Companies and products are now added to the `AnalyzerEngine.registry` instead of the anonymizer
- Entities are properly tracked in the `analyzed_fields` list

### 3. API Compatibility
The public API remains unchanged:
- `anonymize(text: str) -> str`
- `deanonymize(text: str, use_fuzzy_matching: bool = True) -> str`
- `get_mapping() -> dict[str, Any]`
- `add_custom_recognizer(...)`
- `check_spacy_model_status(model_name: str) -> dict[str, Any]`

### 4. Code Quality Improvements
- Updated type hints to use Python 3.12 syntax (`list[str]` instead of `List[str]`)
- Added comprehensive Google-style docstrings with examples
- Fixed all import statements to work with current package versions

## Testing
The test code at the bottom of the file successfully:
- Anonymizes names, emails, phone numbers, companies, and products
- Deanonymizes the text back to original values
- Maintains a complete mapping of all replacements

## Dependencies Required
- `presidio-analyzer >= 2.2.360`
- `presidio-anonymizer >= 2.2.360`
- `rapidfuzz >= 3.0.0`
- `faker`
- `pydantic`

All dependencies are already present in the project environment.
