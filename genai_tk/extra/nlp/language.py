"""Language detection utilities based on lingua-py.

Provides fast and accurate natural language identification restricted to
common business languages (European languages, CJK, Arabic, etc.).
"""

from __future__ import annotations

import functools

from lingua import Language, LanguageDetector, LanguageDetectorBuilder
from loguru import logger

# Curated business languages: major European languages, CJK, Arabic, etc.
BUSINESS_LANGUAGES: tuple[Language, ...] = (
    Language.ENGLISH,
    Language.FRENCH,
    Language.GERMAN,
    Language.SPANISH,
    Language.ITALIAN,
    Language.PORTUGUESE,
    Language.DUTCH,
    Language.RUSSIAN,
    Language.POLISH,
    Language.SWEDISH,
    Language.DANISH,
    Language.BOKMAL,
    Language.NYNORSK,
    Language.FINNISH,
    Language.CZECH,
    Language.HUNGARIAN,
    Language.ROMANIAN,
    Language.GREEK,
    Language.TURKISH,
    Language.ARABIC,
    Language.CHINESE,
    Language.JAPANESE,
    Language.KOREAN,
)

# Map ISO 639-1 code string (e.g. "en", "fr") to lingua Language enum
_ISO_TO_LANGUAGE: dict[str, Language] = {}
for lang in Language.all():
    try:
        iso = lang.iso_code_639_1.name.lower()
        _ISO_TO_LANGUAGE[iso] = lang
    except Exception:  # noqa: BLE001
        continue
# Alias 'no' -> BOKMAL
if "no" not in _ISO_TO_LANGUAGE and hasattr(Language, "BOKMAL"):
    _ISO_TO_LANGUAGE["no"] = Language.BOKMAL


@functools.lru_cache(maxsize=8)
def get_language_detector(
    languages: tuple[Language, ...] | None = None,
    *,
    low_accuracy_mode: bool = False,
) -> LanguageDetector:
    """Return a cached Lingua LanguageDetector instance.

    Args:
        languages: Tuple of `Language` enums to detect. Defaults to `BUSINESS_LANGUAGES`.
        low_accuracy_mode: When True, reduces memory and CPU for slightly less accuracy.

    Returns:
        Configured LanguageDetector.
    """
    selected_langs = list(languages) if languages is not None else list(BUSINESS_LANGUAGES)
    builder = LanguageDetectorBuilder.from_languages(*selected_langs)
    if low_accuracy_mode:
        builder = builder.with_low_accuracy_mode()
    return builder.build()


def detect_language(
    text: str,
    *,
    languages: list[str] | None = None,
    min_length: int = 15,
    max_sample_chars: int = 5000,
) -> str | None:
    """Detect the dominant language of a text snippet or document.

    Args:
        text: Input text to analyze.
        languages: Optional list of ISO 639-1 language codes (e.g. `["en", "fr", "de"]`)
            to restrict detection to. Defaults to all `BUSINESS_LANGUAGES`.
        min_length: Minimum non-whitespace character length required to attempt detection.
        max_sample_chars: Maximum characters to inspect from the input text (for speed).

    Returns:
        Two-letter ISO 639-1 lowercase code (e.g. ``"en"``, ``"fr"``, ``"de"``, ``"zh"``),
        or ``None`` if the text is empty/too short or the language cannot be identified.

    Example:
        ```python
        lang = detect_language("Ce rapport présente les résultats financiers annuels.")
        assert lang == "fr"
        ```
    """
    cleaned = text.strip() if text else ""
    if len(cleaned) < min_length:
        return None

    sample = cleaned[:max_sample_chars]

    selected_languages: tuple[Language, ...] | None = None
    if languages:
        resolved = []
        for code in languages:
            norm = code.strip().lower().split("-")[0].split("_")[0]
            if norm in _ISO_TO_LANGUAGE:
                resolved.append(_ISO_TO_LANGUAGE[norm])
        if resolved:
            selected_languages = tuple(resolved)

    try:
        detector = get_language_detector(selected_languages)
        result = detector.detect_language_of(sample)
        if result is None:
            return None
        iso_code = result.iso_code_639_1.name.lower()
        # Normalize Norwegian Bokmal/Nynorsk (nb/nn) to 'no' for standard ISO-639-1 if needed
        if iso_code in ("nb", "nn"):
            return "no"
        return iso_code
    except Exception as exc:  # noqa: BLE001
        logger.warning("Language detection failed: {}", exc)
        return None
