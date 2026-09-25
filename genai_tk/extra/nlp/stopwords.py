"""Stop words retrieval and Ladybug / Snowball stemmer mapping.

Provides language stop-words retrieval backed by spaCy language classes
and maps language codes to Ladybug full-text search (BM25) stemmer configurations.
"""

from __future__ import annotations

import functools
from collections import Counter
from typing import Sequence

from loguru import logger

try:
    import snowballstemmer
except ImportError:  # pragma: no cover
    snowballstemmer = None  # type: ignore[assignment]

# Map ISO 639-1 code to Ladybug / Snowball stemmer names
# Supported Snowball stemmers in Ladybug:
# arabic, basque, catalan, danish, dutch, english, finnish, french, german,
# greek, hindi, hungarian, indonesian, irish, italian, lithuanian, nepali,
# norwegian, porter, portuguese, romanian, russian, serbian, spanish, swedish,
# tamil, turkish, none
_ISO_TO_LADYBUG_STEMMER: dict[str, str] = {
    "ar": "arabic",
    "eu": "basque",
    "ca": "catalan",
    "da": "danish",
    "nl": "dutch",
    "en": "english",
    "fi": "finnish",
    "fr": "french",
    "de": "german",
    "el": "greek",
    "hi": "hindi",
    "hu": "hungarian",
    "id": "indonesian",
    "ga": "irish",
    "it": "italian",
    "lt": "lithuanian",
    "ne": "nepali",
    "no": "norwegian",
    "nb": "norwegian",
    "nn": "norwegian",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "sr": "serbian",
    "es": "spanish",
    "sv": "swedish",
    "ta": "tamil",
    "tr": "turkish",
    # Non-stemmed languages (CJK, Vietnamese, Thai, etc.)
    "zh": "none",
    "ja": "none",
    "ko": "none",
    "th": "none",
    "vi": "none",
}

# Fallback minimal English stop words if spaCy is not installed
_FALLBACK_EN_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can't",
        "cannot",
        "could",
        "couldn't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn't",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "he'd",
        "he'll",
        "he's",
        "her",
        "here",
        "here's",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "how's",
        "i",
        "i'd",
        "i'll",
        "i'm",
        "i've",
        "if",
        "in",
        "into",
        "is",
        "isn't",
        "it",
        "it's",
        "its",
        "itself",
        "let's",
        "me",
        "more",
        "most",
        "mustn't",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "shan't",
        "she",
        "she'd",
        "she'll",
        "she's",
        "should",
        "shouldn't",
        "so",
        "some",
        "such",
        "than",
        "that",
        "that's",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "there's",
        "these",
        "they",
        "they'd",
        "they'll",
        "they're",
        "they've",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "wasn't",
        "we",
        "we'd",
        "we'll",
        "we're",
        "we've",
        "were",
        "weren't",
        "what",
        "what's",
        "when",
        "when's",
        "where",
        "where's",
        "which",
        "while",
        "who",
        "who's",
        "whom",
        "why",
        "why's",
        "with",
        "won't",
        "would",
        "wouldn't",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)


def _normalize_lang_code(code: str) -> str:
    """Normalize language code to lowercase 2-letter base code."""
    cleaned = code.strip().lower()
    cleaned = cleaned.split("-")[0].split("_")[0]
    return cleaned


@functools.lru_cache(maxsize=32)
def get_stopwords(language_code: str = "en", *, fallback_to_english: bool = True) -> set[str]:
    """Return the set of stop words for a given language code using spaCy Defaults.

    Args:
        language_code: Two-letter ISO 639-1 code (e.g. ``"en"``, ``"fr"``, ``"de"``, ``"es"``).
        fallback_to_english: If True, returns English stop words when language is not found.

    Returns:
        Set of lowercased stop word strings.

    Example:
        ```python
        french_stops = get_stopwords("fr")
        assert "le" in french_stops and "la" in french_stops
        ```
    """
    code = _normalize_lang_code(language_code)
    try:
        import spacy.util

        lang_cls = spacy.util.get_lang_class(code)
        if hasattr(lang_cls, "Defaults") and hasattr(lang_cls.Defaults, "stop_words"):
            return set(lang_cls.Defaults.stop_words)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load spaCy stop words for '{}': {}", code, exc)

    if fallback_to_english:
        try:
            import spacy.util

            en_cls = spacy.util.get_lang_class("en")
            return set(en_cls.Defaults.stop_words)
        except Exception:  # noqa: BLE001
            return set(_FALLBACK_EN_STOP_WORDS)
    return set()


def get_stopwords_union(language_codes: Sequence[str]) -> set[str]:
    """Return the union of stop words for a collection of language codes.

    Args:
        language_codes: Sequence of ISO 639-1 language codes.

    Returns:
        Union set of all stop words across the specified languages.
    """
    result: set[str] = set()
    for code in language_codes:
        if not code:
            continue
        stops = get_stopwords(code, fallback_to_english=False)
        result.update(stops)
    return result


def get_ladybug_stemmer(language_code: str = "en", default: str = "english") -> str:
    """Return the Ladybug / Snowball stemmer name corresponding to a language code.

    Args:
        language_code: ISO 639-1 language code (e.g. ``"en"``, ``"fr"``, ``"de"``, ``"zh"``).
        default: Fallback stemmer name if code is unknown (defaults to ``"english"``).

    Returns:
        Snowball stemmer name accepted by Ladybug's ``CREATE_FTS_INDEX(..., stemmer := '...')``.
    """
    code = _normalize_lang_code(language_code)
    return _ISO_TO_LADYBUG_STEMMER.get(code, default)


@functools.lru_cache(maxsize=32)
def _get_stemmer(stemmer_name: str) -> object | None:
    """Return a cached Snowball stemmer for a Ladybug stemmer name.

    Args:
        stemmer_name: Ladybug/Snowball stemmer name (e.g. ``"english"``, ``"french"``).

    Returns:
        A Snowball stemmer instance, or ``None`` when the snowballstemmer package
        is not installed or the name is not a known Snowball stemmer.
    """
    if snowballstemmer is None:
        return None
    try:
        return snowballstemmer.stemmer(stemmer_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unknown Snowball stemmer '{}': {}", stemmer_name, exc)
        return None


def stem_stopwords(stopwords: set[str], stemmer_name: str) -> set[str]:
    """Return the stop words reduced to the form Ladybug's FTS index compares against.

    Ladybug stems indexed tokens with the index's Snowball stemmer before matching
    them against the stop-word list, so stop words should be provided in their
    stemmed form for best accuracy (see the FTS extension documentation).

    Args:
        stopwords: Stop words (lowercased), e.g. from `get_stopwords` or `get_stopwords_union`.
        stemmer_name: Ladybug/Snowball stemmer name (e.g. ``"english"``, ``"french"``,
            ``"porter"``), or ``"none"``.

    Returns:
        The set of stemmed stop words (duplicates collapsed). Returned unchanged
        when the stemmer is ``"none"`` or snowballstemmer is not installed.
    """
    if not stopwords or stemmer_name == "none":
        return set(stopwords)
    stemmer = _get_stemmer(stemmer_name)
    if stemmer is None:
        return set(stopwords)
    return {stemmer.stemWord(word) for word in stopwords}


def get_dominant_language(language_codes: Sequence[str], default: str = "en") -> str:
    """Determine the dominant (most frequent) language code from a collection.

    Args:
        language_codes: Sequence of language codes (e.g. from documents in a corpus).
        default: Fallback code if sequence is empty.

    Returns:
        The most frequent language code.
    """
    valid = [_normalize_lang_code(c) for c in language_codes if c and c.strip()]
    if not valid:
        return default
    counts = Counter(valid)
    return counts.most_common(1)[0][0]
