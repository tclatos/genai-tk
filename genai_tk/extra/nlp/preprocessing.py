"""Text preprocessing functions for NLP tasks (BM25, hybrid search, etc.).

Provides spaCy-based preprocessing (lemmatization, stop-word removal) that can be
used as a tokenizer for BM25 retrieval, hybrid search, or any other text processing
that benefits from linguistic normalization.

Example:
    ```python
    from genai_tk.extra.nlp.preprocessing import get_spacy_preprocess_fn

    preprocess = get_spacy_preprocess_fn()  # uses NlpConfig defaults
    tokens = preprocess("The quick brown foxes are jumping over the lazy dogs")
    # → ["quick", "brown", "fox", "jump", "lazy", "dog"]
    ```
"""

from __future__ import annotations

from collections.abc import Callable


def default_preprocessing_func(text: str) -> list[str]:
    """Simple whitespace tokenizer (no spaCy required)."""
    return text.split()


def get_spacy_preprocess_fn(
    model: str | None = None,
    language: str | None = None,
    more_stop_words: list[str] | None = None,
) -> Callable[[str], list[str]]:
    """Return a preprocessing function: lemmatization + lowercasing + stop-word removal.

    Args:
        model: spaCy model name. Defaults to the model from ``NlpConfig`` for *language*.
        language: Language code. Defaults to ``NlpConfig.default_language``.
        more_stop_words: Additional stop words to filter out.

    Returns:
        A callable that takes text and returns a list of preprocessed tokens.

    Raises:
        ImportError: If spaCy or the requested model is not available.
    """
    from genai_tk.extra.nlp.engine import get_nlp

    nlp = get_nlp(language=language, model=model)
    stop_words = set(nlp.Defaults.stop_words)
    if more_stop_words:
        stop_words.update(more_stop_words)

    def preprocess_text(text: str) -> list[str]:
        lemmas = [token.lemma_.lower() for token in nlp(text)]
        return [token for token in lemmas if token not in stop_words]

    return preprocess_text
