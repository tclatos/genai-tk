"""DEPRECATED: use genai_tk.extra.nlp.model_manager instead.

This module is kept for backward compatibility only.
"""

import warnings

warnings.warn(
    "genai_tk.utils.spacy_model_mngr is deprecated. Use genai_tk.extra.nlp.model_manager instead.",
    DeprecationWarning,
    stacklevel=2,
)

from genai_tk.extra.nlp.model_manager import SpaCyModelManager  # noqa: F401, E402

__all__ = ["SpaCyModelManager"]
