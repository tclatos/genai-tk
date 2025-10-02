"""Constants for testing with fake models.

This module provides standardized constants for fake models to ensure
consistent usage across all test files.
"""

# Fake model identifiers
FAKE_LLM_ID = "parrot_local_fake"
FAKE_EMBEDDINGS_ID = "embeddings_768_fake"

# Fake provider names
FAKE_LLM_PROVIDER = "fake"
FAKE_EMBEDDINGS_PROVIDER = "fake"

# Fake model properties
FAKE_EMBEDDINGS_DIMENSION = 768
FAKE_LLM_MODEL_NAME = "parrot"

# Test configuration
PYTEST_CONFIG_NAME = "pytest"

# Common test data
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Python is a powerful programming language for AI development.",
    "Machine learning models require training data to perform well.",
    "Vector databases enable efficient similarity search.",
    "Embeddings represent text as numerical vectors.",
]

SAMPLE_QUERIES = [
    "programming language",
    "machine learning",
    "similarity search",
    "artificial intelligence",
    "data science",
]
