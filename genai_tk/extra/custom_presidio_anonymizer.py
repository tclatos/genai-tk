"""Reversible anonymization with Presidio and fuzzy matching support.

Custom implementation that works with updated presidio-anonymizer API.
Based on LangChain's approach but adapted for current presidio versions.
"""

import string
from typing import Any

from faker import Faker
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from rapidfuzz import fuzz

from genai_tk.utils.spacy_model_mngr import SpaCyModelManager


class CustomizedPresidioAnonymizer(BaseModel):
    """A configurable anonymizer with reversible operations and fuzzy matching.

    Provides a generic interface for PII detection, anonymization, and deanonymization
    with support for custom recognizers and reversible operations using fuzzy matching.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    analyzed_fields: list[str] = Field(
        default_factory=lambda: ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD"]
    )

    company_names: list[str] = Field(default_factory=list)
    product_names: list[str] = Field(default_factory=list)
    spacy_model: str = Field(default="en_core_web_sm")  # use large one in production
    language: str = Field(default="en")

    faker_seed: int | None = None
    fuzzy_matching_threshold: int = Field(default=85)

    _analyzer: AnalyzerEngine = PrivateAttr()
    _anonymizer: AnonymizerEngine = PrivateAttr()
    _faker: Faker = PrivateAttr()
    _mapping: dict[str, str] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Initialize the anonymizer after model creation."""
        # Ensure SpaCy model is available and configure the environment
        SpaCyModelManager.setup_spacy_model(self.spacy_model)

        # Initialize Presidio engines
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()

        # Initialize faker for custom operators
        if self.faker_seed is not None:
            Faker.seed(self.faker_seed)
        self._faker = Faker(locale=["en-US", "fr-FR"])

        # Initialize mapping
        self._mapping = {}

        # Add custom recognizers
        self._add_custom_recognizers()

    def _add_custom_recognizers(self) -> None:
        """Add custom recognizers for companies and products."""
        if self.company_names:
            company_pattern = r"(?i)\b(" + "|".join(map(str, self.company_names)) + r")\b"
            company_recognizer = PatternRecognizer(
                supported_entity="COMPANY",
                patterns=[Pattern(name="company_pattern", regex=company_pattern, score=0.9)],
                context=["company", "organization", "firm", "enterprise", "business"],
            )
            self._analyzer.registry.add_recognizer(company_recognizer)

        if self.product_names:
            product_pattern = r"(?i)\b(" + "|".join(map(str, self.product_names)) + r")\b"
            product_recognizer = PatternRecognizer(
                supported_entity="PRODUCT",
                patterns=[Pattern(name="product_pattern", regex=product_pattern, score=0.9)],
                context=["product", "service", "solution", "platform", "tool"],
            )
            self._analyzer.registry.add_recognizer(product_recognizer)

    def _generate_fake_value(self, entity_type: str) -> str:
        """Generate a fake value based on entity type."""
        if entity_type == "PERSON":
            return self._faker.name()
        elif entity_type == "EMAIL_ADDRESS":
            return self._faker.email()
        elif entity_type == "PHONE_NUMBER":
            return self._faker.phone_number()
        elif entity_type == "CREDIT_CARD":
            return self._faker.credit_card_number()
        elif entity_type == "COMPANY":
            return self._faker.bothify(text="COMP####")
        elif entity_type == "PRODUCT":
            return self._faker.bothify(text="PROD####")
        else:
            return self._faker.bothify(text=f"{entity_type[:4].upper()}####")

    def anonymize(self, text: str) -> str:
        """Anonymize text by replacing PII with fake data.

        Args:
            text: Input text containing PII

        Returns:
            Anonymized text with PII replaced by fake data

        Example:
            ```python
            anonymizer = CustomizedPresidioAnonymizer()
            result = anonymizer.anonymize("John Smith's email is john@example.com")
            # Returns: "Jane Doe's email is jane123@example.org"
            ```
        """
        # Build entity list including custom recognizers
        entities_to_analyze = self.analyzed_fields.copy()
        if self.company_names:
            entities_to_analyze.append("COMPANY")
        if self.product_names:
            entities_to_analyze.append("PRODUCT")

        # Analyze text to find PII
        analyzer_results = self._analyzer.analyze(
            text=text,
            entities=entities_to_analyze,
            language=self.language,
        )

        # Sort by start position to process in order
        analyzer_results = sorted(analyzer_results, key=lambda x: x.start)

        # Build the anonymized text manually to handle overlapping entities
        result_text = text
        offset = 0

        for result in analyzer_results:
            entity_text = text[result.start : result.end]

            # Check if we already have a mapping for this text
            if entity_text not in self._mapping:
                fake_value = self._generate_fake_value(result.entity_type)
                self._mapping[entity_text] = fake_value
            else:
                fake_value = self._mapping[entity_text]

            # Replace in the result text
            start_pos = result.start + offset
            end_pos = result.end + offset
            result_text = result_text[:start_pos] + fake_value + result_text[end_pos:]
            offset += len(fake_value) - len(entity_text)

        return result_text

    def deanonymize(
        self,
        text: str,
        use_fuzzy_matching: bool = True,
    ) -> str:
        """Deanonymize text by restoring original PII.

        Args:
            text: Anonymized text to restore
            use_fuzzy_matching: Whether to use fuzzy string matching for better results

        Returns:
            Text with original PII restored

        Example:
            ```python
            anonymizer = CustomizedPresidioAnonymizer()
            anonymized = anonymizer.anonymize("John Smith works here")
            original = anonymizer.deanonymize(anonymized)
            # Returns: "John Smith works here"
            ```
        """
        result_text = text

        # Create reverse mapping
        reverse_mapping = {v: k for k, v in self._mapping.items()}

        if use_fuzzy_matching:
            # Use fuzzy matching to handle slight variations
            for fake_value, original_value in reverse_mapping.items():
                # Try exact match first
                if fake_value in result_text:
                    result_text = result_text.replace(fake_value, original_value)
                else:
                    # Try fuzzy matching for parts that might have changed
                    words = result_text.split()
                    for i, word in enumerate(words):
                        # Remove punctuation for matching
                        clean_word = word.strip(string.punctuation)
                        if fuzz.ratio(clean_word, fake_value) >= self.fuzzy_matching_threshold:
                            words[i] = word.replace(clean_word, original_value)
                    result_text = " ".join(words)
        else:
            # Exact matching only
            for fake_value, original_value in reverse_mapping.items():
                result_text = result_text.replace(fake_value, original_value)

        return result_text

    def get_mapping(self) -> dict[str, Any]:
        """Get the anonymization mapping for inspection.

        Returns:
            Dictionary mapping original PII values to their fake replacements
        """
        return self._mapping.copy()

    @staticmethod
    def check_spacy_model_status(model_name: str) -> dict[str, Any]:
        """Check the status of the SpaCy model.

        Args:
            model_name: Name of the SpaCy model to check

        Returns:
            Dictionary with model status information
        """
        model_path = SpaCyModelManager.get_model_path(model_name)

        return {
            "model_name": model_name,
            "is_installed": SpaCyModelManager.is_model_installed(model_name),
            "model_path": str(model_path),
            "path_exists": model_path.exists(),
        }

    def add_custom_recognizer(
        self,
        entity_name: str,
        patterns: list[str],
        context_words: list[str],
        replacement_format: str = "####",
    ) -> None:
        """Add a custom recognizer for specific entity types.

        Args:
            entity_name: Name of the entity
            patterns: List of regex patterns to match
            context_words: List of context words for better recognition
            replacement_format: Format for fake replacement values

        Example:
            ```python
            anonymizer = CustomizedPresidioAnonymizer()
            anonymizer.add_custom_recognizer(
                entity_name="EMPLOYEE_ID",
                patterns=[r"EMP-\\d{5}"],
                context_words=["employee", "staff", "worker"],
                replacement_format="EMP-####"
            )
            ```
        """
        combined_pattern = r"(?i)\b(" + "|".join(patterns) + r")\b"

        recognizer = PatternRecognizer(
            supported_entity=entity_name,
            patterns=[Pattern(name=f"{entity_name}_pattern", regex=combined_pattern, score=0.9)],
            context=context_words,
        )

        self._analyzer.registry.add_recognizer(recognizer)

        # Add entity to analyzed fields if not present
        if entity_name not in self.analyzed_fields:
            self.analyzed_fields.append(entity_name)


if __name__ == "__main__":
    """Quick test of the anonymizer functionality."""
    print("Testing CustomizedPresidioAnonymizer...")

    # Initialize anonymizer with sample data
    anonymizer = CustomizedPresidioAnonymizer(
        company_names=["Acme Corp", "Tech Solutions"], product_names=["WidgetPro", "CloudMaster"], faker_seed=42
    )

    # Test text with PII
    test_text = """
    John Smith works at Acme Corp and uses WidgetPro for development.
    His email is john.smith@email.com and phone is (555) 123-4567.
    He previously worked at Tech Solutions where he used CloudMaster.
    """

    print("\nOriginal text:")
    print(test_text)

    # Anonymize
    anonymized = anonymizer.anonymize(test_text)
    print("\nAnonymized text:")
    print(anonymized)

    # Deanonymize
    deanonymized = anonymizer.deanonymize(anonymized)
    print("\nDeanonymized text:")
    print(deanonymized)

    # Show mapping
    mapping = anonymizer.get_mapping()
    print("\nAnonymization mapping:")
    for fake, real in mapping.items():
        print(f"  {fake} -> {real}")

    print("\nTest completed!")
