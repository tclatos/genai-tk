"""LangChain agent middleware for PII anonymization and sensitivity-based LLM routing.

Public exports
--------------
Shared detector
    :class:`~genai_tk.agents.langchain.middleware.presidio_detector.PresidioDetectorConfig`
    :class:`~genai_tk.agents.langchain.middleware.presidio_detector.PresidioDetector`
    :class:`~genai_tk.agents.langchain.middleware.presidio_detector.DetectedEntity`
    :class:`~genai_tk.agents.langchain.middleware.presidio_detector.CustomRecognizerConfig`

Anonymization middleware
    :class:`~genai_tk.agents.langchain.middleware.anonymization_middleware.AnonymizationConfig`
    :class:`~genai_tk.agents.langchain.middleware.anonymization_middleware.AnonymizationMiddleware`

Sensitivity scorer
    :class:`~genai_tk.agents.langchain.middleware.sensitivity_scorer.SensitivityScorer`
    :class:`~genai_tk.agents.langchain.middleware.sensitivity_scorer.DefaultScorerConfig`
    :class:`~genai_tk.agents.langchain.middleware.sensitivity_scorer.DefaultSensitivityScorer`
    :class:`~genai_tk.agents.langchain.middleware.sensitivity_scorer.SensitivityAssessment`

Sensitivity router middleware
    :class:`~genai_tk.agents.langchain.middleware.sensitivity_router_middleware.SensitivityRouterConfig`
    :class:`~genai_tk.agents.langchain.middleware.sensitivity_router_middleware.SensitivityRouterMiddleware`

Built-in middlewares (pre-existing)
    :class:`~genai_tk.agents.langchain.middleware.rich_middleware.RichToolCallMiddleware`
    :class:`~genai_tk.agents.langchain.middleware.rich_middleware.ToolCallLimitMiddleware`
    :class:`~genai_tk.agents.langchain.middleware.empty_response_retry.EmptyResponseRetryMiddleware`
"""

from genai_tk.agents.langchain.middleware.anonymization_middleware import (
    AnonymizationConfig,
    AnonymizationMiddleware,
)
from genai_tk.agents.langchain.middleware.empty_response_retry import EmptyResponseRetryMiddleware
from genai_tk.agents.langchain.middleware.presidio_detector import (
    CustomRecognizerConfig,
    DetectedEntity,
    PresidioDetector,
    PresidioDetectorConfig,
)
from genai_tk.agents.langchain.middleware.rich_middleware import RichToolCallMiddleware, ToolCallLimitMiddleware
from genai_tk.agents.langchain.middleware.sensitivity_router_middleware import (
    SensitivityRouterConfig,
    SensitivityRouterMiddleware,
)
from genai_tk.agents.langchain.middleware.sensitivity_scorer import (
    DefaultScorerConfig,
    DefaultSensitivityScorer,
    SensitivityAssessment,
    SensitivityScorer,
)

__all__ = [
    # Presidio detector
    "PresidioDetectorConfig",
    "PresidioDetector",
    "DetectedEntity",
    "CustomRecognizerConfig",
    # Anonymization
    "AnonymizationConfig",
    "AnonymizationMiddleware",
    # Sensitivity scorer
    "SensitivityScorer",
    "DefaultScorerConfig",
    "DefaultSensitivityScorer",
    "SensitivityAssessment",
    # Router
    "SensitivityRouterConfig",
    "SensitivityRouterMiddleware",
    # Pre-existing
    "RichToolCallMiddleware",
    "ToolCallLimitMiddleware",
    "EmptyResponseRetryMiddleware",
]
