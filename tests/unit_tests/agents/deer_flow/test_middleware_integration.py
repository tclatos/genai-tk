"""Prove the shared LangChain anonymization/routing middleware runs unmodified in DeerFlow.

DeerFlow profiles reuse the exact same ``AgentMiddleware`` classes as
LangChain agents (see docs/middleware-pii-and-routing.md). This module
verifies the DeerFlow profile's ``middlewares`` field accepts the same
``class`` + kwargs shape, and that the runtime middleware builder produces
real, correctly configured middleware instances — no DeerFlow-specific
middleware wrapper is needed.
"""

from genai_tk.agents.deer_flow.profile import DeerFlowProfile
from genai_tk.agents.deer_flow.runtime import build_cli_middlewares
from genai_tk.agents.langchain.middleware import AnonymizationMiddleware, SensitivityRouterMiddleware
from genai_tk.agents.langchain.middleware.rich_middleware import RichToolCallMiddleware


def test_deerflow_profile_has_harness_discriminator() -> None:
    """DeerFlowProfile always self-identifies as the 'deerflow' harness."""
    profile = DeerFlowProfile(name="Research Assistant")
    assert profile.harness == "deerflow"


def test_deerflow_profile_parses_middleware_config_with_kwargs() -> None:
    """DeerFlowProfile.middlewares accepts class + kwargs, same shape as LangChain profiles."""
    profile = DeerFlowProfile.model_validate(
        {
            "name": "Privacy-Safe Research",
            "middlewares": [
                {
                    "class": "genai_tk.agents.langchain.middleware.anonymization_middleware.AnonymizationMiddleware",
                    "analyzed_fields": ["PERSON", "EMAIL_ADDRESS"],
                    "faker_seed": 42,
                },
            ],
        }
    )
    assert profile.middlewares[0].class_path.endswith(".AnonymizationMiddleware")
    assert profile.middlewares[0].extra_kwargs["faker_seed"] == 42


def test_build_cli_middlewares_instantiates_shared_middleware(fake_llm_id: str) -> None:
    """The DeerFlow runtime middleware builder produces the exact same middleware classes as LangChain agents."""
    profile = DeerFlowProfile.model_validate(
        {
            "name": "Privacy-Safe Research",
            "middlewares": [
                {
                    "class": "genai_tk.agents.langchain.middleware.anonymization_middleware.AnonymizationMiddleware",
                    "faker_seed": 42,
                },
                {
                    "class": (
                        "genai_tk.agents.langchain.middleware.sensitivity_router_middleware.SensitivityRouterMiddleware"
                    ),
                    "safe_llm": fake_llm_id,
                },
            ],
        }
    )

    middlewares = build_cli_middlewares(profile.middlewares)

    assert any(isinstance(m, AnonymizationMiddleware) for m in middlewares)
    assert any(isinstance(m, SensitivityRouterMiddleware) for m in middlewares)
    # RichToolCallMiddleware is always prepended for DeerFlow runs, same as LangChain agents.
    assert any(isinstance(m, RichToolCallMiddleware) for m in middlewares)
