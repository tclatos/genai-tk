"""Unit tests for Decision / System One models and DecisionModelFactory."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from genai_tk.core.decision import (
    Choice,
    ChoiceAnswer,
    ClassifierRequest,
    ClassifierResponse,
    FakeDecisionModel,
    Noul,
    Score,
    ScoreAnswer,
    serialize_decision_state,
)
from genai_tk.core.factories import (
    DecisionModelFactory,
    get_decision_model,
    get_decision_model_from_chat_model,
)


@pytest.mark.unit
def test_decision_state_serialization() -> None:
    # String state
    assert serialize_decision_state("hello") == "hello"

    # Dict state
    assert serialize_decision_state({"key": "val", "num": 42}) == {"key": "val", "num": 42}

    # LangChain BaseMessage state
    msg = HumanMessage(content="Hello world")
    serialized = serialize_decision_state(msg)
    assert isinstance(serialized, dict)
    assert serialized.get("content") == "Hello world"
    assert serialized.get("role") == "user"


@pytest.mark.unit
def test_decision_schemas_and_discriminators() -> None:
    req = ClassifierRequest(
        state="Checkout is failing",
        questions={
            "urgent": Noul(instructions="Is this time-sensitive?"),
            "category": Choice(
                instructions="Select category",
                criteria={"billing": "Payment errors", "tech": "Bugs"},
            ),
            "severity": Score(
                instructions="Impact level",
                criteria=["Low", "Medium", "High"],
            ),
        },
    )
    dumped = req.model_dump()
    assert dumped["questions"]["urgent"]["type"] == "noul"
    assert dumped["questions"]["category"]["type"] == "choice"
    assert dumped["questions"]["severity"]["type"] == "score"


@pytest.mark.unit
def test_fake_decision_model_invocation() -> None:
    fake = FakeDecisionModel(fixed_noul=0.92, fixed_choice="billing", fixed_score=2.0)
    req = ClassifierRequest(
        state="Order error",
        questions={
            "is_bug": Noul(instructions="Is it a bug?"),
            "team": Choice(instructions="Assign team", criteria={"billing": "Pay", "ops": "Ops"}),
            "level": Score(instructions="Score", criteria=["C1", "C2", "C3"]),
        },
    )
    resp = fake.invoke(req)
    assert isinstance(resp, ClassifierResponse)
    assert resp.model == "fake_decision@fake"

    # Check answers
    assert resp.nouls["is_bug"].noul == 0.92
    assert resp.choices["team"].choice == "billing"
    assert resp.choices["team"].confidence == 0.95
    assert resp.scores["level"].score == 2.0


@pytest.mark.unit
def test_fake_decision_model_convenience_methods() -> None:
    fake = FakeDecisionModel(fixed_noul=0.75, fixed_choice="tech", fixed_score=1.5)

    noul_val = fake.decide_noul("Server 500", "Is it urgent?")
    assert noul_val == 0.75

    choice_val = fake.decide_choice("Server 500", "Choose team", {"tech": "Tech team", "other": "Other"})
    assert isinstance(choice_val, ChoiceAnswer)
    assert choice_val.choice == "tech"

    score_val = fake.decide_score("Server 500", "Rate frustration", ["Calm", "Upset", "Angry"])
    assert isinstance(score_val, ScoreAnswer)
    assert score_val.score == 1.5


@pytest.mark.unit
def test_decision_factory_resolution_and_instantiation() -> None:
    # Test known models
    known = DecisionModelFactory.get_known_models()
    assert len(known) >= 5
    ids = [m.id for m in known]
    assert "clef_flash@openrouter" in ids
    assert "jev@openrouter" in ids
    assert "fake_decision@fake" in ids

    # Test resolving tags (under pytest profile, default resolves to fake_decision@fake)
    resolved_default = DecisionModelFactory.resolve_model_id("default")
    assert resolved_default == "fake_decision@fake"

    resolved_fake = DecisionModelFactory.resolve_model_id("fake")
    assert resolved_fake == "fake_decision@fake"

    # Test create via factory
    dec = get_decision_model("fake")
    assert isinstance(dec, FakeDecisionModel)


@pytest.mark.unit
def test_chat_model_adapter_instantiation(fake_llm) -> None:
    adapter = get_decision_model_from_chat_model(fake_llm)
    assert adapter is not None
    assert hasattr(adapter, "chat_model")
