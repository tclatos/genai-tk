"""Unit tests for Decision / System One models and DecisionModelFactory."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from genai_tk.core.decision_models import (
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
def test_decision_model_factory_resolution_and_instantiation() -> None:
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


@pytest.mark.unit
def test_max_questions_validation() -> None:
    fake = FakeDecisionModel(max_questions=5)
    questions = {f"q_{i}": Noul(instructions=f"Is {i} > 0?") for i in range(10)}
    req = ClassifierRequest(state="Test overflow", questions=questions)

    with pytest.raises(ValueError, match="accepts at most 5 questions"):
        fake.invoke(req)


@pytest.mark.unit
def test_batch_invoke_chunking() -> None:
    fake = FakeDecisionModel(max_questions=5)
    questions = {f"q_{i}": Noul(instructions=f"Is {i} > 0?") for i in range(12)}
    req = ClassifierRequest(state="Test batching", questions=questions)

    resp = fake.batch_invoke(req)
    assert len(resp.answers) == 12
    assert all(f"q_{i}" in resp.answers for i in range(12))
    # Tokens: 10 per call * 3 calls (5 + 5 + 2) = 30
    assert resp.usage.input_tokens == 30


@pytest.mark.asyncio
@pytest.mark.unit
async def test_abatch_invoke_chunking() -> None:
    fake = FakeDecisionModel(max_questions=4)
    questions = {f"q_{i}": Noul(instructions=f"Is {i} > 0?") for i in range(9)}
    req = ClassifierRequest(state="Test async batching", questions=questions)

    resp = await fake.abatch_invoke(req)
    assert len(resp.answers) == 9
    assert all(f"q_{i}" in resp.answers for i in range(9))


@pytest.mark.unit
def test_edenai_decision_model_wire_format() -> None:
    """Test EdenAIDecisionModel builds payload and parses response correctly."""
    from unittest.mock import MagicMock, patch

    from genai_tk.core.decision_models.edenai import EdenAIDecisionModel

    model = EdenAIDecisionModel(
        model="typesafe/jev-latest",
        api_key="mock-key",  # type: ignore[arg-type]
    )

    req = ClassifierRequest(
        state="I was charged twice for my subscription. Please refund me.",
        questions={
            "is_billing": Noul(instructions="Is this message about billing?"),
        },
    )

    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "id": "eden-123",
        "model": "typesafe/jev-latest",
        "answers": {
            "is_billing": {"noul": 0.98},
        },
        "usage": {"input_tokens": 12, "output_tokens": 1},
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        res = model.invoke(req)

    mock_post.assert_called_once()
    payload = mock_post.call_args.kwargs["json"]
    assert payload["model"] == "typesafe/jev-latest"
    assert "state" in payload
    assert payload["questions"]["is_billing"]["type"] == "noul"
    assert res.nouls["is_billing"].noul == 0.98


@pytest.mark.unit
def test_laya_decision_model_server_mode_wire_format() -> None:
    """Test LayaDecisionModel in server mode targeting laya-serve /v1/systemone."""
    from unittest.mock import MagicMock, patch

    from genai_tk.core.decision_models.laya import LayaDecisionModel

    model = LayaDecisionModel(
        model="laya",
        mode="server",
        server_url="http://localhost:8000/v1/systemone",
    )

    req = ClassifierRequest(
        state="Server error 500 on checkout",
        questions={
            "dept": Choice(
                instructions="Which dept?",
                criteria={"billing": "Payment", "tech": "Bugs"},
            ),
        },
    )

    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "id": "laya-1",
        "model": "laya",
        "answers": {
            "dept": {"choice": "tech", "confidence": 0.94},
        },
        "usage": {"input_tokens": 15, "output_tokens": 2},
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        res = model.invoke(req)

    mock_post.assert_called_once()
    assert res.choices["dept"].choice == "tech"
    assert res.choices["dept"].confidence == 0.94


@pytest.mark.unit
def test_openai_decision_model_adaptation() -> None:
    """Test OpenAIDecisionModel adapts ClassifierRequest to langchain_openai OpenAIDecisions."""
    from unittest.mock import MagicMock, patch

    from genai_tk.core.decision_models.openai import OpenAIDecisionModel

    model = OpenAIDecisionModel(model="gpt-6-luna", api_key="sk-fake")  # type: ignore[arg-type]

    req = ClassifierRequest(
        state="Deploy failed twice and customers are seeing 500 errors.",
        questions={
            "urgent": Noul(instructions="Does this need attention now?"),
            "team": Choice(
                instructions="Which team?",
                criteria={"infra": "Deploys, availability", "billing": "Payments"},
            ),
            "severity": Score(
                instructions="Impact level",
                criteria=["Cosmetic", "Degraded", "Outage"],
            ),
        },
    )

    mock_lc_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.model = "gpt-6-luna"
    mock_resp.request_id = "req-123"

    pred_ans = MagicMock()
    pred_ans.probability = 0.96
    mock_resp.predicates = {"urgent": pred_ans}

    choice_ans = MagicMock()
    choice_ans.choice = "infra"
    choice_ans.confidence = 0.92
    choice_ans.probabilities = {"infra": 0.92, "billing": 0.08}
    mock_resp.choices = {"team": choice_ans}

    score_ans = MagicMock()
    score_ans.score = 2.0
    score_ans.confidence = 0.88
    score_ans.probabilities = {0: 0.02, 1: 0.10, 2: 0.88}
    score_ans.legend = {0: "Cosmetic", 1: "Degraded", 2: "Outage"}
    mock_resp.scores = {"severity": score_ans}

    mock_resp.usage.input_tokens = 32
    mock_resp.usage.output_tokens = 3

    mock_lc_client.invoke.return_value = mock_resp

    with patch.object(model, "_get_client", return_value=mock_lc_client):
        res = model.invoke(req)

    mock_lc_client.invoke.assert_called_once()
    assert res.model == "gpt-6-luna"
    assert res.nouls["urgent"].noul == 0.96
    assert res.choices["team"].choice == "infra"
    assert res.choices["team"].confidence == 0.92
    assert res.scores["severity"].score == 2.0
    assert res.scores["severity"].legend[2] == "Outage"
    assert res.usage.input_tokens == 32


@pytest.mark.unit
def test_laya_decision_model_router_mode_e2e() -> None:
    """End-to-end test with Laya Router mode."""
    pytest.importorskip("laya")
    from genai_tk.core.factories.decision_model_factory import get_decision_model

    dm = get_decision_model("laya@laya")
    req = ClassifierRequest(
        state="Hi, we were billed twice for March. Please refund the duplicate today.",
        questions={
            "department": Choice(
                instructions="Which department should handle this?",
                criteria={
                    "billing": "invoices, payments, refunds",
                    "technical": "bugs, outages, system errors",
                },
            ),
            "refund_requested": Noul(instructions="Does the user ask for refund?"),
        },
    )
    res = dm.invoke(req)
    assert res.provider == "laya"
    assert "department" in res.choices
    assert res.choices["department"].choice == "billing"
    assert "refund_requested" in res.nouls
    assert 0.0 <= res.nouls["refund_requested"].noul <= 1.0
