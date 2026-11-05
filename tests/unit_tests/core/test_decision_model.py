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


@pytest.mark.unit
def test_max_options_validation() -> None:
    """A Choice wider than max_options is rejected; one exactly at the cap passes."""
    fake = FakeDecisionModel(max_options=255)

    at_cap = Choice(instructions="Pick one", criteria={f"opt_{i}": f"Option {i}" for i in range(255)})
    fake.validate_request(ClassifierRequest(state="x", questions={"c": at_cap}))

    over_cap = Choice(instructions="Pick one", criteria={f"opt_{i}": f"Option {i}" for i in range(256)})
    with pytest.raises(ValueError, match="at most 255 options per Choice"):
        fake.validate_request(ClassifierRequest(state="x", questions={"c": over_cap}))


@pytest.mark.unit
def test_max_levels_validation() -> None:
    """A Score with more levels than max_levels is rejected; one exactly at the cap passes."""
    fake = FakeDecisionModel(max_levels=10)

    at_cap = Score(instructions="Rate", criteria=[f"level {i}" for i in range(10)])
    fake.validate_request(ClassifierRequest(state="x", questions={"s": at_cap}))

    over_cap = Score(instructions="Rate", criteria=[f"level {i}" for i in range(11)])
    with pytest.raises(ValueError, match="at most 10 levels per Score"):
        fake.validate_request(ClassifierRequest(state="x", questions={"s": over_cap}))


@pytest.mark.unit
def test_option_and_level_limits_are_independent_of_question_count() -> None:
    """A model with no question cap still enforces its per-question option and level caps."""
    fake = FakeDecisionModel(max_questions=None, max_options=255, max_levels=10)

    many_questions = {f"q_{i}": Noul(instructions=f"Is {i} true?") for i in range(300)}
    fake.validate_request(ClassifierRequest(state="x", questions=many_questions))

    with pytest.raises(ValueError, match="at most 255 options per Choice"):
        fake.validate_request(
            ClassifierRequest(
                state="x",
                questions={"wide": Choice(instructions="Pick", criteria={f"o{i}": f"Option {i}" for i in range(300)})},
            )
        )


@pytest.mark.unit
def test_jev_question_count_is_not_capped_but_clef_is() -> None:
    """The registry distinguishes Jev's context-bounded question count from Clef's hard cap of 64."""
    known = DecisionModelFactory.get_known_models_dict()

    jev = known["jev@openrouter"]
    assert jev.max_questions is None, "Jev documents no cap on questions per request"
    assert jev.max_options == 255
    assert jev.max_levels == 10
    assert jev.recommended_batch_size == 32

    clef = known["clef_flash@openrouter"]
    assert clef.max_questions == 64, "Clef's 64-question ceiling is a hard API constraint"
    assert clef.max_options == 255
    assert clef.max_levels == 10

    # Constructed without contacting the provider: only local validation runs.
    jev_model = get_decision_model("jev@openrouter")
    jev_model.validate_request(
        ClassifierRequest(state="x", questions={f"q_{i}": Noul(instructions="?") for i in range(200)})
    )

    clef_model = get_decision_model("clef_flash@openrouter")
    with pytest.raises(ValueError, match="at most 64 questions"):
        clef_model.validate_request(
            ClassifierRequest(state="x", questions={f"q_{i}": Noul(instructions="?") for i in range(65)})
        )


@pytest.mark.unit
def test_recommended_batch_size_chunks_when_there_is_no_hard_cap() -> None:
    """recommended_batch_size drives batch_invoke chunking for models with no question cap."""
    fake = FakeDecisionModel(max_questions=None, recommended_batch_size=32)
    questions = {f"q_{i}": Noul(instructions=f"Is {i} true?") for i in range(100)}
    req = ClassifierRequest(state="Test policy batching", questions=questions)

    resp = fake.batch_invoke(req)
    assert len(resp.answers) == 100
    # Tokens: 10 per sub-request * 4 chunks (32 + 32 + 32 + 4) = 40
    assert resp.usage.input_tokens == 40


@pytest.mark.unit
def test_registry_limits_are_threaded_into_instances() -> None:
    """Registry limits reach the constructed model, so validation uses configured values."""
    fake = get_decision_model("fake")
    assert fake.max_questions == 32
    assert fake.max_options == 255
    assert fake.max_levels == 10
    assert fake.recommended_batch_size == 32

    with pytest.raises(ValueError, match="at most 255 options per Choice"):
        fake.validate_request(
            ClassifierRequest(
                state="x",
                questions={"wide": Choice(instructions="Pick", criteria={f"o{i}": f"Opt {i}" for i in range(256)})},
            )
        )


# ── Tracing must never break a decision ──────────────────────────────────────


class _RaisingDecisionModel(FakeDecisionModel):
    """Fake model whose body raises after the tracing scope is open."""

    def invoke(self, input, config=None, **kwargs):  # type: ignore[no-untyped-def]
        request = ClassifierRequest.model_validate(input)
        with self._trace_scope(request):
            raise ValueError("decision backend unavailable")


def _decision_request() -> ClassifierRequest:
    return ClassifierRequest(state="x", questions={"q": Noul(instructions="Is it relevant?")})


@pytest.mark.unit
def test_tracing_scope_close_failure_does_not_fail_the_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing scope close must not abort the call.

    NeMo Relay's scope stack is native thread-local, so a handle pushed on one
    thread cannot be popped on another. The close failure used to be re-raised as
    ``RuntimeError: generator didn't stop`` (the scope contextmanager yielded
    twice), which killed a document navigation after the decision had already
    been computed.
    """
    pytest.importorskip("nemo_relay")
    import nemo_relay.scope as scope_module

    def failing_pop(handle, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("not found: scope handle not found")

    monkeypatch.setattr(scope_module, "pop", failing_pop, raising=True)

    response = FakeDecisionModel(fixed_noul=0.9).invoke(_decision_request())
    assert response.nouls["q"].noul == 0.9


@pytest.mark.unit
def test_tracing_scope_does_not_mask_a_body_exception() -> None:
    """The real error must propagate, not a contextlib 'generator didn't stop'."""
    pytest.importorskip("nemo_relay")
    with pytest.raises(ValueError, match="decision backend unavailable"):
        _RaisingDecisionModel().invoke(_decision_request())


@pytest.mark.unit
def test_tracing_scope_opens_exactly_once_and_closes_cleanly() -> None:
    """One yield, one close: the handle is real and re-entering is not possible."""
    pytest.importorskip("nemo_relay")
    model = FakeDecisionModel()
    scope = model._trace_scope(_decision_request())
    handle = scope.__enter__()
    assert handle is not None
    scope.__exit__(None, None, None)  # must not raise


@pytest.mark.unit
def test_concurrent_decision_calls_from_worker_threads_all_succeed() -> None:
    """Concurrent decision calls across threads keep working with tracing active.

    This is the navigator's shape: one decision call per document on a worker
    thread, all sharing the process-wide Relay scope stack.
    """
    pytest.importorskip("nemo_relay")
    from concurrent.futures import ThreadPoolExecutor

    model = FakeDecisionModel(fixed_noul=0.8)

    def call(_: int) -> float:
        return model.invoke(_decision_request()).nouls["q"].noul

    with ThreadPoolExecutor(max_workers=3) as pool:
        scores = list(pool.map(call, range(6)))

    assert scores == [0.8] * 6
