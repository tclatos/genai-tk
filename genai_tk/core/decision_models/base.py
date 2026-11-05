"""Base class for decision models (System One models)."""

from __future__ import annotations

import abc
import sys
from contextlib import contextmanager
from typing import Any, Generator

from langchain_core.runnables import RunnableConfig, RunnableSerializable
from loguru import logger
from pydantic import ConfigDict

from genai_tk.core.decision_models.types import (
    Choice,
    ChoiceAnswer,
    ClassifierRequest,
    ClassifierResponse,
    DecisionState,
    Noul,
    NoulAnswer,
    NoulCriteria,
    Score,
    ScoreAnswer,
    Usage,
)


class BaseDecisionModel(RunnableSerializable[ClassifierRequest, ClassifierResponse], abc.ABC):
    """Abstract base class for decision models.

    Inherits from LangChain RunnableSerializable to integrate cleanly with chains,
    agents, and runnables while operating on typed ClassifierRequest / ClassifierResponse.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    max_questions: int | None = None
    max_options: int | None = None
    max_levels: int | None = None
    recommended_batch_size: int | None = None

    def validate_request(self, request: ClassifierRequest) -> None:
        """Validate a request against the model's documented question, option and level limits.

        Limits are independent: a provider may accept any number of questions while
        still capping the options inside one Choice or the levels inside one Score.
        A limit left as ``None`` is not checked, because the provider publishes none.
        """
        model_name = getattr(self, "model", self.__class__.__name__)

        if self.max_questions is not None and len(request.questions) > self.max_questions:
            raise ValueError(
                f"Decision model '{model_name}' accepts at most {self.max_questions} questions per request, "
                f"but {len(request.questions)} were provided. Consider chunking questions into batches."
            )

        for question_id, question in request.questions.items():
            if isinstance(question, Choice) and self.max_options is not None:
                if len(question.criteria) > self.max_options:
                    raise ValueError(
                        f"Decision model '{model_name}' accepts at most {self.max_options} options per Choice "
                        f"question, but question '{question_id}' has {len(question.criteria)}. "
                        f"Consider narrowing the candidate set or chaining Choice questions."
                    )
            elif isinstance(question, Score) and self.max_levels is not None:
                if len(question.criteria) > self.max_levels:
                    raise ValueError(
                        f"Decision model '{model_name}' accepts at most {self.max_levels} levels per Score "
                        f"question, but question '{question_id}' has {len(question.criteria)}."
                    )

    @contextmanager
    def _trace_scope(self, request: ClassifierRequest) -> Generator[Any, None, None]:
        """Wrap decision model execution in a NeMo Relay Evaluator scope if available.

        The scope is strictly best-effort observability and must never affect the
        decision itself. Two hazards are handled explicitly:

        - **A scope open/close failure must not fail the call.** NeMo Relay keeps
          its scope stack in native thread-local storage, so a handle opened on
          one thread cannot be closed on another (``pop`` raises ``not found``).
          Concurrent decision calls from worker threads — e.g. the document
          navigator scoring several documents in parallel — hit exactly that,
          which used to abort the model call after the HTTP response had already
          been received.
        - **Exactly one ``yield``.** The previous implementation had a second
          ``yield`` in its ``except`` branch, so any failure raised while closing
          the scope made ``contextlib`` raise ``RuntimeError: generator didn't
          stop`` (or ``... after throw()``) instead of the real error, and killed
          the call. That masked message is what surfaced as an in-agent
          ``navigate_folder`` failure.
        """
        scope_cm: Any = None
        handle: Any = None
        try:
            import nemo_relay

            model_name = getattr(self, "model", self.__class__.__name__)
            scope_cm = nemo_relay.scope.scope(
                name=f"decision.{model_name}",
                scope_type=nemo_relay.ScopeType.Evaluator,
                input={"questions": list(request.questions.keys())},
                metadata={"decision_model": model_name},
            )
            handle = scope_cm.__enter__()
        except Exception as exc:  # noqa: BLE001 - tracing must never fail the call
            logger.debug("Decision-model tracing scope unavailable: {}: {}", type(exc).__name__, exc)
            scope_cm, handle = None, None

        try:
            yield handle
        finally:
            if scope_cm is not None:
                # Close with the in-flight exception (if any) so the scope's
                # status is recorded, but never let the close failure escape:
                # the return value is ignored, so the exception is not suppressed.
                exc_info = sys.exc_info()
                try:
                    scope_cm.__exit__(*exc_info) if exc_info[0] is not None else scope_cm.__exit__(None, None, None)
                except Exception as close_exc:  # noqa: BLE001 - tracing must never fail the call
                    logger.debug(
                        "Decision-model tracing scope close failed: {}: {}", type(close_exc).__name__, close_exc
                    )

    def _emit_trace_event(self, handle: Any, response: ClassifierResponse) -> None:
        """Emit a decision.verdict event if NeMo Relay is active."""
        if handle is None:
            return
        try:
            import nemo_relay

            nemo_relay.scope.event(
                "decision.verdict",
                handle=handle,
                data={q_id: ans.model_dump() for q_id, ans in response.answers.items()},
                metadata={
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
            )
        except Exception:
            pass

    @abc.abstractmethod
    def invoke(
        self,
        input: ClassifierRequest | dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        """Execute a decision request synchronously."""
        ...

    @abc.abstractmethod
    async def ainvoke(
        self,
        input: ClassifierRequest | dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        """Execute a decision request asynchronously."""
        ...

    def batch_invoke(
        self,
        input: ClassifierRequest | dict[str, Any],
        batch_size: int | None = None,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        """Execute questions in batches if they exceed batch_size, recommended_batch_size or max_questions."""
        request = input if isinstance(input, ClassifierRequest) else ClassifierRequest.model_validate(input)
        limit = batch_size or self.recommended_batch_size or self.max_questions
        if limit is None or len(request.questions) <= limit:
            return self.invoke(request, config=config, **kwargs)

        items = list(request.questions.items())
        combined_answers: dict[str, Any] = {}
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost: float = 0.0
        model_name = getattr(self, "model", self.__class__.__name__)
        last_request_id = None

        for i in range(0, len(items), limit):
            chunk = dict(items[i : i + limit])
            sub_req = ClassifierRequest(state=request.state, questions=chunk)
            sub_res = self.invoke(sub_req, config=config, **kwargs)
            combined_answers.update(sub_res.answers)
            model_name = sub_res.model
            last_request_id = sub_res.request_id
            total_input_tokens += sub_res.usage.input_tokens
            total_output_tokens += sub_res.usage.output_tokens
            if sub_res.usage.cost is not None:
                total_cost += sub_res.usage.cost

        return ClassifierResponse(
            model=model_name,
            answers=combined_answers,
            usage=Usage(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cost=total_cost if total_cost > 0 else None,
            ),
            provider=getattr(self, "provider", None),
            request_id=last_request_id,
        )

    async def abatch_invoke(
        self,
        input: ClassifierRequest | dict[str, Any],
        batch_size: int | None = None,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        """Execute questions asynchronously in batches if they exceed batch_size, recommended_batch_size or max_questions."""
        request = input if isinstance(input, ClassifierRequest) else ClassifierRequest.model_validate(input)
        limit = batch_size or self.recommended_batch_size or self.max_questions
        if limit is None or len(request.questions) <= limit:
            return await self.ainvoke(request, config=config, **kwargs)

        items = list(request.questions.items())
        combined_answers: dict[str, Any] = {}
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost: float = 0.0
        model_name = getattr(self, "model", self.__class__.__name__)
        last_request_id = None

        for i in range(0, len(items), limit):
            chunk = dict(items[i : i + limit])
            sub_req = ClassifierRequest(state=request.state, questions=chunk)
            sub_res = await self.ainvoke(sub_req, config=config, **kwargs)
            combined_answers.update(sub_res.answers)
            model_name = sub_res.model
            last_request_id = sub_res.request_id
            total_input_tokens += sub_res.usage.input_tokens
            total_output_tokens += sub_res.usage.output_tokens
            if sub_res.usage.cost is not None:
                total_cost += sub_res.usage.cost

        return ClassifierResponse(
            model=model_name,
            answers=combined_answers,
            usage=Usage(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cost=total_cost if total_cost > 0 else None,
            ),
            provider=getattr(self, "provider", None),
            request_id=last_request_id,
        )

    def decide_noul(
        self,
        state: DecisionState,
        instructions: str,
        criteria: NoulCriteria | dict[str, Any] | None = None,
        question_id: str = "decision",
    ) -> float:
        """Convenience method to evaluate a single boolean question returning probability [0.0, 1.0]."""
        request = ClassifierRequest(
            state=state,
            questions={question_id: Noul(instructions=instructions, criteria=criteria)},
        )
        response = self.invoke(request)
        answer = response.answers[question_id]
        if isinstance(answer, NoulAnswer):
            return answer.noul
        raise TypeError(f"Expected NoulAnswer, got {type(answer).__name__}")

    def decide_choice(
        self,
        state: DecisionState,
        instructions: str,
        criteria: dict[str, Any],
        question_id: str = "decision",
    ) -> ChoiceAnswer:
        """Convenience method to classify state into one of several categorical options."""
        request = ClassifierRequest(
            state=state,
            questions={question_id: Choice(instructions=instructions, criteria=criteria)},
        )
        response = self.invoke(request)
        answer = response.answers[question_id]
        if isinstance(answer, ChoiceAnswer):
            return answer
        raise TypeError(f"Expected ChoiceAnswer, got {type(answer).__name__}")

    def decide_score(
        self,
        state: DecisionState,
        instructions: str,
        criteria: list[Any],
        question_id: str = "decision",
    ) -> ScoreAnswer:
        """Convenience method to score state against an ordered rubric."""
        request = ClassifierRequest(
            state=state,
            questions={question_id: Score(instructions=instructions, criteria=criteria)},
        )
        response = self.invoke(request)
        answer = response.answers[question_id]
        if isinstance(answer, ScoreAnswer):
            return answer
        raise TypeError(f"Expected ScoreAnswer, got {type(answer).__name__}")
