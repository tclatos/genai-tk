"""Base class for decision models (System One models)."""

from __future__ import annotations

import abc
from contextlib import contextmanager
from typing import Any, Generator

from langchain_core.runnables import RunnableConfig, RunnableSerializable
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

    def validate_question_count(self, request: ClassifierRequest) -> None:
        """Validate that the number of questions does not exceed max_questions."""
        if self.max_questions is not None and len(request.questions) > self.max_questions:
            model_name = getattr(self, "model", self.__class__.__name__)
            raise ValueError(
                f"Decision model '{model_name}' accepts at most {self.max_questions} questions per request, "
                f"but {len(request.questions)} were provided. Consider chunking questions into batches."
            )

    @contextmanager
    def _trace_scope(self, request: ClassifierRequest) -> Generator[Any, None, None]:
        """Wrap decision model execution in a NeMo Relay Evaluator scope if available."""
        try:
            import nemo_relay

            model_name = getattr(self, "model", self.__class__.__name__)
            with nemo_relay.scope.scope(
                name=f"decision.{model_name}",
                scope_type=nemo_relay.ScopeType.Evaluator,
                input={"questions": list(request.questions.keys())},
                metadata={"decision_model": model_name},
            ) as handle:
                yield handle
        except (ImportError, Exception):
            yield None

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
        """Execute questions in batches if they exceed batch_size (or self.max_questions)."""
        request = input if isinstance(input, ClassifierRequest) else ClassifierRequest.model_validate(input)
        limit = batch_size or self.max_questions
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
        """Execute questions asynchronously in batches if they exceed batch_size (or self.max_questions)."""
        request = input if isinstance(input, ClassifierRequest) else ClassifierRequest.model_validate(input)
        limit = batch_size or self.max_questions
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
