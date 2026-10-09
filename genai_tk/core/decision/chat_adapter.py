"""Adapter allowing any LangChain BaseChatModel to function as a Decision Model."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from genai_tk.core.decision.base import BaseDecisionModel
from genai_tk.core.decision.types import (
    Answer,
    Choice,
    ChoiceAnswer,
    ClassifierRequest,
    ClassifierResponse,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    Usage,
    serialize_decision_state,
)


class _RawDecisionItem(BaseModel):
    """Base model for structured raw decision items returned by chat model."""

    question_id: str = Field(description="The exact identifier of the question being answered.")


class _RawNoulItem(_RawDecisionItem):
    probability: float = Field(
        description="Probability between 0.0 and 1.0 that the statement is true / yes.",
        ge=0.0,
        le=1.0,
    )


class _RawChoiceItem(_RawDecisionItem):
    selected_choice: str = Field(description="The exact choice key chosen from criteria.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(default_factory=dict)


class _RawScoreItem(_RawDecisionItem):
    score: float = Field(description="Expected score value along the rubric levels (e.g. 0.0, 1.0, 2.0).")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    probabilities: dict[int, float] = Field(default_factory=dict)


class _AllDecisionsOutput(BaseModel):
    nouls: list[_RawNoulItem] = Field(default_factory=list, description="Answers to all noul (binary) questions.")
    choices: list[_RawChoiceItem] = Field(default_factory=list, description="Answers to all choice questions.")
    scores: list[_RawScoreItem] = Field(default_factory=list, description="Answers to all score questions.")


SYSTEM_PROMPT = """You are a precise, calibrated decision engine (System One classifier).
You do not output prose or conversational text.
Evaluate the given STATE against the specified QUESTIONS.
For each question, follow the instructions and criteria carefully.
Return calibrated probabilities and choices matching the structured schema.
"""


class ChatModelDecisionModel(BaseDecisionModel):
    """Wraps a LangChain BaseChatModel to provide the BaseDecisionModel interface."""

    chat_model: BaseChatModel
    model_name: str = "chat_decision_adapter"

    def _build_prompt(self, request: ClassifierRequest) -> list[Any]:
        state_repr = serialize_decision_state(request.state)
        questions_dict = {qid: q.model_dump(exclude_none=True) for qid, q in request.questions.items()}

        user_content = (
            f"STATE:\n{json.dumps(state_repr, indent=2, ensure_ascii=False)}\n\n"
            f"QUESTIONS TO EVALUATE:\n{json.dumps(questions_dict, indent=2, ensure_ascii=False)}"
        )
        return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_content)]

    def _parse_structured_result(self, raw: _AllDecisionsOutput, request: ClassifierRequest) -> dict[str, Answer]:
        answers: dict[str, Answer] = {}
        nouls_by_id = {item.question_id: item for item in raw.nouls}
        choices_by_id = {item.question_id: item for item in raw.choices}
        scores_by_id = {item.question_id: item for item in raw.scores}

        for q_id, q in request.questions.items():
            if isinstance(q, Noul):
                noul_raw = nouls_by_id.get(q_id) or (raw.nouls[0] if len(raw.nouls) == 1 else None)
                prob = noul_raw.probability if noul_raw else 0.5
                answers[q_id] = NoulAnswer(noul=prob)
            elif isinstance(q, Choice):
                choice_raw = choices_by_id.get(q_id) or (raw.choices[0] if len(raw.choices) == 1 else None)
                keys = list(q.criteria.keys())
                choice_key = (
                    choice_raw.selected_choice
                    if choice_raw and choice_raw.selected_choice in keys
                    else (keys[0] if keys else "unknown")
                )
                probs = (
                    choice_raw.probabilities
                    if choice_raw and choice_raw.probabilities
                    else {k: (1.0 if k == choice_key else 0.0) for k in keys}
                )
                conf = choice_raw.confidence if choice_raw else 0.9
                answers[q_id] = ChoiceAnswer(choice=choice_key, confidence=conf, probabilities=probs)
            elif isinstance(q, Score):
                score_raw = scores_by_id.get(q_id) or (raw.scores[0] if len(raw.scores) == 1 else None)
                n_levels = len(q.criteria)
                sc = score_raw.score if score_raw else 0.0
                legend = {i: q.criteria[i] for i in range(n_levels)}
                probs = (
                    score_raw.probabilities
                    if score_raw and score_raw.probabilities
                    else {i: (1.0 if i == int(round(sc)) else 0.0) for i in range(n_levels)}
                )
                conf = score_raw.confidence if score_raw else 0.9
                answers[q_id] = ScoreAnswer(score=sc, confidence=conf, probabilities=probs, legend=legend)
        return answers

    def invoke(
        self,
        input: ClassifierRequest | dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        request = input if isinstance(input, ClassifierRequest) else ClassifierRequest.model_validate(input)
        self.validate_question_count(request)
        messages = self._build_prompt(request)

        try:
            structured_llm = self.chat_model.with_structured_output(_AllDecisionsOutput)
            result: _AllDecisionsOutput = structured_llm.invoke(messages, config=config)  # type: ignore[assignment]
        except (NotImplementedError, AttributeError):
            # Fallback for models without native with_structured_output (e.g. fake models or simple mock LLMs)
            schema_json = json.dumps(_AllDecisionsOutput.model_json_schema(), indent=2)
            prompt_with_schema = messages + [
                HumanMessage(content=f"Respond strictly with valid JSON conforming to this schema:\n{schema_json}")
            ]
            response = self.chat_model.invoke(prompt_with_schema, config=config)
            content = getattr(response, "content", "")
            if isinstance(content, list):
                content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
            try:
                # Clean possible markdown fences
                cleaned = content.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                parsed_dict = json.loads(cleaned)
                result = _AllDecisionsOutput.model_validate(parsed_dict)
            except Exception:
                # Default empty output if mock/fake returns raw text
                result = _AllDecisionsOutput()

        answers = self._parse_structured_result(result, request)

        return ClassifierResponse(
            model=getattr(self.chat_model, "model_name", self.model_name),
            answers=answers,
            usage=Usage(input_tokens=0, output_tokens=0),
            provider="chat_adapter",
        )

    async def ainvoke(
        self,
        input: ClassifierRequest | dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        request = input if isinstance(input, ClassifierRequest) else ClassifierRequest.model_validate(input)
        self.validate_question_count(request)
        messages = self._build_prompt(request)

        try:
            structured_llm = self.chat_model.with_structured_output(_AllDecisionsOutput)
            result: _AllDecisionsOutput = await structured_llm.ainvoke(messages, config=config)  # type: ignore[assignment]
        except (NotImplementedError, AttributeError):
            schema_json = json.dumps(_AllDecisionsOutput.model_json_schema(), indent=2)
            prompt_with_schema = messages + [
                HumanMessage(content=f"Respond strictly with valid JSON conforming to this schema:\n{schema_json}")
            ]
            response = await self.chat_model.ainvoke(prompt_with_schema, config=config)
            content = getattr(response, "content", "")
            if isinstance(content, list):
                content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
            try:
                cleaned = content.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                parsed_dict = json.loads(cleaned)
                result = _AllDecisionsOutput.model_validate(parsed_dict)
            except Exception:
                result = _AllDecisionsOutput()

        answers = self._parse_structured_result(result, request)

        return ClassifierResponse(
            model=getattr(self.chat_model, "model_name", self.model_name),
            answers=answers,
            usage=Usage(input_tokens=0, output_tokens=0),
            provider="chat_adapter",
        )
