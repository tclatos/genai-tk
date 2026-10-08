"""Decision / System One model schemas and serialization.

This module defines Pydantic v2 schemas for structured decision models (System One),
supporting binary judgments (Noul), categorical choices (Choice), and ordinal rubric
scoring (Score). The schemas match the wire contract of OpenRouter's /api/alpha/decisions,
/systemone, and TypeSafe System One.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal, TypeAlias

from langchain_core.messages import BaseMessage, convert_to_openai_messages
from pydantic import BaseModel, ConfigDict, Field, JsonValue

# --- State types & serialization ---

DecisionState: TypeAlias = Any


def _serialize_state_value(value: object) -> JsonValue:
    if isinstance(value, BaseMessage):
        return _serialize_state_value(convert_to_openai_messages(value))
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Decision state object keys must be strings.")
        return {key: _serialize_state_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_serialize_state_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported decision state value: {type(value).__name__}.")


def serialize_decision_state(state: DecisionState) -> JsonValue:
    """Recursively convert LangChain messages and objects inside decision state to JSON-safe data."""
    if state is None or isinstance(state, (int, float, bool)):
        raise TypeError(
            "Decision state must be a string, dict, list, BaseMessage, or sequence of BaseMessage objects."
        )
    return _serialize_state_value(state)


# --- Questions ---

_QuestionContent: TypeAlias = str | dict[str, JsonValue] | list[JsonValue]


class NoulCriteria(BaseModel):
    """Descriptions for the true and false outcomes of a Noul question."""

    model_config = ConfigDict(populate_by_name=True)

    true: JsonValue = None
    false: JsonValue = None


class Noul(BaseModel):
    """Binary decision asking whether an assertion is true, returning the probability of yes."""

    type: Literal["noul"] = "noul"
    instructions: _QuestionContent
    criteria: NoulCriteria | dict[str, JsonValue] | None = None


class Choice(BaseModel):
    """Categorical decision choosing one label among supplied candidate criteria."""

    type: Literal["choice"] = "choice"
    instructions: _QuestionContent
    criteria: dict[str, JsonValue] = Field(min_length=1)


class Score(BaseModel):
    """Ordinal decision scoring the state against an ordered rubric list."""

    type: Literal["score"] = "score"
    instructions: _QuestionContent
    criteria: list[JsonValue] = Field(min_length=2)


Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]


class ClassifierRequest(BaseModel):
    """Request payload for a decision model invocation."""

    state: DecisionState
    questions: dict[str, Question]


# --- Answers ---


class NoulAnswer(BaseModel):
    """Answer for a binary Noul question."""

    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0.0, le=1.0)


class ChoiceAnswer(BaseModel):
    """Answer for a categorical Choice question."""

    type: Literal["choice"] = "choice"
    choice: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(default_factory=dict)


class ScoreAnswer(BaseModel):
    """Answer for an ordinal Score question."""

    type: Literal["score"] = "score"
    score: float
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    probabilities: dict[int, float] = Field(default_factory=dict)
    legend: dict[int, JsonValue] = Field(default_factory=dict)


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    """Token usage reported for a decision request."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None


class ClassifierResponse(BaseModel):
    """Unified response returned by a decision model."""

    model: str
    answers: dict[str, Answer]
    usage: Usage = Field(default_factory=Usage)
    provider: str | None = None
    request_id: str | None = None

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        """Convenience filter for binary Noul answers."""
        return {k: v for k, v in self.answers.items() if isinstance(v, NoulAnswer)}

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        """Convenience filter for categorical Choice answers."""
        return {k: v for k, v in self.answers.items() if isinstance(v, ChoiceAnswer)}

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        """Convenience filter for ordinal Score answers."""
        return {k: v for k, v in self.answers.items() if isinstance(v, ScoreAnswer)}
