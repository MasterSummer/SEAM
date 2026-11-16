from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from core.workflow_condition_policy import (
    ConditionEvaluationError,
    ConditionValue,
    evaluate_boolean_expression,
    replace_condition_field_references,
)


class StopCondition(NamedTuple):
    condition: str
    status: str


class StopDecision(NamedTuple):
    status: str | None
    evaluation_errors: tuple[str, ...]


def select_stop_status(
    conditions: Sequence[StopCondition],
    loop_state: Mapping[str, ConditionValue],
    workflow_globals: Mapping[str, ConditionValue],
) -> StopDecision:
    errors: list[str] = []
    for condition in conditions:
        environment: dict[str, ConditionValue] = {
            **workflow_globals,
            **loop_state,
        }
        expression = replace_condition_field_references(
            condition.condition, (loop_state, workflow_globals), environment
        )
        if expression.lower() == "true":
            return StopDecision(condition.status, tuple(errors))
        if expression.lower() == "false":
            continue
        try:
            if evaluate_boolean_expression(expression, environment):
                return StopDecision(condition.status, tuple(errors))
        except (ConditionEvaluationError, IndexError, TypeError, ValueError) as exc:
            errors.append(f"{condition.condition}: {exc}")
    return StopDecision(None, tuple(errors))
