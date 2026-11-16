from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import NamedTuple, cast

from core.workflow_condition_parser import (
    ConditionEvaluationError as ConditionEvaluationError,
    ConditionEvaluationMode as ConditionEvaluationMode,
    ConditionValue as ConditionValue,
    evaluate_boolean_expression as evaluate_boolean_expression,
)


class ConditionRequest(NamedTuple):
    condition: str
    state: Mapping[str, ConditionValue]
    workflow_globals: Mapping[str, ConditionValue]
    context: Mapping[str, ConditionValue]
    loop_variables: Mapping[str, ConditionValue]
    loop_state: Mapping[str, ConditionValue]
    step_outputs: Mapping[str, ConditionValue]


class ConditionDecision(NamedTuple):
    matched: bool
    evaluation_error: str | None


def replace_condition_field_references(
    expression: str,
    sources: tuple[Mapping[str, ConditionValue], ...],
    environment: dict[str, ConditionValue],
) -> str:
    """Bind $.field.path references as values instead of source-code text."""
    path_pattern = re.compile(r"\$\.([\w]+(?:\.[\w]+)*)")
    missing = object()
    next_index = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal next_index
        path = match.group(1).split(".")
        value: ConditionValue = ".".join(path)
        for source in sources:
            candidate: object = source
            for part in path:
                if not isinstance(candidate, Mapping) or part not in candidate:
                    candidate = missing
                    break
                candidate = candidate[part]
            if candidate is not missing:
                value = cast(ConditionValue, candidate)
                break

        while True:
            name = f"__seam_condition_field_{next_index}"
            next_index += 1
            if name not in environment and name not in expression:
                break
        environment[name] = value
        return name

    return path_pattern.sub(replace, expression)


def select_condition_evaluation_mode(
    workflow_globals: Mapping[str, ConditionValue],
) -> ConditionEvaluationMode:
    return (
        ConditionEvaluationMode.ACTIVE_V3
        if isinstance(workflow_globals.get("review_fail_closed"), bool)
        else ConditionEvaluationMode.LEGACY
    )


def evaluate_condition(
    request: ConditionRequest,
    resolve_template: Callable[[str], ConditionValue],
    resolve_expression: Callable[[str], ConditionValue],
) -> ConditionDecision:
    pattern = re.compile(r"\$\{([^{}]+)\}")
    if pattern.fullmatch(request.condition):
        resolved = resolve_template(request.condition)
    elif "${" in request.condition:
        resolved = pattern.sub(
            lambda match: json.dumps(
                resolve_expression(match.group(1).strip()),
                ensure_ascii=False,
                default=str,
            ),
            request.condition,
        )
    else:
        resolved = resolve_template(request.condition)
    if not isinstance(resolved, str):
        return ConditionDecision(bool(resolved), None)

    environment = {
        **request.state,
        **request.workflow_globals,
        **request.context,
        **request.loop_state,
        **request.loop_variables,
        **request.step_outputs,
    }
    expression = replace_condition_field_references(
        resolved,
        (
            request.step_outputs,
            request.workflow_globals,
            request.context,
            request.loop_state,
        ),
        environment,
    )
    lowered = expression.lower()
    if lowered in ("true", "1"):
        return ConditionDecision(True, None)
    if lowered in ("false", "0", ""):
        return ConditionDecision(False, None)
    try:
        mode = select_condition_evaluation_mode(request.workflow_globals)
        return ConditionDecision(
            evaluate_boolean_expression(expression, environment, mode), None
        )
    except (ConditionEvaluationError, IndexError, TypeError, ValueError) as exc:
        return ConditionDecision(True, str(exc))
