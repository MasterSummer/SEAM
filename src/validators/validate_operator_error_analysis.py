"""Validation for LLM-produced operator failure analysis."""

from core.validator_engine import ValidationDict


VALID_CATEGORIES = {
    "task_schema",
    "build",
    "link",
    "install",
    "adapter",
    "runtime",
    "numerical",
    "api_policy_violation",
    "fallback",
    "environment",
    "unknown",
}
VALID_REPAIR_ROLES = {"operator_fixer", "operator_environment_fixer", "none"}
REQUIRED_STRINGS = ("root_cause", "suggested_fix", "handoff")
REQUIRED_BOOLEANS = (
    "evaluation_valid",
    "requires_clean_rerun",
    "operator_code_evidence",
)
REQUIRED_ARRAYS = (
    "failed_workloads",
    "evidence",
    "missing_evidence",
    "hypotheses",
    "next_verification",
)


def validate(data: dict[str, object]) -> ValidationDict:
    errors: list[str] = []
    category = data.get("category")
    if not isinstance(category, str) or category not in VALID_CATEGORIES:
        errors.append(f"category must be one of {sorted(VALID_CATEGORIES)}")

    repair_role = data.get("repair_role")
    if not isinstance(repair_role, str) or repair_role not in VALID_REPAIR_ROLES:
        errors.append(f"repair_role must be one of {sorted(VALID_REPAIR_ROLES)}")

    for field in REQUIRED_STRINGS:
        if not isinstance(data.get(field), str):
            errors.append(f"{field} must be a string")
    for field in REQUIRED_BOOLEANS:
        if not isinstance(data.get(field), bool):
            errors.append(f"{field} must be a boolean")
    for field in REQUIRED_ARRAYS:
        if not isinstance(data.get(field), list):
            errors.append(f"{field} must be an array")

    return {"passed": not errors, "errors": errors, "warnings": []}
