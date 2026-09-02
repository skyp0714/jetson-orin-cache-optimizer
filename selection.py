"""Power-budget-specific selections from evaluated cache candidates.

The optimizer produces one common set of full-cache Pareto candidates.  This
module applies policy constraints afterwards so that changing a power budget
does not require another Accel-Sim run.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .models import Evaluation


MIN_ENERGY_ISO_PERFORMANCE = "min_energy_iso_performance"
MAX_PERFORMANCE_ISO_ENERGY = "max_performance_iso_energy"
BALANCED_KNEE = "balanced_knee"

_SELECTORS = (
    MIN_ENERGY_ISO_PERFORMANCE,
    MAX_PERFORMANCE_ISO_ENERGY,
    BALANCED_KNEE,
)
_PARETO_FULL_CACHE = "pareto_full_cache"
_TOLERANCE = 1e-12


@dataclasses.dataclass
class SelectionResult:
    """One policy-specific optimum under a named power budget."""

    technology: str
    budget: dict[str, Any]
    selector: str
    status: str
    evaluation: Evaluation | None
    reasons: list[str]
    observed_power: float | None
    observed_unit: str


def select_power_budget_optima(
    evaluations: Iterable[Evaluation],
    power_budgets: Sequence[Mapping[str, Any]],
    constraints: Mapping[str, Any],
) -> list[SelectionResult]:
    """Select three useful operating points for every technology and budget.

    A technology's ``pareto_full_cache`` candidates take precedence when they
    exist.  This prevents legacy objective-specific searches from being mixed
    into a new Pareto run, while retaining backwards compatibility for old
    result directories.

    ``min_energy_iso_performance`` preserves the configured runtime tolerance
    within the SRAM-L1/target-L2 retrofit family.
    ``max_performance_iso_energy`` preserves the configured energy tolerance
    within the target-L1/L2 full-cache family.  ``balanced_knee`` is the
    cross-family view and chooses the closest normalized energy/performance
    point to the ideal corner of the budget-specific four-dimensional Pareto
    front (area, energy, performance, and power).
    """

    values = list(evaluations)
    normalized_budgets = [_normalize_budget(budget) for budget in power_budgets]
    scope = str(constraints.get("constraint_scope", "per_application"))
    if scope not in {"per_application", "suite"}:
        raise ValueError(
            "constraints.constraint_scope must be 'per_application' or 'suite'"
        )

    max_area_ratio = _finite_constraint(
        constraints.get("max_area_ratio", 1.0),
        "constraints.max_area_ratio",
        positive=True,
    )
    runtime_tolerance = _finite_constraint(
        constraints.get("runtime_tolerance", 0.0),
        "constraints.runtime_tolerance",
        non_negative=True,
    )
    energy_tolerance = _finite_constraint(
        constraints.get("energy_tolerance", 0.0),
        "constraints.energy_tolerance",
        non_negative=True,
    )
    runtime_limit = 1.0 + runtime_tolerance
    energy_limit = 1.0 + energy_tolerance

    by_technology: dict[str, list[Evaluation]] = {}
    for evaluation in values:
        by_technology.setdefault(evaluation.design.technology, []).append(evaluation)

    results: list[SelectionResult] = []
    for technology in sorted(by_technology):
        all_candidates = by_technology[technology]
        pareto_candidates = [
            candidate
            for candidate in all_candidates
            if candidate.design.objective == _PARETO_FULL_CACHE
        ]
        candidates = pareto_candidates or all_candidates
        candidates = sorted(candidates, key=_stable_evaluation_key)
        retrofit_candidates = [
            candidate
            for candidate in candidates
            if candidate.design.candidate_family == "l2_retrofit"
        ]
        full_cache_candidates = [
            candidate
            for candidate in candidates
            if candidate.design.candidate_family == "full_cache"
        ]

        for budget in normalized_budgets:
            observed_unit = "x" if budget["type"] == "relative" else "mW"
            def common_candidates(
                family_candidates: Sequence[Evaluation],
            ) -> list[Evaluation]:
                return [
                    candidate
                    for candidate in family_candidates
                    if _passes_area(candidate, max_area_ratio)
                    and _passes_power(candidate, budget, scope)
                    and _has_finite_objectives(candidate)
                ]

            common = common_candidates(candidates)
            retrofit_common = common_candidates(retrofit_candidates)
            full_cache_common = common_candidates(full_cache_candidates)

            iso_performance = [
                candidate
                for candidate in retrofit_common
                if _runtime_value(candidate, scope) <= runtime_limit + _TOLERANCE
            ]
            results.append(
                _make_selection(
                    technology=technology,
                    budget=budget,
                    selector=MIN_ENERGY_ISO_PERFORMANCE,
                    eligible=iso_performance,
                    observed_unit=observed_unit,
                    scope=scope,
                    ranking=lambda candidate: (
                        candidate.energy_score,
                        -candidate.performance_score,
                        _observed_power(candidate, budget, scope),
                        candidate.area_ratio,
                        candidate.design.id,
                    ),
                    no_feasible_reasons=_failure_reasons(
                        candidates=retrofit_candidates,
                        common=retrofit_common,
                        selector_candidates=iso_performance,
                        budget=budget,
                        scope=scope,
                        max_area_ratio=max_area_ratio,
                        selector_limit=("runtime", runtime_limit),
                    ),
                )
            )

            iso_energy = [
                candidate
                for candidate in full_cache_common
                if _energy_value(candidate, scope) <= energy_limit + _TOLERANCE
            ]
            results.append(
                _make_selection(
                    technology=technology,
                    budget=budget,
                    selector=MAX_PERFORMANCE_ISO_ENERGY,
                    eligible=iso_energy,
                    observed_unit=observed_unit,
                    scope=scope,
                    ranking=lambda candidate: (
                        -candidate.performance_score,
                        candidate.energy_score,
                        _observed_power(candidate, budget, scope),
                        candidate.area_ratio,
                        candidate.design.id,
                    ),
                    no_feasible_reasons=_failure_reasons(
                        candidates=full_cache_candidates,
                        common=full_cache_common,
                        selector_candidates=iso_energy,
                        budget=budget,
                        scope=scope,
                        max_area_ratio=max_area_ratio,
                        selector_limit=("energy", energy_limit),
                    ),
                )
            )

            budget_front = _power_aware_pareto(common, budget, scope)
            results.append(
                _make_selection(
                    technology=technology,
                    budget=budget,
                    selector=BALANCED_KNEE,
                    eligible=budget_front,
                    observed_unit=observed_unit,
                    scope=scope,
                    ranking=_knee_ranking(budget_front, budget, scope),
                    no_feasible_reasons=_failure_reasons(
                        candidates=candidates,
                        common=common,
                        selector_candidates=budget_front,
                        budget=budget,
                        scope=scope,
                        max_area_ratio=max_area_ratio,
                    ),
                )
            )

    return results


def _normalize_budget(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("Each power budget must be an object")
    name = raw.get("name")
    budget_type = raw.get("type")
    limit = raw.get("limit")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Each power budget requires a non-empty string name")
    if budget_type not in {"relative", "absolute_mw"}:
        raise ValueError(
            f"Power budget {name!r} type must be 'relative' or 'absolute_mw'"
        )
    numeric_limit = _finite_constraint(
        limit, f"power budget {name!r} limit", positive=True
    )
    return {"name": name, "type": budget_type, "limit": numeric_limit}


def _finite_constraint(
    value: Any,
    label: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    if positive and result <= 0:
        raise ValueError(f"{label} must be > 0")
    if non_negative and result < 0:
        raise ValueError(f"{label} must be >= 0")
    return result


def _stable_evaluation_key(evaluation: Evaluation) -> tuple[Any, ...]:
    return (
        evaluation.design.technology,
        evaluation.design.objective,
        evaluation.design.coordinates(),
        evaluation.design.id,
    )


def _has_finite_objectives(evaluation: Evaluation) -> bool:
    return all(
        math.isfinite(value)
        for value in (
            evaluation.area_ratio,
            evaluation.energy_score,
            evaluation.performance_score,
        )
    )


def _passes_area(evaluation: Evaluation, max_area_ratio: float) -> bool:
    return (
        math.isfinite(evaluation.area_ratio)
        and evaluation.area_ratio <= max_area_ratio + _TOLERANCE
    )


def _passes_power(
    evaluation: Evaluation,
    budget: Mapping[str, Any],
    scope: str,
) -> bool:
    observed = _observed_power(evaluation, budget, scope)
    return math.isfinite(observed) and observed <= float(budget["limit"]) + _TOLERANCE


def _observed_power(
    evaluation: Evaluation,
    budget: Mapping[str, Any],
    scope: str,
) -> float:
    if budget["type"] == "relative":
        if scope == "suite":
            return float(evaluation.power_score)
        app_values = [
            float(result.power_ratio)
            for result in evaluation.app_results
            if math.isfinite(result.power_ratio)
        ]
        if app_values:
            return max(app_values)
        fallback = float(getattr(evaluation, "worst_power_ratio", math.nan))
        if math.isfinite(fallback):
            return fallback
        return float(evaluation.power_score)

    if scope == "suite":
        aggregate = float(getattr(evaluation, "power_mw_score", math.nan))
        if math.isfinite(aggregate):
            return aggregate
        app_values = [
            float(result.cache_power_mw)
            for result in evaluation.app_results
            if math.isfinite(result.cache_power_mw)
        ]
        if app_values:
            return max(app_values)
        return float(getattr(evaluation, "max_power_mw", math.nan))

    app_values = [
        float(result.cache_power_mw)
        for result in evaluation.app_results
        if math.isfinite(result.cache_power_mw)
    ]
    if app_values:
        return max(app_values)
    fallback = float(getattr(evaluation, "max_power_mw", math.nan))
    if math.isfinite(fallback):
        return fallback
    return float(getattr(evaluation, "power_mw_score", math.nan))


def _runtime_value(evaluation: Evaluation, scope: str) -> float:
    if scope == "per_application":
        return float(evaluation.worst_runtime_ratio)
    if not math.isfinite(evaluation.performance_score) or evaluation.performance_score <= 0:
        return math.inf
    return 1.0 / evaluation.performance_score


def _energy_value(evaluation: Evaluation, scope: str) -> float:
    if scope == "per_application":
        return float(evaluation.worst_energy_ratio)
    return float(evaluation.energy_score)


def _dominates(
    left: Evaluation,
    right: Evaluation,
    budget: Mapping[str, Any],
    scope: str,
) -> bool:
    left_power = _observed_power(left, budget, scope)
    right_power = _observed_power(right, budget, scope)
    no_worse = (
        left.area_ratio <= right.area_ratio + _TOLERANCE
        and left.energy_score <= right.energy_score + _TOLERANCE
        and left.performance_score + _TOLERANCE >= right.performance_score
        and left_power <= right_power + _TOLERANCE
    )
    strictly_better = (
        left.area_ratio < right.area_ratio - _TOLERANCE
        or left.energy_score < right.energy_score - _TOLERANCE
        or left.performance_score > right.performance_score + _TOLERANCE
        or left_power < right_power - _TOLERANCE
    )
    return no_worse and strictly_better


def _power_aware_pareto(
    candidates: Sequence[Evaluation],
    budget: Mapping[str, Any],
    scope: str,
) -> list[Evaluation]:
    front = [
        candidate
        for candidate in candidates
        if not any(
            _dominates(other, candidate, budget, scope)
            for other in candidates
            if other is not candidate
        )
    ]
    return sorted(
        front,
        key=lambda candidate: (
            candidate.area_ratio,
            candidate.energy_score,
            -candidate.performance_score,
            _observed_power(candidate, budget, scope),
            candidate.design.id,
        ),
    )


def _knee_ranking(
    front: Sequence[Evaluation],
    budget: Mapping[str, Any],
    scope: str,
):
    if not front:
        return _stable_evaluation_key

    energy_min = min(candidate.energy_score for candidate in front)
    energy_max = max(candidate.energy_score for candidate in front)
    performance_min = min(candidate.performance_score for candidate in front)
    performance_max = max(candidate.performance_score for candidate in front)
    energy_span = energy_max - energy_min
    performance_span = performance_max - performance_min

    def ranking(candidate: Evaluation) -> tuple[Any, ...]:
        normalized_energy = (
            0.0
            if energy_span <= _TOLERANCE
            else (candidate.energy_score - energy_min) / energy_span
        )
        normalized_performance_loss = (
            0.0
            if performance_span <= _TOLERANCE
            else (performance_max - candidate.performance_score) / performance_span
        )
        distance = math.hypot(normalized_energy, normalized_performance_loss)
        return (
            distance,
            _observed_power(candidate, budget, scope),
            candidate.area_ratio,
            candidate.energy_score,
            -candidate.performance_score,
            candidate.design.id,
        )

    return ranking


def _make_selection(
    *,
    technology: str,
    budget: dict[str, Any],
    selector: str,
    eligible: Sequence[Evaluation],
    observed_unit: str,
    scope: str,
    ranking,
    no_feasible_reasons: list[str],
) -> SelectionResult:
    if selector not in _SELECTORS:
        raise ValueError(f"Unsupported selector: {selector}")
    if not eligible:
        return SelectionResult(
            technology=technology,
            budget=dict(budget),
            selector=selector,
            status="NO_FEASIBLE_CANDIDATE",
            evaluation=None,
            reasons=no_feasible_reasons,
            observed_power=None,
            observed_unit=observed_unit,
        )

    selected = min(eligible, key=ranking)
    return SelectionResult(
        technology=technology,
        budget=dict(budget),
        selector=selector,
        status="FEASIBLE",
        evaluation=selected,
        reasons=[],
        observed_power=_observed_power(selected, budget, scope),
        observed_unit=observed_unit,
    )


def _failure_reasons(
    *,
    candidates: Sequence[Evaluation],
    common: Sequence[Evaluation],
    selector_candidates: Sequence[Evaluation],
    budget: Mapping[str, Any],
    scope: str,
    max_area_ratio: float,
    selector_limit: tuple[str, float] | None = None,
) -> list[str]:
    if selector_candidates:
        return []
    if not candidates:
        return ["no evaluated candidates"]

    area_eligible = [
        candidate for candidate in candidates if _passes_area(candidate, max_area_ratio)
    ]
    if not area_eligible:
        return [f"no candidate satisfies area <= {max_area_ratio:.6g}x"]

    finite_objectives = [
        candidate for candidate in area_eligible if _has_finite_objectives(candidate)
    ]
    if not finite_objectives:
        return ["all area-eligible candidates have non-finite objective metrics"]

    power_eligible = [
        candidate
        for candidate in finite_objectives
        if _passes_power(candidate, budget, scope)
    ]
    if not power_eligible:
        unit = "x" if budget["type"] == "relative" else "mW"
        return [
            "no candidate satisfies power <= "
            f"{float(budget['limit']):.6g}{unit} ({scope})"
        ]

    if not common:
        return ["no candidate satisfies the common area and power constraints"]
    if selector_limit is not None:
        metric, limit = selector_limit
        return [f"no candidate satisfies {metric} <= {limit:.6g}x"]
    return ["no candidate remains on the budget-specific Pareto front"]
