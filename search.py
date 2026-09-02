"""Deterministic iterative beam/Pareto search over a discrete cache space."""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Iterable
from typing import Any, Callable, List

from .models import CacheDesign, Evaluation


MIN_ENERGY = "min_energy_runtime_bound"
MAX_PERFORMANCE = "max_performance_energy_bound"
PARETO_FULL_CACHE = "pareto_full_cache"


@dataclasses.dataclass(frozen=True)
class DiscreteSpace:
    l1_capacity_kib: tuple[int, ...]
    l1_associativity: tuple[int, ...]
    l2_capacity_kib: tuple[int, ...]
    l2_associativity: tuple[int, ...]

    @classmethod
    def from_config(cls, raw: dict) -> "DiscreteSpace":
        return cls(
            tuple(sorted(raw["l1_capacity_kib"])),
            tuple(sorted(raw["l1_associativity"])),
            tuple(sorted(raw["l2_capacity_kib"])),
            tuple(sorted(raw["l2_associativity"])),
        )

    def nearest(self, values: tuple[int, ...], target: int) -> int:
        return min(values, key=lambda value: (abs(value - target), value))


@dataclasses.dataclass
class SearchOutcome:
    evaluations: list[Evaluation]
    errors: list[dict[str, str | int]]
    stopped_reason: str
    attempted: int = 0

    @property
    def optimum(self) -> Evaluation | None:
        if not self.evaluations:
            return None
        return select_optimum(self.evaluations, self.evaluations[0].design.objective)


BatchEvaluator = Callable[[List[CacheDesign], int], List[Evaluation]]


class AdaptiveParetoSearch:
    """Small-budget search which expands neighbors of feasible and Pareto elites.

    This is intentionally deterministic. Long Accel-Sim runs can be resumed and
    compared exactly; a random Bayesian optimizer would otherwise change the
    evaluated set when one subprocess fails or times out.
    """

    def __init__(
        self,
        *,
        technology: str,
        objective: str,
        space: DiscreteSpace,
        baseline_l1_capacity_kib: int,
        baseline_l1_associativity: int,
        baseline_l2_capacity_kib: int,
        baseline_l2_associativity: int,
        max_evaluations: int,
        max_iterations: int,
        beam_width: int,
        power_constraints: list[dict[str, Any]] | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> None:
        if objective not in {MIN_ENERGY, MAX_PERFORMANCE, PARETO_FULL_CACHE}:
            raise ValueError(f"Unsupported objective: {objective}")
        self.technology = technology
        self.objective = objective
        self.space = space
        self.baseline = (
            baseline_l1_capacity_kib,
            baseline_l1_associativity,
            baseline_l2_capacity_kib,
            baseline_l2_associativity,
        )
        self.max_evaluations = max_evaluations
        self.max_iterations = max_iterations
        self.beam_width = beam_width
        self.power_constraints = list(power_constraints or [])
        self.constraints = dict(constraints or {})

    def run(self, evaluate_batch: BatchEvaluator) -> SearchOutcome:
        evaluations: list[Evaluation] = []
        errors: list[dict[str, str | int]] = []
        visited: set[str] = set()
        attempted = 0
        pending = self._initial_designs()
        stopped_reason = "max_iterations"

        for iteration in range(self.max_iterations):
            batch: list[CacheDesign] = []
            for design in pending:
                if design.id in visited:
                    continue
                visited.add(design.id)
                batch.append(design)
                if attempted + len(batch) >= self.max_evaluations:
                    break
            if not batch:
                stopped_reason = "search_space_exhausted"
                break

            attempted += len(batch)

            try:
                new_evaluations = evaluate_batch(batch, iteration)
            except Exception as exc:  # evaluator supplies candidate-local detail where possible
                errors.append({"iteration": iteration, "error": str(exc)})
                new_evaluations = []
            returned = {evaluation.design.id for evaluation in new_evaluations}
            for design in batch:
                if design.id not in returned:
                    errors.append(
                        {
                            "iteration": iteration,
                            "candidate": design.id,
                            "error": "evaluator returned no result",
                        }
                    )
            evaluations.extend(new_evaluations)
            if attempted >= self.max_evaluations:
                stopped_reason = "max_evaluations"
                break
            if not evaluations:
                pending = self._fallback_unvisited(visited)
                continue

            elites = self._select_elites(evaluations)
            pending = []
            for elite in elites:
                pending.extend(self._neighbors(elite.design))
            pending = self._unique_prioritized(pending, visited, elites)
            if not pending:
                pending = self._fallback_unvisited(visited)

        return SearchOutcome(
            evaluations=evaluations,
            errors=errors,
            stopped_reason=stopped_reason,
            attempted=attempted,
        )

    def _make(
        self,
        values: tuple[int, int, int, int],
        candidate_family: str | None = None,
    ) -> CacheDesign:
        l1_cap, l1_assoc, l2_cap, l2_assoc = values
        if self.objective == MIN_ENERGY or candidate_family == "l2_retrofit":
            # Objective 1 deliberately preserves the immutable SRAM L1.
            l1_cap, l1_assoc = self.baseline[0], self.baseline[1]
            l1_technology = "sram"
        else:
            l1_technology = self.technology
        return CacheDesign(
            technology=self.technology,
            objective=self.objective,
            l1_technology=l1_technology,
            l2_technology=self.technology,
            l1_capacity_kib=l1_cap,
            l1_associativity=l1_assoc,
            l2_capacity_kib=l2_cap,
            l2_associativity=l2_assoc,
        )

    def _initial_designs(self) -> list[CacheDesign]:
        s = self.space
        baseline = (
            s.nearest(s.l1_capacity_kib, self.baseline[0]),
            s.nearest(s.l1_associativity, self.baseline[1]),
            s.nearest(s.l2_capacity_kib, self.baseline[2]),
            s.nearest(s.l2_associativity, self.baseline[3]),
        )
        low = (
            s.l1_capacity_kib[0],
            baseline[1],
            s.l2_capacity_kib[0],
            baseline[3],
        )
        high = (
            s.l1_capacity_kib[-1],
            baseline[1],
            s.l2_capacity_kib[-1],
            baseline[3],
        )
        mid = (
            s.l1_capacity_kib[len(s.l1_capacity_kib) // 2],
            baseline[1],
            s.l2_capacity_kib[len(s.l2_capacity_kib) // 2],
            baseline[3],
        )
        assoc_low = (baseline[0], s.l1_associativity[0], baseline[2], s.l2_associativity[0])
        assoc_high = (baseline[0], s.l1_associativity[-1], baseline[2], s.l2_associativity[-1])
        raw = [baseline, low, high, mid, assoc_low, assoc_high]
        if self.objective == MIN_ENERGY:
            raw = [
                (self.baseline[0], self.baseline[1], values[2], values[3])
                for values in raw
            ]
        if self.objective == PARETO_FULL_CACHE:
            full = self._dedupe_designs(
                self._make(values, "full_cache") for values in raw
            )
            retrofit = self._dedupe_designs(
                self._make(values, "l2_retrofit") for values in raw
            )
            # Seed both placement families before either family consumes the
            # whole beam.  Subsequent neighbor expansion preserves the family.
            interleaved: list[CacheDesign] = []
            for index in range(max(len(full), len(retrofit))):
                if index < len(full):
                    interleaved.append(full[index])
                if index < len(retrofit):
                    interleaved.append(retrofit[index])
            return self._dedupe_designs(interleaved)[: self.beam_width]
        return self._dedupe_designs(self._make(values) for values in raw)[: self.beam_width]

    def _neighbors(self, design: CacheDesign) -> list[CacheDesign]:
        values = list(design.coordinates())
        dimensions = (
            self.space.l1_capacity_kib,
            self.space.l1_associativity,
            self.space.l2_capacity_kib,
            self.space.l2_associativity,
        )
        retrofit = design.candidate_family == "l2_retrofit"
        active_indices = (2, 3) if retrofit else (0, 1, 2, 3)
        neighbors: list[CacheDesign] = []
        for dim_index in active_indices:
            dimension = dimensions[dim_index]
            try:
                current_index = dimension.index(values[dim_index])
            except ValueError:
                continue
            for delta in (-1, 1):
                next_index = current_index + delta
                if 0 <= next_index < len(dimension):
                    changed = values.copy()
                    changed[dim_index] = dimension[next_index]
                    neighbors.append(
                        self._make(tuple(changed), design.candidate_family)
                    )
        return self._dedupe_designs(neighbors)

    def _select_elites(self, evaluations: list[Evaluation]) -> list[Evaluation]:
        ranked = sorted(evaluations, key=lambda evaluation: ranking_key(evaluation, self.objective))
        selected = ranked[: self.beam_width]
        for evaluation in pareto_front(evaluations):
            if evaluation not in selected:
                selected.append(evaluation)
            if len(selected) >= self.beam_width * 2:
                break
        if self.objective == PARETO_FULL_CACHE and self.power_constraints:
            # A geometric-mean power frontier can hide a point needed by a
            # per-application or absolute-mW cap.  Preserve every currently
            # selected budget operating point as a neighbor-expansion seed.
            from .selection import select_power_budget_optima

            for choice in select_power_budget_optima(
                evaluations,
                self.power_constraints,
                self.constraints,
            ):
                if (
                    choice.evaluation is not None
                    and choice.evaluation not in selected
                ):
                    selected.append(choice.evaluation)
        # Preserve one point in each coarse area band so the saturation curve
        # is explored even when those points violate the strict SRAM area bound.
        by_area = sorted(evaluations, key=lambda evaluation: evaluation.area_ratio)
        if by_area:
            for index in (0, len(by_area) // 2, len(by_area) - 1):
                if by_area[index] not in selected:
                    selected.append(by_area[index])
        return selected

    def _unique_prioritized(
        self,
        designs: Iterable[CacheDesign],
        visited: set[str],
        elites: list[Evaluation],
    ) -> list[CacheDesign]:
        unique = [design for design in self._dedupe_designs(designs) if design.id not in visited]
        elite_coords = [elite.design.coordinates() for elite in elites]

        def distance(design: CacheDesign) -> tuple[float, tuple[int, int, int, int]]:
            coords = design.coordinates()
            normalized = []
            spans = (
                max(self.space.l1_capacity_kib) / min(self.space.l1_capacity_kib),
                max(self.space.l1_associativity) / min(self.space.l1_associativity),
                max(self.space.l2_capacity_kib) / min(self.space.l2_capacity_kib),
                max(self.space.l2_associativity) / min(self.space.l2_associativity),
            )
            for elite in elite_coords:
                normalized.append(
                    sum(abs(a - b) / max(a, b) / max(span, 1.0) for a, b, span in zip(coords, elite, spans))
                )
            return (min(normalized) if normalized else 0.0, coords)

        return sorted(unique, key=distance)[: self.beam_width]

    def _fallback_unvisited(self, visited: set[str]) -> list[CacheDesign]:
        if self.objective == PARETO_FULL_CACHE:
            full_dimensions = (
                self.space.l1_capacity_kib,
                self.space.l1_associativity,
                self.space.l2_capacity_kib,
                self.space.l2_associativity,
            )
            retrofit_dimensions = (
                (self.baseline[0],),
                (self.baseline[1],),
                self.space.l2_capacity_kib,
                self.space.l2_associativity,
            )
            designs = itertools.chain(
                (
                    self._make(values, "full_cache")
                    for values in itertools.product(*full_dimensions)
                ),
                (
                    self._make(values, "l2_retrofit")
                    for values in itertools.product(*retrofit_dimensions)
                ),
            )
        else:
            dimensions = (
                (self.baseline[0],) if self.objective == MIN_ENERGY else self.space.l1_capacity_kib,
                (self.baseline[1],) if self.objective == MIN_ENERGY else self.space.l1_associativity,
                self.space.l2_capacity_kib,
                self.space.l2_associativity,
            )
            designs = (self._make(values) for values in itertools.product(*dimensions))
        return [design for design in designs if design.id not in visited][: self.beam_width]

    @staticmethod
    def _dedupe_designs(designs: Iterable[CacheDesign]) -> list[CacheDesign]:
        result: list[CacheDesign] = []
        seen: set[str] = set()
        for design in designs:
            if design.id not in seen:
                seen.add(design.id)
                result.append(design)
        return result


def constraint_violation(evaluation: Evaluation, objective: str) -> float:
    if evaluation.feasible:
        return 0.0
    if evaluation.constraint_violation_score > 0:
        return evaluation.constraint_violation_score
    area = max(0.0, evaluation.area_ratio - 1.0)
    if objective == MIN_ENERGY:
        runtime = max(0.0, evaluation.worst_runtime_ratio - 1.0)
        return area + runtime
    if objective == MAX_PERFORMANCE:
        energy = max(0.0, evaluation.worst_energy_ratio - 1.0)
        return area + energy
    if objective == PARETO_FULL_CACHE:
        return area
    raise ValueError(f"Unsupported objective: {objective}")


def ranking_key(evaluation: Evaluation, objective: str) -> tuple:
    if objective == MIN_ENERGY:
        if evaluation.feasible:
            return (
                0,
                evaluation.energy_score,
                -evaluation.performance_score,
                evaluation.area_ratio,
                evaluation.design.id,
            )
        return (
            1,
            constraint_violation(evaluation, objective),
            evaluation.energy_score,
            -evaluation.performance_score,
            evaluation.area_ratio,
            evaluation.design.id,
        )
    if objective == MAX_PERFORMANCE:
        if evaluation.feasible:
            return (
                0,
                -evaluation.performance_score,
                evaluation.energy_score,
                evaluation.area_ratio,
                evaluation.design.id,
            )
        return (
            1,
            constraint_violation(evaluation, objective),
            -evaluation.performance_score,
            evaluation.energy_score,
            evaluation.area_ratio,
            evaluation.design.id,
        )
    if objective == PARETO_FULL_CACHE:
        violation = constraint_violation(evaluation, objective)
        # Pareto elites are added separately below.  This scalar key provides a
        # deterministic balanced seed and favors legal-area candidates.
        return (
            0 if evaluation.feasible else 1,
            violation,
            evaluation.area_ratio
            * evaluation.energy_score
            * evaluation.power_score
            / max(evaluation.performance_score, 1e-30),
            evaluation.energy_score,
            -evaluation.performance_score,
            evaluation.power_score,
            evaluation.design.id,
        )
    raise ValueError(f"Unsupported objective: {objective}")


def select_optimum(evaluations: Iterable[Evaluation], objective: str) -> Evaluation | None:
    values = list(evaluations)
    if not values:
        return None
    return min(values, key=lambda evaluation: ranking_key(evaluation, objective))


def dominates(left: Evaluation, right: Evaluation) -> bool:
    left_metrics = (
        left.area_ratio,
        left.energy_score,
        left.performance_score,
        left.power_score,
    )
    right_metrics = (
        right.area_ratio,
        right.energy_score,
        right.performance_score,
        right.power_score,
    )
    if not all(math.isfinite(value) for value in left_metrics + right_metrics):
        return False
    no_worse = (
        left.area_ratio <= right.area_ratio
        and left.energy_score <= right.energy_score
        and left.performance_score >= right.performance_score
        and left.power_score <= right.power_score
    )
    strictly_better = (
        left.area_ratio < right.area_ratio
        or left.energy_score < right.energy_score
        or left.performance_score > right.performance_score
        or left.power_score < right.power_score
    )
    return no_worse and strictly_better


def pareto_front(evaluations: Iterable[Evaluation]) -> list[Evaluation]:
    values = [
        evaluation
        for evaluation in evaluations
        if all(
            math.isfinite(value)
            for value in (
                evaluation.area_ratio,
                evaluation.energy_score,
                evaluation.performance_score,
                evaluation.power_score,
            )
        )
    ]
    front = [candidate for candidate in values if not any(dominates(other, candidate) for other in values if other is not candidate)]
    return sorted(
        front,
        key=lambda evaluation: (
            evaluation.area_ratio,
            -evaluation.performance_score,
            evaluation.energy_score,
            evaluation.power_score,
            evaluation.design.id,
        ),
    )
