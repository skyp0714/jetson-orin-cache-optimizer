"""Shared immutable data models for cache optimization."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Iterable


@dataclasses.dataclass(frozen=True)
class CacheDesign:
    """One physical/architectural cache candidate.

    Legacy objective 1 is represented as SRAM in L1 and the target technology
    in L2; legacy objective 2 uses the target technology in both levels.  The
    unified Pareto policy contains both physical families in one candidate pool.
    """

    technology: str
    objective: str
    l1_technology: str
    l2_technology: str
    l1_capacity_kib: int
    l1_associativity: int
    l2_capacity_kib: int
    l2_associativity: int

    @property
    def id(self) -> str:
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"{self.technology}-{self.objective}-{digest}"

    def coordinates(self) -> tuple[int, int, int, int]:
        return (
            self.l1_capacity_kib,
            self.l1_associativity,
            self.l2_capacity_kib,
            self.l2_associativity,
        )

    @property
    def candidate_family(self) -> str:
        """Return the physical placement policy used by this candidate."""
        return "l2_retrofit" if self.l1_technology == "sram" else "full_cache"


@dataclasses.dataclass
class AccessCounts:
    l1_read_hit: int = 0
    l1_read_miss: int = 0
    l1_write: int = 0
    l2_read_hit: int = 0
    l2_read_miss: int = 0
    l2_write: int = 0

    def validate(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value < 0:
                raise ValueError(f"{field.name} must be non-negative, got {value}")


@dataclasses.dataclass
class AppResult:
    application: str
    cycles: int
    instructions: int
    runtime_s: float
    ipc: float
    accesses: AccessCounts
    l1_energy_nj: float = 0.0
    l2_energy_nj: float = 0.0
    cache_energy_nj: float = 0.0
    cache_power_mw: float = 0.0
    runtime_ratio: float = math.nan
    speedup: float = math.nan
    energy_ratio: float = math.nan
    power_ratio: float = math.nan


@dataclasses.dataclass
class Evaluation:
    design: CacheDesign
    iteration: int
    area_mm2: float
    area_ratio: float
    leakage_power_mw: float
    refresh_power_mw: float
    app_results: list[AppResult]
    performance_score: float
    energy_score: float
    power_score: float
    worst_runtime_ratio: float
    worst_energy_ratio: float
    feasible: bool
    constraint_reasons: list[str]
    constraint_violation_score: float = 0.0
    ppa: dict[str, Any] = dataclasses.field(default_factory=dict)
    generated_files: dict[str, str] = dataclasses.field(default_factory=dict)
    power_mw_score: float = math.nan
    worst_power_ratio: float = math.nan
    max_power_mw: float = math.nan

    @property
    def objective_value(self) -> float:
        if self.design.objective == "min_energy_runtime_bound":
            return self.energy_score
        if self.design.objective == "max_performance_energy_bound":
            return self.performance_score
        if self.design.objective == "pareto_full_cache":
            # A Pareto search has no single scalar objective.  This value is
            # only a deterministic convenience for callers which require one.
            return self.performance_score / max(
                self.energy_score * self.power_score * self.area_ratio, 1e-30
            )
        raise ValueError(f"Unknown objective: {self.design.objective}")


def weighted_geomean(values: Iterable[float], weights: Iterable[float] | None = None) -> float:
    values_list = list(values)
    if not values_list:
        raise ValueError("weighted_geomean requires at least one value")
    if any((not math.isfinite(v) or v <= 0) for v in values_list):
        raise ValueError(f"Geometric mean values must be finite and > 0: {values_list}")
    weights_list = list(weights) if weights is not None else [1.0] * len(values_list)
    if len(weights_list) != len(values_list):
        raise ValueError("values and weights must have equal length")
    if any((not math.isfinite(w) or w < 0) for w in weights_list):
        raise ValueError("weights must be finite and non-negative")
    total_weight = sum(weights_list)
    if total_weight <= 0:
        raise ValueError("At least one weight must be positive")
    return math.exp(
        sum(w * math.log(v) for v, w in zip(values_list, weights_list)) / total_weight
    )


def to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
