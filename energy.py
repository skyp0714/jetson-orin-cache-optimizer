"""Workload-level cache energy integration and constraint evaluation."""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

from .models import AccessCounts, AppResult, CacheDesign, Evaluation, weighted_geomean
from .search import MAX_PERFORMANCE, MIN_ENERGY, PARETO_FULL_CACHE


@dataclasses.dataclass(frozen=True)
class LevelEnergy:
    dynamic_nj: float
    leakage_nj: float
    refresh_nj: float

    @property
    def total_nj(self) -> float:
        return self.dynamic_nj + self.leakage_nj + self.refresh_nj


def compute_level_energy(
    ppa: Any,
    *,
    read_hit: int,
    read_miss: int,
    writes: int,
    runtime_s: float,
    instances: int,
    counts_scope: str,
    bank_count: int,
    retention_time_us: float | None,
) -> LevelEnergy:
    if min(read_hit, read_miss, writes) < 0:
        raise ValueError("Access counts cannot be negative")
    if runtime_s <= 0 or instances <= 0 or bank_count <= 0:
        raise ValueError("runtime, instances, and bank_count must be positive")

    if counts_scope not in {"all_instances", "per_instance"}:
        raise ValueError("counts_scope must be 'all_instances' or 'per_instance'")
    dynamic_multiplier = instances if counts_scope == "per_instance" else 1
    dynamic_nj = dynamic_multiplier * (
        read_hit * float(ppa.hit_energy_nj)
        + read_miss * float(ppa.miss_energy_nj)
        + writes * float(ppa.write_energy_nj)
    )
    leakage_power_mw = instances * float(ppa.leakage_power_mw)
    leakage_nj = leakage_power_mw * runtime_s * 1e6

    refresh_power_mw = _refresh_power(ppa, retention_time_us, bank_count, instances)
    refresh_nj = refresh_power_mw * runtime_s * 1e6
    return LevelEnergy(dynamic_nj, leakage_nj, refresh_nj)


def _get_stats_value(stats: Any, name: str) -> Any:
    if isinstance(stats, Mapping):
        return stats[name]
    return getattr(stats, name)


def build_raw_app_result(
    name: str,
    stats: Any,
    *,
    clock_hz: float,
    l1_ppa: Any,
    l2_ppa: Any,
    l1_instances: int,
    l2_instances: int,
    l1_counts_scope: str = "all_instances",
    l2_counts_scope: str = "all_instances",
    l1_bank_count: int,
    l2_bank_count: int,
    l1_retention_time_us: float | None,
    l2_retention_time_us: float | None,
) -> AppResult:
    cycles = int(_get_stats_value(stats, "cycles"))
    instructions = int(_get_stats_value(stats, "instructions"))
    ipc = float(_get_stats_value(stats, "ipc"))
    accesses = _get_stats_value(stats, "accesses")
    if isinstance(accesses, Mapping):
        accesses = AccessCounts(**accesses)
    if not isinstance(accesses, AccessCounts):
        raise TypeError("stats.accesses must be AccessCounts or a compatible mapping")
    accesses.validate()
    if cycles <= 0 or clock_hz <= 0:
        raise ValueError("cycles and clock_hz must be positive")
    runtime_s = cycles / clock_hz

    l1 = compute_level_energy(
        l1_ppa,
        read_hit=accesses.l1_read_hit,
        read_miss=accesses.l1_read_miss,
        writes=accesses.l1_write,
        runtime_s=runtime_s,
        instances=l1_instances,
        counts_scope=l1_counts_scope,
        bank_count=l1_bank_count,
        retention_time_us=l1_retention_time_us,
    )
    l2 = compute_level_energy(
        l2_ppa,
        read_hit=accesses.l2_read_hit,
        read_miss=accesses.l2_read_miss,
        writes=accesses.l2_write,
        runtime_s=runtime_s,
        instances=l2_instances,
        counts_scope=l2_counts_scope,
        bank_count=l2_bank_count,
        retention_time_us=l2_retention_time_us,
    )
    total = l1.total_nj + l2.total_nj
    return AppResult(
        application=name,
        cycles=cycles,
        instructions=instructions,
        runtime_s=runtime_s,
        ipc=ipc,
        accesses=accesses,
        l1_energy_nj=l1.total_nj,
        l2_energy_nj=l2.total_nj,
        cache_energy_nj=total,
        cache_power_mw=total / runtime_s / 1e6,
    )


def normalize_app_result(result: AppResult, baseline: AppResult) -> AppResult:
    if baseline.runtime_s <= 0 or baseline.cache_energy_nj <= 0 or baseline.cache_power_mw <= 0:
        raise ValueError(f"Invalid baseline metrics for {baseline.application}")
    result.runtime_ratio = result.runtime_s / baseline.runtime_s
    result.speedup = baseline.runtime_s / result.runtime_s
    result.energy_ratio = result.cache_energy_nj / baseline.cache_energy_nj
    result.power_ratio = result.cache_power_mw / baseline.cache_power_mw
    return result


def build_evaluation(
    *,
    design: CacheDesign,
    iteration: int,
    stats_by_app: Mapping[str, Any],
    baseline_by_app: Mapping[str, AppResult],
    weights_by_app: Mapping[str, float],
    clock_hz: float,
    l1_ppa: Any,
    l2_ppa: Any,
    baseline_area_mm2: float,
    l1_instances: int,
    l2_instances: int,
    l1_counts_scope: str,
    l2_counts_scope: str,
    l1_bank_count: int,
    l2_bank_count: int,
    l1_retention_time_us: float | None,
    l2_retention_time_us: float | None,
    constraints: Mapping[str, Any],
    generated_files: dict[str, str] | None = None,
) -> Evaluation:
    app_results: list[AppResult] = []
    weights: list[float] = []
    for name, baseline in baseline_by_app.items():
        if name not in stats_by_app:
            raise ValueError(f"Candidate is missing simulator stats for application '{name}'")
        raw = build_raw_app_result(
            name,
            stats_by_app[name],
            clock_hz=clock_hz,
            l1_ppa=l1_ppa,
            l2_ppa=l2_ppa,
            l1_instances=l1_instances,
            l2_instances=l2_instances,
            l1_counts_scope=l1_counts_scope,
            l2_counts_scope=l2_counts_scope,
            l1_bank_count=l1_bank_count,
            l2_bank_count=l2_bank_count,
            l1_retention_time_us=l1_retention_time_us,
            l2_retention_time_us=l2_retention_time_us,
        )
        app_results.append(normalize_app_result(raw, baseline))
        weights.append(float(weights_by_app.get(name, 1.0)))

    performance_score = weighted_geomean((result.speedup for result in app_results), weights)
    energy_score = weighted_geomean((result.energy_ratio for result in app_results), weights)
    power_score = weighted_geomean((result.power_ratio for result in app_results), weights)
    power_mw_score = weighted_geomean(
        (result.cache_power_mw for result in app_results), weights
    )
    worst_runtime_ratio = max(result.runtime_ratio for result in app_results)
    worst_energy_ratio = max(result.energy_ratio for result in app_results)
    worst_power_ratio = max(result.power_ratio for result in app_results)
    max_power_mw = max(result.cache_power_mw for result in app_results)
    area_mm2 = (
        float(l1_ppa.area_mm2) * l1_instances + float(l2_ppa.area_mm2) * l2_instances
    )
    if baseline_area_mm2 <= 0:
        raise ValueError("baseline_area_mm2 must be positive")
    area_ratio = area_mm2 / baseline_area_mm2

    max_area_ratio = float(constraints["max_area_ratio"])
    runtime_limit = 1.0 + float(constraints["runtime_tolerance"])
    energy_limit = 1.0 + float(constraints["energy_tolerance"])
    reasons: list[str] = []
    violation_score = max(0.0, area_ratio / max_area_ratio - 1.0)
    if area_ratio > max_area_ratio + 1e-12:
        reasons.append(f"area {area_ratio:.6g} > {max_area_ratio:.6g}")

    scope = constraints.get("constraint_scope", "per_application")
    if design.objective == MIN_ENERGY:
        runtime_value = worst_runtime_ratio if scope == "per_application" else 1.0 / performance_score
        if runtime_value > runtime_limit + 1e-12:
            reasons.append(f"runtime {runtime_value:.6g} > {runtime_limit:.6g}")
            violation_score += runtime_value / runtime_limit - 1.0
    elif design.objective == MAX_PERFORMANCE:
        energy_value = worst_energy_ratio if scope == "per_application" else energy_score
        if energy_value > energy_limit + 1e-12:
            reasons.append(f"energy {energy_value:.6g} > {energy_limit:.6g}")
            violation_score += energy_value / energy_limit - 1.0
    elif design.objective == PARETO_FULL_CACHE:
        # Runtime, energy, and power limits define independent post-selection
        # scenarios on this common physical candidate pool.  The master search
        # therefore uses only the common area constraint as a hard constraint.
        pass
    else:
        raise ValueError(f"Unknown objective: {design.objective}")

    return Evaluation(
        design=design,
        iteration=iteration,
        area_mm2=area_mm2,
        area_ratio=area_ratio,
        leakage_power_mw=(
            float(l1_ppa.leakage_power_mw) * l1_instances
            + float(l2_ppa.leakage_power_mw) * l2_instances
        ),
        refresh_power_mw=_refresh_power(
            l1_ppa, l1_retention_time_us, l1_bank_count, l1_instances
        )
        + _refresh_power(l2_ppa, l2_retention_time_us, l2_bank_count, l2_instances),
        app_results=app_results,
        performance_score=performance_score,
        energy_score=energy_score,
        power_score=power_score,
        worst_runtime_ratio=worst_runtime_ratio,
        worst_energy_ratio=worst_energy_ratio,
        feasible=not reasons,
        constraint_reasons=reasons,
        constraint_violation_score=violation_score,
        ppa={"l1": _ppa_dict(l1_ppa), "l2": _ppa_dict(l2_ppa)},
        generated_files=generated_files or {},
        power_mw_score=power_mw_score,
        worst_power_ratio=worst_power_ratio,
        max_power_mw=max_power_mw,
    )


def _refresh_power(ppa: Any, retention_us: float | None, banks: int, instances: int) -> float:
    # nJ / us is numerically mW. NS-Cache labels refresh energy per bank.
    energy = getattr(ppa, "refresh_energy_nj", None)
    if retention_us is not None and energy is not None:
        return float(energy) / retention_us * banks * instances
    reported = getattr(ppa, "refresh_power_mw", None)
    return 0.0 if reported is None else float(reported) * banks * instances


def _ppa_dict(ppa: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(ppa):
        return dataclasses.asdict(ppa)
    return {
        name: getattr(ppa, name)
        for name in (
            "area_mm2", "hit_latency_ns", "miss_latency_ns", "write_latency_ns",
            "hit_energy_nj", "miss_energy_nj", "write_energy_nj", "leakage_power_mw",
            "refresh_latency_us", "refresh_energy_nj", "refresh_power_mw", "availability_percent",
        )
        if hasattr(ppa, name)
    }
