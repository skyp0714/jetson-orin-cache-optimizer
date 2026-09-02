from __future__ import annotations

from dataclasses import dataclass

import pytest

from cache_optimizer.energy import build_evaluation, build_raw_app_result, compute_level_energy
from cache_optimizer.models import AccessCounts, CacheDesign
from cache_optimizer.search import MAX_PERFORMANCE, PARETO_FULL_CACHE


@dataclass
class PPA:
    area_mm2: float = 1.0
    hit_latency_ns: float = 1.0
    miss_latency_ns: float = 1.0
    write_latency_ns: float = 1.0
    hit_energy_nj: float = 2.0
    miss_energy_nj: float = 3.0
    write_energy_nj: float = 4.0
    leakage_power_mw: float = 5.0
    refresh_energy_nj: float | None = None
    refresh_power_mw: float | None = None


def test_energy_units_and_refresh():
    result = compute_level_energy(PPA(refresh_energy_nj=10.0), read_hit=2, read_miss=1, writes=3, runtime_s=2.0, instances=2, counts_scope="per_instance", bank_count=4, retention_time_us=100.0)
    assert result.dynamic_nj == 2 * (2 * 2 + 1 * 3 + 3 * 4)
    assert result.leakage_nj == 2 * 5 * 2 * 1e6
    assert result.refresh_nj == pytest.approx((10 / 100 * 4 * 2) * 2 * 1e6)


def test_constraint_is_checked_per_application():
    stats = {
        "cycles": 110,
        "instructions": 1000,
        "ipc": 1.0,
        "accesses": AccessCounts(1, 1, 1, 1, 1, 1),
    }
    baseline = build_raw_app_result("app", {**stats, "cycles": 100}, clock_hz=100.0, l1_ppa=PPA(), l2_ppa=PPA(), l1_instances=1, l2_instances=1, l1_bank_count=1, l2_bank_count=1, l1_retention_time_us=None, l2_retention_time_us=None)
    design = CacheDesign("gc", MAX_PERFORMANCE, "gc", "gc", 256, 4, 4096, 16)
    evaluation = build_evaluation(
        design=design,
        iteration=0,
        stats_by_app={"app": stats},
        baseline_by_app={"app": baseline},
        weights_by_app={"app": 1},
        clock_hz=100.0,
        l1_ppa=PPA(),
        l2_ppa=PPA(),
        baseline_area_mm2=2.0,
        l1_instances=1,
        l2_instances=1,
        l1_counts_scope="all_instances",
        l2_counts_scope="all_instances",
        l1_bank_count=1,
        l2_bank_count=1,
        l1_retention_time_us=None,
        l2_retention_time_us=None,
        constraints={"max_area_ratio": 1.0, "runtime_tolerance": 0.02, "energy_tolerance": 0.0, "constraint_scope": "per_application"},
    )
    assert evaluation.worst_energy_ratio > 1.0
    assert evaluation.power_mw_score == pytest.approx(
        evaluation.app_results[0].cache_power_mw
    )
    assert evaluation.worst_power_ratio == pytest.approx(
        evaluation.app_results[0].power_ratio
    )
    assert evaluation.max_power_mw == pytest.approx(
        evaluation.app_results[0].cache_power_mw
    )
    assert not evaluation.feasible


def test_pareto_pool_defers_runtime_energy_and_power_to_post_selection():
    stats = {
        "cycles": 120,
        "instructions": 1000,
        "ipc": 1.0,
        "accesses": AccessCounts(1, 1, 1, 1, 1, 1),
    }
    baseline = build_raw_app_result(
        "app",
        {**stats, "cycles": 100},
        clock_hz=100.0,
        l1_ppa=PPA(),
        l2_ppa=PPA(),
        l1_instances=1,
        l2_instances=1,
        l1_bank_count=1,
        l2_bank_count=1,
        l1_retention_time_us=None,
        l2_retention_time_us=None,
    )
    design = CacheDesign(
        "gc", PARETO_FULL_CACHE, "gc", "gc", 256, 4, 4096, 16
    )
    evaluation = build_evaluation(
        design=design,
        iteration=0,
        stats_by_app={"app": stats},
        baseline_by_app={"app": baseline},
        weights_by_app={"app": 1},
        clock_hz=100.0,
        l1_ppa=PPA(),
        l2_ppa=PPA(),
        baseline_area_mm2=2.0,
        l1_instances=1,
        l2_instances=1,
        l1_counts_scope="all_instances",
        l2_counts_scope="all_instances",
        l1_bank_count=1,
        l2_bank_count=1,
        l1_retention_time_us=None,
        l2_retention_time_us=None,
        constraints={
            "max_area_ratio": 1.0,
            "runtime_tolerance": 0.02,
            "energy_tolerance": 0.0,
            "constraint_scope": "per_application",
        },
    )
    assert evaluation.worst_runtime_ratio == pytest.approx(1.2)
    assert evaluation.feasible
    assert evaluation.constraint_reasons == []
