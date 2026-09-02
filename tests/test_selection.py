import math

from cache_optimizer.models import AccessCounts, AppResult, CacheDesign, Evaluation
from cache_optimizer.search import PARETO_FULL_CACHE
from cache_optimizer.selection import (
    MAX_PERFORMANCE_ISO_ENERGY,
    MIN_ENERGY_ISO_PERFORMANCE,
    select_power_budget_optima,
)


CONSTRAINTS = {
    "max_area_ratio": 1.0,
    "runtime_tolerance": 0.02,
    "energy_tolerance": 0.0,
    "constraint_scope": "per_application",
}


def candidate(
    name,
    *,
    perf,
    energy,
    power_ratio,
    power_mw,
    area=0.9,
    app_power_ratios=None,
):
    l1_technology = "sram" if name.startswith("hybrid") else "gain_cell"
    design = CacheDesign(
        "gain_cell",
        PARETO_FULL_CACHE,
        l1_technology,
        "gain_cell",
        256,
        4 if name != "fast" else 8,
        4096 if name != "fast" else 2048,
        16,
    )
    ratios = app_power_ratios or [power_ratio]
    apps = [
        AppResult(
            application=f"app-{index}",
            cycles=100,
            instructions=100,
            runtime_s=1.0,
            ipc=1.0,
            accesses=AccessCounts(),
            cache_power_mw=power_mw,
            runtime_ratio=1.0 / perf,
            speedup=perf,
            energy_ratio=energy,
            power_ratio=ratio,
        )
        for index, ratio in enumerate(ratios)
    ]
    return Evaluation(
        design=design,
        iteration=0,
        area_mm2=area,
        area_ratio=area,
        leakage_power_mw=0.0,
        refresh_power_mw=0.0,
        app_results=apps,
        performance_score=perf,
        energy_score=energy,
        power_score=math.prod(ratios) ** (1.0 / len(ratios)),
        worst_runtime_ratio=1.0 / perf,
        worst_energy_ratio=energy,
        feasible=True,
        constraint_reasons=[],
        power_mw_score=power_mw,
        worst_power_ratio=max(ratios),
        max_power_mw=power_mw,
    )


def selected(results, budget, selector):
    return next(
        result
        for result in results
        if result.budget["name"] == budget and result.selector == selector
    )


def test_ratio_budgets_select_different_performance_points():
    low = candidate("low", perf=1.05, energy=0.9, power_ratio=0.9, power_mw=90)
    mid = candidate("mid", perf=1.2, energy=0.9, power_ratio=1.4, power_mw=140)
    fast = candidate("fast", perf=1.3, energy=0.9, power_ratio=1.6, power_mw=160)
    results = select_power_budget_optima(
        [low, mid, fast],
        [
            {"name": "one", "type": "relative", "limit": 1.0},
            {"name": "one_half", "type": "relative", "limit": 1.5},
        ],
        CONSTRAINTS,
    )
    assert (
        selected(results, "one", MAX_PERFORMANCE_ISO_ENERGY).evaluation is low
    )
    assert (
        selected(results, "one_half", MAX_PERFORMANCE_ISO_ENERGY).evaluation
        is mid
    )


def test_absolute_mw_budget_is_independent_of_power_ratio():
    high_ratio_low_mw = candidate(
        "a", perf=1.2, energy=0.9, power_ratio=1.4, power_mw=120
    )
    low_ratio_high_mw = candidate(
        "b", perf=1.1, energy=0.9, power_ratio=0.9, power_mw=180
    )
    results = select_power_budget_optima(
        [high_ratio_low_mw, low_ratio_high_mw],
        [{"name": "150mw", "type": "absolute_mw", "limit": 150.0}],
        CONSTRAINTS,
    )
    assert (
        selected(results, "150mw", MAX_PERFORMANCE_ISO_ENERGY).evaluation
        is high_ratio_low_mw
    )


def test_per_application_power_cap_is_not_hidden_by_geomean():
    hot = candidate(
        "hot",
        perf=1.1,
        energy=0.9,
        power_ratio=math.sqrt(0.8),
        power_mw=100,
        app_power_ratios=[0.5, 1.6],
    )
    budget = [{"name": "one", "type": "relative", "limit": 1.0}]
    per_app = select_power_budget_optima([hot], budget, CONSTRAINTS)
    assert selected(
        per_app, "one", MAX_PERFORMANCE_ISO_ENERGY
    ).status == "NO_FEASIBLE_CANDIDATE"

    suite_constraints = dict(CONSTRAINTS, constraint_scope="suite")
    suite = select_power_budget_optima([hot], budget, suite_constraints)
    assert selected(suite, "one", MAX_PERFORMANCE_ISO_ENERGY).evaluation is hot


def test_unified_pool_keeps_hybrid_for_iso_performance_selection():
    hybrid = candidate(
        "hybrid-gain", perf=0.99, energy=0.72, power_ratio=0.71, power_mw=149
    )
    full = candidate(
        "full", perf=0.90, energy=0.69, power_ratio=0.63, power_mw=132
    )
    results = select_power_budget_optima(
        [hybrid, full],
        [{"name": "one", "type": "relative", "limit": 1.0}],
        CONSTRAINTS,
    )
    assert (
        selected(results, "one", MIN_ENERGY_ISO_PERFORMANCE).evaluation
        is hybrid
    )


def test_original_two_objectives_keep_their_candidate_families():
    hybrid = candidate(
        "hybrid-gain", perf=1.1, energy=0.8, power_ratio=0.8, power_mw=80
    )
    full = candidate(
        "full", perf=1.05, energy=0.4, power_ratio=0.8, power_mw=80
    )
    results = select_power_budget_optima(
        [hybrid, full],
        [{"name": "one", "type": "relative", "limit": 1.0}],
        CONSTRAINTS,
    )
    assert (
        selected(results, "one", MIN_ENERGY_ISO_PERFORMANCE).evaluation
        is hybrid
    )
    assert (
        selected(results, "one", MAX_PERFORMANCE_ISO_ENERGY).evaluation
        is full
    )
