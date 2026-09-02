from cache_optimizer.models import CacheDesign, Evaluation
from cache_optimizer.search import (
    AdaptiveParetoSearch,
    DiscreteSpace,
    MAX_PERFORMANCE,
    MIN_ENERGY,
    PARETO_FULL_CACHE,
    pareto_front,
    select_optimum,
)


def make_eval(design: CacheDesign, iteration: int) -> Evaluation:
    area = design.l1_capacity_kib / 256 * 0.2 + design.l2_capacity_kib / 4096 * 0.8
    perf = 0.8 + 0.15 * min(area, 1.5)
    energy = 0.5 + 0.4 * area
    feasible = area <= 1.0 and (design.objective == MIN_ENERGY or energy <= 1.0)
    return Evaluation(design, iteration, area, area, 0.0, 0.0, [], perf, energy, energy / perf, 1 / perf, energy, feasible, [])


def test_search_is_iterative_and_bounded():
    search = AdaptiveParetoSearch(
        technology="gain_cell",
        objective=MAX_PERFORMANCE,
        space=DiscreteSpace((128, 256, 512), (2, 4), (2048, 4096, 8192), (8, 16)),
        baseline_l1_capacity_kib=256,
        baseline_l1_associativity=4,
        baseline_l2_capacity_kib=4096,
        baseline_l2_associativity=16,
        max_evaluations=9,
        max_iterations=8,
        beam_width=3,
    )

    outcome = search.run(lambda designs, iteration: [make_eval(design, iteration) for design in designs])
    assert 1 < len(outcome.evaluations) <= 9
    assert max(e.iteration for e in outcome.evaluations) >= 1
    assert outcome.optimum is not None


def test_min_energy_keeps_l1_sram_fixed():
    search = AdaptiveParetoSearch(
        technology="stt_mram",
        objective=MIN_ENERGY,
        space=DiscreteSpace((128, 256, 512), (2, 4), (2048, 4096), (8, 16)),
        baseline_l1_capacity_kib=256,
        baseline_l1_associativity=4,
        baseline_l2_capacity_kib=4096,
        baseline_l2_associativity=16,
        max_evaluations=6,
        max_iterations=3,
        beam_width=2,
    )
    outcome = search.run(lambda designs, iteration: [make_eval(design, iteration) for design in designs])
    assert all(e.design.l1_technology == "sram" for e in outcome.evaluations)
    assert all(e.design.l1_capacity_kib == 256 for e in outcome.evaluations)


def test_failed_candidates_still_consume_the_evaluation_budget():
    search = AdaptiveParetoSearch(
        technology="gain_cell",
        objective=MAX_PERFORMANCE,
        space=DiscreteSpace((128, 256, 512), (2, 4), (2048, 4096), (8, 16)),
        baseline_l1_capacity_kib=256,
        baseline_l1_associativity=4,
        baseline_l2_capacity_kib=4096,
        baseline_l2_associativity=16,
        max_evaluations=3,
        max_iterations=10,
        beam_width=3,
    )
    batches = []

    def reject_everything(designs, iteration):
        batches.extend(designs)
        return []

    outcome = search.run(reject_everything)
    assert len(batches) == 3
    assert outcome.stopped_reason == "max_evaluations"


def test_feasible_tolerance_is_not_mistaken_for_constraint_violation():
    fast_design = CacheDesign("gc", MIN_ENERGY, "sram", "gc", 256, 4, 4096, 16)
    efficient_design = CacheDesign("gc", MIN_ENERGY, "sram", "gc", 256, 4, 2048, 16)
    fast = Evaluation(
        fast_design, 0, 1.0, 1.0, 0, 0, [], 1.0, 0.8, 0.8,
        1.0, 0.8, True, [],
    )
    # A 2% runtime increase is feasible under the configured tolerance.  It
    # must not outrank the actual min-energy objective as if it were a penalty.
    efficient = Evaluation(
        efficient_design, 0, 0.9, 0.9, 0, 0, [], 1 / 1.02, 0.5, 0.51,
        1.02, 0.5, True, [],
    )
    assert select_optimum([fast, efficient], MIN_ENERGY) is efficient


def test_pareto_front_removes_dominated_point():
    base = CacheDesign("gc", MAX_PERFORMANCE, "gc", "gc", 256, 4, 4096, 16)
    a = make_eval(base, 0)
    b = make_eval(CacheDesign("gc", MAX_PERFORMANCE, "gc", "gc", 128, 4, 2048, 16), 0)
    dominated = Evaluation(base, 0, 2.0, 2.0, 0, 0, [], 0.5, 2.0, 4.0, 2.0, 2.0, False, [])
    assert dominated not in pareto_front([a, b, dominated])


def test_power_aware_pareto_keeps_low_power_operating_point():
    design = CacheDesign(
        "gc", PARETO_FULL_CACHE, "gc", "gc", 256, 4, 4096, 16
    )
    high_power = Evaluation(
        design, 0, 0.8, 0.8, 0, 0, [], 1.2, 0.8, 2.0,
        1 / 1.2, 0.8, True, [],
    )
    low_power = Evaluation(
        design, 0, 0.9, 0.9, 0, 0, [], 1.1, 0.9, 0.8,
        1 / 1.1, 0.9, True, [],
    )
    dominated = Evaluation(
        design, 0, 1.0, 1.0, 0, 0, [], 1.0, 1.0, 1.0,
        1.0, 1.0, True, [],
    )
    front = pareto_front([high_power, low_power, dominated])
    assert high_power in front
    assert low_power in front
    assert dominated not in front


def test_unified_pareto_search_covers_full_and_l2_retrofit_families():
    search = AdaptiveParetoSearch(
        technology="gain_cell",
        objective=PARETO_FULL_CACHE,
        space=DiscreteSpace((256,), (4, 8), (2048, 4096, 8192), (16,)),
        baseline_l1_capacity_kib=256,
        baseline_l1_associativity=4,
        baseline_l2_capacity_kib=4096,
        baseline_l2_associativity=16,
        max_evaluations=9,
        max_iterations=4,
        beam_width=4,
    )
    outcome = search.run(
        lambda designs, iteration: [
            make_eval(design, iteration) for design in designs
        ]
    )
    assert len(outcome.evaluations) == 9
    assert {
        evaluation.design.candidate_family
        for evaluation in outcome.evaluations
    } == {"full_cache", "l2_retrofit"}
