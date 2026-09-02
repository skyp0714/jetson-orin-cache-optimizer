"""Dependency-free CSV, Markdown, HTML, and SVG reporting."""

from __future__ import annotations

import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .models import Evaluation, to_jsonable
from .search import (
    MAX_PERFORMANCE,
    MIN_ENERGY,
    PARETO_FULL_CACHE,
    pareto_front,
    select_optimum,
)
from .selection import (
    BALANCED_KNEE,
    MAX_PERFORMANCE_ISO_ENERGY,
    MIN_ENERGY_ISO_PERFORMANCE,
    SelectionResult,
    select_power_budget_optima,
)


COLORS = {
    "gain_cell": "#2563eb",
    "stt_mram": "#dc2626",
    "sram": "#111827",
}

SATURATION_SLOPE_THRESHOLD = 0.10


def _color(name: str) -> str:
    if name in COLORS:
        return COLORS[name]
    palette = ("#7c3aed", "#059669", "#d97706", "#0891b2", "#be185d")
    return palette[sum(ord(ch) for ch in name) % len(palette)]


def _fmt(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "n/a"
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.{digits}g}"


def write_reports(
    output_dir: Path,
    evaluations: Iterable[Evaluation],
    *,
    baseline: dict[str, Any],
    metadata: dict[str, Any],
    power_constraints: list[dict[str, Any]] | None = None,
    constraints: dict[str, Any] | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    values = list(evaluations)
    optima = _optima(values)
    resolved_power_constraints = (
        power_constraints
        if power_constraints is not None
        else metadata.get("power_constraints", [])
    )
    resolved_constraints = (
        constraints if constraints is not None else metadata.get("constraints", {})
    )
    if not isinstance(resolved_power_constraints, list):
        resolved_power_constraints = list(resolved_power_constraints or [])
    if not isinstance(resolved_constraints, dict):
        resolved_constraints = dict(resolved_constraints or {})
    power_selections = (
        select_power_budget_optima(
            values,
            resolved_power_constraints,
            resolved_constraints,
        )
        if resolved_power_constraints
        else []
    )

    paths = {
        "evaluations_csv": output_dir / "all_evaluations.csv",
        "applications_csv": output_dir / "per_application.csv",
        "optima_csv": output_dir / "optimal_configs.csv",
        "pareto_csv": output_dir / "pareto_front.csv",
        "power_optima_csv": output_dir / "power_constrained_optima.csv",
        "saturation_csv": output_dir / "saturation_points.csv",
        "results_json": output_dir / "results.json",
        "history_svg": output_dir / "optimization_history.svg",
        "tradeoff_svg": output_dir / "energy_performance_tradeoff.svg",
        "power_svg": output_dir / "power_constrained_optima.svg",
        "saturation_svg": output_dir / "area_saturation.svg",
        "summary_md": output_dir / "summary.md",
        "report_html": output_dir / "report.html",
    }

    _write_evaluations_csv(paths["evaluations_csv"], values)
    _write_applications_csv(paths["applications_csv"], values)
    _write_optima_csv(paths["optima_csv"], optima)
    _write_pareto_csv(paths["pareto_csv"], values)
    _write_power_optima_csv(paths["power_optima_csv"], power_selections)
    _write_saturation_csv(paths["saturation_csv"], values)
    paths["results_json"].write_text(
        json.dumps(
            {
                "baseline": to_jsonable(baseline),
                "metadata": to_jsonable(metadata),
                "optima": {f"{key[0]}::{key[1]}": to_jsonable(value) for key, value in optima.items()},
                "power_constrained_optima": to_jsonable(power_selections),
                "evaluations": to_jsonable(values),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    paths["history_svg"].write_text(_history_svg(values), encoding="utf-8")
    paths["tradeoff_svg"].write_text(_tradeoff_svg(values), encoding="utf-8")
    paths["power_svg"].write_text(
        _power_constrained_svg(
            values,
            power_selections,
            resolved_power_constraints,
            resolved_constraints,
        ),
        encoding="utf-8",
    )
    paths["saturation_svg"].write_text(_saturation_svg(values), encoding="utf-8")
    paths["summary_md"].write_text(
        _summary_markdown(
            optima,
            baseline,
            values,
            metadata,
            power_selections,
        ),
        encoding="utf-8",
    )
    paths["report_html"].write_text(
        _html_report(
            optima,
            baseline,
            values,
            metadata,
            power_selections,
        ),
        encoding="utf-8",
    )
    return {key: str(value) for key, value in paths.items()}


def _optima(evaluations: list[Evaluation]) -> dict[tuple[str, str], Evaluation]:
    groups: dict[tuple[str, str], list[Evaluation]] = defaultdict(list)
    for evaluation in evaluations:
        groups[(evaluation.design.technology, evaluation.design.objective)].append(evaluation)
    result: dict[tuple[str, str], Evaluation] = {}
    for key, values in groups.items():
        optimum = select_optimum(values, key[1])
        if optimum is not None:
            result[key] = optimum
    return result


def _write_evaluations_csv(path: Path, evaluations: list[Evaluation]) -> None:
    headers = [
        "candidate_id", "candidate_family", "technology", "objective", "iteration", "l1_technology", "l2_technology",
        "l1_capacity_kib", "l1_associativity", "l2_capacity_kib", "l2_associativity",
        "area_mm2", "area_ratio", "performance_score", "energy_score", "power_score",
        "power_mw_score", "worst_power_ratio", "max_power_mw",
        "worst_runtime_ratio", "worst_energy_ratio", "feasible", "constraint_reasons",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for value in evaluations:
            row = {
                "candidate_id": value.design.id,
                "candidate_family": value.design.candidate_family,
                "technology": value.design.technology,
                "objective": value.design.objective,
                "iteration": value.iteration,
                "l1_technology": value.design.l1_technology,
                "l2_technology": value.design.l2_technology,
                "l1_capacity_kib": value.design.l1_capacity_kib,
                "l1_associativity": value.design.l1_associativity,
                "l2_capacity_kib": value.design.l2_capacity_kib,
                "l2_associativity": value.design.l2_associativity,
                "area_mm2": value.area_mm2,
                "area_ratio": value.area_ratio,
                "performance_score": value.performance_score,
                "energy_score": value.energy_score,
                "power_score": value.power_score,
                "power_mw_score": value.power_mw_score,
                "worst_power_ratio": value.worst_power_ratio,
                "max_power_mw": value.max_power_mw,
                "worst_runtime_ratio": value.worst_runtime_ratio,
                "worst_energy_ratio": value.worst_energy_ratio,
                "feasible": value.feasible,
                "constraint_reasons": "; ".join(value.constraint_reasons),
            }
            writer.writerow(row)


def _write_applications_csv(path: Path, evaluations: list[Evaluation]) -> None:
    headers = [
        "candidate_id", "technology", "objective", "application", "cycles", "instructions", "ipc",
        "runtime_s", "runtime_ratio", "speedup", "cache_energy_nj", "energy_ratio",
        "cache_power_mw", "power_ratio", "l1_read_hit", "l1_read_miss", "l1_write",
        "l2_read_hit", "l2_read_miss", "l2_write",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for evaluation in evaluations:
            for app in evaluation.app_results:
                writer.writerow(
                    {
                        "candidate_id": evaluation.design.id,
                        "technology": evaluation.design.technology,
                        "objective": evaluation.design.objective,
                        "application": app.application,
                        "cycles": app.cycles,
                        "instructions": app.instructions,
                        "ipc": app.ipc,
                        "runtime_s": app.runtime_s,
                        "runtime_ratio": app.runtime_ratio,
                        "speedup": app.speedup,
                        "cache_energy_nj": app.cache_energy_nj,
                        "energy_ratio": app.energy_ratio,
                        "cache_power_mw": app.cache_power_mw,
                        "power_ratio": app.power_ratio,
                        **to_jsonable(app.accesses),
                    }
                )


def _write_optima_csv(path: Path, optima: dict[tuple[str, str], Evaluation]) -> None:
    headers = [
        "technology", "objective", "candidate_id", "status", "l1_technology", "l2_technology",
        "l1_capacity_kib", "l1_associativity", "l2_capacity_kib", "l2_associativity",
        "area_mm2", "area_ratio", "performance_score", "energy_score", "power_score",
        "worst_runtime_ratio", "worst_energy_ratio", "constraint_reasons",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for (technology, objective), value in sorted(optima.items()):
            writer.writerow(
                {
                    "technology": technology,
                    "objective": objective,
                    "candidate_id": value.design.id,
                    "status": "FEASIBLE" if value.feasible else "NO_FEASIBLE_CANDIDATE",
                    "l1_technology": value.design.l1_technology,
                    "l2_technology": value.design.l2_technology,
                    "l1_capacity_kib": value.design.l1_capacity_kib,
                    "l1_associativity": value.design.l1_associativity,
                    "l2_capacity_kib": value.design.l2_capacity_kib,
                    "l2_associativity": value.design.l2_associativity,
                    "area_mm2": value.area_mm2,
                    "area_ratio": value.area_ratio,
                    "performance_score": value.performance_score,
                    "energy_score": value.energy_score,
                    "power_score": value.power_score,
                    "worst_runtime_ratio": value.worst_runtime_ratio,
                    "worst_energy_ratio": value.worst_energy_ratio,
                    "constraint_reasons": "; ".join(value.constraint_reasons),
                }
            )


def _write_pareto_csv(path: Path, evaluations: list[Evaluation]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "candidate_id",
                "candidate_family",
                "technology",
                "objective",
                "area_ratio",
                "performance_score",
                "energy_score",
                "power_score",
                "power_mw_score",
                "worst_power_ratio",
                "max_power_mw",
                "feasible",
            )
        )
        for key, front in sorted(_grouped_pareto_fronts(evaluations).items()):
            for value in front:
                writer.writerow(
                    (
                        value.design.id,
                        value.design.candidate_family,
                        key[0],
                        key[1],
                        value.area_ratio,
                        value.performance_score,
                        value.energy_score,
                        value.power_score,
                        value.power_mw_score,
                        value.worst_power_ratio,
                        value.max_power_mw,
                        value.feasible,
                    )
                )


def _write_power_optima_csv(
    path: Path, selections: list[SelectionResult]
) -> None:
    headers = [
        "technology",
        "budget_name",
        "budget_type",
        "budget_limit",
        "budget_unit",
        "selector",
        "status",
        "candidate_id",
        "candidate_family",
        "l1_technology",
        "l2_technology",
        "l1_capacity_kib",
        "l1_associativity",
        "l2_capacity_kib",
        "l2_associativity",
        "area_ratio",
        "performance_score",
        "energy_score",
        "power_score",
        "power_mw_score",
        "worst_power_ratio",
        "max_power_mw",
        "observed_power",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for selection in selections:
            budget_type = str(selection.budget["type"])
            row: dict[str, Any] = {
                "technology": selection.technology,
                "budget_name": selection.budget["name"],
                "budget_type": budget_type,
                "budget_limit": selection.budget["limit"],
                "budget_unit": "x" if budget_type == "relative" else "mW",
                "selector": selection.selector,
                "status": selection.status,
            }
            value = selection.evaluation
            if value is not None:
                row.update(
                    {
                        "candidate_id": value.design.id,
                        "candidate_family": value.design.candidate_family,
                        "l1_technology": value.design.l1_technology,
                        "l2_technology": value.design.l2_technology,
                        "l1_capacity_kib": value.design.l1_capacity_kib,
                        "l1_associativity": value.design.l1_associativity,
                        "l2_capacity_kib": value.design.l2_capacity_kib,
                        "l2_associativity": value.design.l2_associativity,
                        "area_ratio": value.area_ratio,
                        "performance_score": value.performance_score,
                        "energy_score": value.energy_score,
                        "power_score": value.power_score,
                        "power_mw_score": value.power_mw_score,
                        "worst_power_ratio": value.worst_power_ratio,
                        "max_power_mw": value.max_power_mw,
                        "observed_power": selection.observed_power,
                    }
                )
            writer.writerow(row)


def _grouped_pareto_fronts(
    evaluations: list[Evaluation],
) -> dict[tuple[str, str], list[Evaluation]]:
    """Compute independent fronts so unlike technologies/objectives do not hide points."""
    groups: dict[tuple[str, str], list[Evaluation]] = defaultdict(list)
    for evaluation in evaluations:
        groups[(evaluation.design.technology, evaluation.design.objective)].append(evaluation)
    return {key: pareto_front(values) for key, values in groups.items()}


def _preferred_pareto_values(evaluations: list[Evaluation]) -> list[Evaluation]:
    """Use unified Pareto candidates per technology, with legacy fallbacks."""
    result: list[Evaluation] = []
    for technology in sorted({e.design.technology for e in evaluations}):
        technology_values = [
            e for e in evaluations if e.design.technology == technology
        ]
        pareto_values = [
            e
            for e in technology_values
            if e.design.objective == PARETO_FULL_CACHE
        ]
        if pareto_values:
            result.extend(pareto_values)
            continue
        max_performance_values = [
            e
            for e in technology_values
            if e.design.objective == MAX_PERFORMANCE
        ]
        result.extend(max_performance_values or technology_values)
    return result


def _saturation_points(evaluations: list[Evaluation]) -> dict[str, Evaluation | None]:
    relevant = _preferred_pareto_values(evaluations)
    # Preserve a NOT_DETECTED row even when a technology has no relevant
    # completed evaluation.
    technologies = sorted({e.design.technology for e in evaluations})
    return {
        technology: detect_saturation(
            [e for e in relevant if e.design.technology == technology]
        )
        for technology in technologies
    }


def _write_saturation_csv(path: Path, evaluations: list[Evaluation]) -> None:
    headers = [
        "technology", "status", "slope_threshold", "candidate_id", "l1_technology", "l2_technology",
        "l1_capacity_kib", "l1_associativity", "l2_capacity_kib", "l2_associativity",
        "area_mm2", "area_ratio", "performance_score", "energy_score", "power_score",
        "feasible",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for technology, value in _saturation_points(evaluations).items():
            if value is None:
                writer.writerow({
                    "technology": technology,
                    "status": "NOT_DETECTED",
                    "slope_threshold": SATURATION_SLOPE_THRESHOLD,
                })
                continue
            writer.writerow(
                {
                    "technology": technology,
                    "status": "DETECTED",
                    "slope_threshold": SATURATION_SLOPE_THRESHOLD,
                    "candidate_id": value.design.id,
                    "l1_technology": value.design.l1_technology,
                    "l2_technology": value.design.l2_technology,
                    "l1_capacity_kib": value.design.l1_capacity_kib,
                    "l1_associativity": value.design.l1_associativity,
                    "l2_capacity_kib": value.design.l2_capacity_kib,
                    "l2_associativity": value.design.l2_associativity,
                    "area_mm2": value.area_mm2,
                    "area_ratio": value.area_ratio,
                    "performance_score": value.performance_score,
                    "energy_score": value.energy_score,
                    "power_score": value.power_score,
                    "feasible": value.feasible,
                }
            )


class _Plot:
    def __init__(self, width: int, height: int, title: str) -> None:
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<style>text{font-family:Inter,Arial,sans-serif;fill:#111827}.axis{stroke:#374151;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.baseline{stroke:#111827;stroke-width:1.5;stroke-dasharray:7 5}.label{font-size:12px}.small{font-size:10px}.title{font-size:17px;font-weight:600}</style>',
            f'<rect width="100%" height="100%" fill="white"/><text x="{width / 2}" y="25" text-anchor="middle" class="title">{html.escape(title)}</text>',
        ]

    def add(self, value: str) -> None:
        self.parts.append(value)

    def finish(self) -> str:
        return "".join(self.parts + ["</svg>"])


def _bounds(values: list[float], include_one: bool = True) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if include_one:
        finite.append(1.0)
    if not finite:
        return (0.0, 1.0)
    low, high = min(finite), max(finite)
    if math.isclose(low, high):
        pad = max(abs(low) * 0.1, 0.1)
    else:
        pad = (high - low) * 0.12
    return max(0.0, low - pad), high + pad


def _panel(plot: _Plot, rect: tuple[float, float, float, float], xs: list[float], ys: list[float], x_label: str, y_label: str, x_baseline: float | None = 1.0, y_baseline: float | None = 1.0) -> tuple[Any, Any]:
    x, y, width, height = rect
    xlow, xhigh = _bounds(xs)
    ylow, yhigh = _bounds(ys)
    sx = lambda value: x + (value - xlow) / (xhigh - xlow) * width
    sy = lambda value: y + height - (value - ylow) / (yhigh - ylow) * height
    for index in range(6):
        gx = x + width * index / 5
        gy = y + height * index / 5
        plot.add(f'<line x1="{gx}" y1="{y}" x2="{gx}" y2="{y + height}" class="grid"/><line x1="{x}" y1="{gy}" x2="{x + width}" y2="{gy}" class="grid"/>')
        xv = xlow + (xhigh - xlow) * index / 5
        yv = yhigh - (yhigh - ylow) * index / 5
        plot.add(f'<text x="{gx}" y="{y + height + 17}" text-anchor="middle" class="small">{_fmt(xv, 3)}</text><text x="{x - 7}" y="{gy + 3}" text-anchor="end" class="small">{_fmt(yv, 3)}</text>')
    plot.add(f'<line x1="{x}" y1="{y + height}" x2="{x + width}" y2="{y + height}" class="axis"/><line x1="{x}" y1="{y}" x2="{x}" y2="{y + height}" class="axis"/>')
    if x_baseline is not None and xlow <= x_baseline <= xhigh:
        plot.add(f'<line x1="{sx(x_baseline)}" y1="{y}" x2="{sx(x_baseline)}" y2="{y + height}" class="baseline"/>')
    if y_baseline is not None and ylow <= y_baseline <= yhigh:
        plot.add(f'<line x1="{x}" y1="{sy(y_baseline)}" x2="{x + width}" y2="{sy(y_baseline)}" class="baseline"/>')
    plot.add(f'<text x="{x + width / 2}" y="{y + height + 38}" text-anchor="middle" class="label">{html.escape(x_label)}</text><text x="{x - 48}" y="{y + height / 2}" text-anchor="middle" class="label" transform="rotate(-90 {x - 48} {y + height / 2})">{html.escape(y_label)}</text>')
    return sx, sy


def _legend(plot: _Plot, technologies: list[str], x: float, y: float) -> None:
    for index, technology in enumerate(technologies):
        offset = index * 105
        plot.add(f'<circle cx="{x + offset}" cy="{y}" r="5" fill="{_color(technology)}"/><text x="{x + offset + 9}" y="{y + 4}" class="small">{html.escape(technology)}</text>')


def _history_svg(evaluations: list[Evaluation]) -> str:
    plot = _Plot(1000, 650, "Iterative optimization history (SRAM baseline = 1.0)")
    pareto_mode = any(
        evaluation.design.objective == PARETO_FULL_CACHE
        for evaluation in evaluations
    )
    panels = [
        (
            (75, 65, 875, 225),
            "energy_score",
            "Best area-feasible normalized energy",
            PARETO_FULL_CACHE if pareto_mode else MIN_ENERGY,
        ),
        (
            (75, 365, 875, 225),
            "performance_score",
            "Best area-feasible speedup",
            PARETO_FULL_CACHE if pareto_mode else MAX_PERFORMANCE,
        ),
    ]
    technologies = sorted({e.design.technology for e in evaluations})
    for rect, attribute, ylabel, objective in panels:
        objective_values = [
            e for e in evaluations if e.design.objective == objective
        ]
        iterations = [float(e.iteration) for e in objective_values] or [0.0]
        metric_values = [float(getattr(e, attribute)) for e in objective_values]
        sx, sy = _panel(plot, rect, iterations, metric_values, "Iteration", ylabel, x_baseline=None, y_baseline=1.0)
        for technology in technologies:
            points = [e for e in evaluations if e.design.technology == technology and e.design.objective == objective]
            best_by_iteration: list[tuple[int, float]] = []
            running: float | None = None
            for iteration in sorted({point.iteration for point in points}):
                feasible = [point for point in points if point.iteration <= iteration and point.feasible]
                if feasible:
                    metric = (
                        min(p.energy_score for p in feasible)
                        if attribute == "energy_score"
                        else max(p.performance_score for p in feasible)
                    )
                    running = metric
                if running is not None:
                    best_by_iteration.append((iteration, running))
            if best_by_iteration:
                path = " ".join(("M" if index == 0 else "L") + f" {sx(x):.2f} {sy(y):.2f}" for index, (x, y) in enumerate(best_by_iteration))
                plot.add(f'<path d="{path}" fill="none" stroke="{_color(technology)}" stroke-width="2.3"/>')
                for x, y in best_by_iteration:
                    plot.add(f'<circle cx="{sx(x)}" cy="{sy(y)}" r="3.5" fill="{_color(technology)}"/>')
    _legend(plot, technologies, 90, 335)
    return plot.finish()


def _tradeoff_svg(evaluations: list[Evaluation]) -> str:
    plot = _Plot(900, 600, "Cache energy vs application performance")
    relevant = _preferred_pareto_values(evaluations)
    xs = [e.energy_score for e in relevant]
    ys = [e.performance_score for e in relevant]
    sx, sy = _panel(plot, (85, 60, 760, 465), xs, ys, "Normalized cache energy (lower is better)", "Geometric-mean speedup (higher is better)")
    technologies = sorted({e.design.technology for e in relevant})
    front_ids = {
        e.design.id
        for front in _grouped_pareto_fronts(relevant).values()
        for e in front
    }
    for evaluation in relevant:
        radius = 6 if evaluation.design.id in front_ids else 4
        opacity = 1.0 if evaluation.feasible else 0.35
        stroke = "#111827" if evaluation.design.id in front_ids else "none"
        plot.add(f'<circle cx="{sx(evaluation.energy_score)}" cy="{sy(evaluation.performance_score)}" r="{radius}" fill="{_color(evaluation.design.technology)}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="1.2"><title>{html.escape(evaluation.design.id)} | area={evaluation.area_ratio:.3f} | feasible={evaluation.feasible}</title></circle>')
    _legend(plot, technologies, 100, 565)
    plot.add('<text x="515" y="569" class="small">outlined = 4D technology/objective Pareto frontier; faded = constraint violation</text>')
    return plot.finish()


def _selection_candidate_values(evaluations: list[Evaluation]) -> list[Evaluation]:
    """Mirror power selection's unified-Pareto preference for plotting."""
    result: list[Evaluation] = []
    for technology in sorted({e.design.technology for e in evaluations}):
        technology_values = [
            e for e in evaluations if e.design.technology == technology
        ]
        pareto_values = [
            e
            for e in technology_values
            if e.design.objective == PARETO_FULL_CACHE
        ]
        result.extend(pareto_values or technology_values)
    return result


def _observed_power_for_budget(
    evaluation: Evaluation,
    budget: dict[str, Any],
    scope: str,
) -> float:
    if budget.get("type") == "relative":
        if scope == "suite":
            return float(evaluation.power_score)
        app_values = [
            float(app.power_ratio)
            for app in evaluation.app_results
            if math.isfinite(app.power_ratio)
        ]
        if app_values:
            return max(app_values)
        if math.isfinite(evaluation.worst_power_ratio):
            return float(evaluation.worst_power_ratio)
        return float(evaluation.power_score)

    if scope == "suite" and math.isfinite(evaluation.power_mw_score):
        return float(evaluation.power_mw_score)
    app_values = [
        float(app.cache_power_mw)
        for app in evaluation.app_results
        if math.isfinite(app.cache_power_mw)
    ]
    if app_values:
        return max(app_values)
    if math.isfinite(evaluation.max_power_mw):
        return float(evaluation.max_power_mw)
    return float(evaluation.power_mw_score)


def _power_constrained_svg(
    evaluations: list[Evaluation],
    selections: list[SelectionResult],
    power_constraints: list[dict[str, Any]],
    constraints: dict[str, Any],
) -> str:
    if not power_constraints:
        plot = _Plot(900, 180, "Power-constrained Pareto selections")
        plot.add(
            '<text x="450" y="95" text-anchor="middle" class="label">'
            "No power budgets configured.</text>"
        )
        return plot.finish()

    panel_height = 250
    height = 85 + panel_height * len(power_constraints) + 55
    plot = _Plot(1050, height, "Power-constrained Pareto selections")
    scope = str(constraints.get("constraint_scope", "per_application"))
    candidates = _selection_candidate_values(evaluations)
    technologies = sorted({e.design.technology for e in candidates})
    selector_markers = {
        MIN_ENERGY_ISO_PERFORMANCE: "E",
        MAX_PERFORMANCE_ISO_ENERGY: "P",
        BALANCED_KNEE: "K",
    }

    for index, budget in enumerate(power_constraints):
        budget_type = str(budget.get("type"))
        limit = float(budget.get("limit", math.nan))
        observed = [
            (value, _observed_power_for_budget(value, budget, scope))
            for value in candidates
        ]
        all_points = [
            (value, power)
            for value, power in observed
            if math.isfinite(power) and math.isfinite(value.performance_score)
        ]
        # Thermal outliers (notably the current STT model) can otherwise make
        # all cap-feasible points indistinguishable at the left edge.  Show the
        # decision region through 2x the cap and report clipped counts.
        display_limit = limit * 2.0
        points = [
            (value, power)
            for value, power in all_points
            if power <= display_limit + 1e-12
        ]
        hidden_by_technology: dict[str, int] = {}
        for value, power in all_points:
            if power > display_limit + 1e-12:
                hidden_by_technology[value.design.technology] = (
                    hidden_by_technology.get(value.design.technology, 0) + 1
                )
        xs = [power for _, power in points]
        ys = [value.performance_score for value, _ in points]
        if math.isfinite(limit):
            xs.extend((limit, display_limit))
        unit = "SRAM x" if budget_type == "relative" else "mW"
        top = 65 + index * panel_height
        plot.add(
            f'<text x="90" y="{top - 8}" class="label">'
            f'{html.escape(str(budget.get("name", "unnamed")))}: '
            f'power ≤ {_fmt(limit)} {html.escape(unit)} '
            f'({html.escape(scope)})</text>'
        )
        if hidden_by_technology:
            clipped = ", ".join(
                "{}={}".format(technology, count)
                for technology, count in sorted(hidden_by_technology.items())
            )
            plot.add(
                f'<text x="940" y="{top - 8}" text-anchor="end" class="small">'
                f'above 2x-cap view: {html.escape(clipped)}</text>'
            )
        sx, sy = _panel(
            plot,
            (90, top, 850, 175),
            xs,
            ys,
            f"Observed average cache-array power ({unit})",
            "Application speedup",
            x_baseline=limit if math.isfinite(limit) else None,
            y_baseline=1.0,
        )
        for value, power in points:
            opacity = 1.0 if value.feasible else 0.35
            plot.add(
                f'<circle cx="{sx(power)}" cy="{sy(value.performance_score)}" '
                f'r="4" fill="{_color(value.design.technology)}" '
                f'fill-opacity="{opacity}"><title>'
                f'{html.escape(value.design.id)} | power={_fmt(power)} {html.escape(unit)} '
                f'| speedup={_fmt(value.performance_score)}'
                "</title></circle>"
            )

        panel_selections = [
            selection
            for selection in selections
            if selection.budget.get("name") == budget.get("name")
            and selection.evaluation is not None
            and selection.observed_power is not None
        ]
        for marker_index, selection in enumerate(panel_selections):
            value = selection.evaluation
            if value is None or selection.observed_power is None:
                continue
            power = float(selection.observed_power)
            if not (
                math.isfinite(power)
                and math.isfinite(value.performance_score)
            ):
                continue
            marker = selector_markers.get(selection.selector, "?")
            px = sx(power)
            py = sy(value.performance_score)
            offset = (marker_index % 3 - 1) * 11
            plot.add(
                f'<circle cx="{px}" cy="{py}" r="9" fill="white" '
                f'fill-opacity="0.72" stroke="{_color(selection.technology)}" '
                'stroke-width="2"/>'
                f'<text x="{px + offset}" y="{py - 11}" text-anchor="middle" '
                f'class="small" font-weight="700">{marker}<title>'
                f'{html.escape(selection.technology)} | '
                f'{html.escape(selection.selector)} | '
                f'power={_fmt(power)} {html.escape(selection.observed_unit)} | '
                f'speedup={_fmt(value.performance_score)}'
                "</title></text>"
            )

    _legend(plot, technologies, 100, height - 25)
    plot.add(
        f'<text x="610" y="{height - 21}" class="small">'
        "E = min energy @ iso-performance; P = max performance @ iso-energy; "
        "K = balanced knee</text>"
    )
    return plot.finish()


def _saturation_envelope(values: list[Evaluation]) -> list[Evaluation]:
    ordered = sorted(values, key=lambda e: (e.area_ratio, -e.performance_score))
    envelope: list[Evaluation] = []
    best = -math.inf
    for value in ordered:
        if value.performance_score > best + 1e-12:
            envelope.append(value)
            best = value.performance_score
    return envelope


def detect_saturation(
    values: list[Evaluation], slope_threshold: float = SATURATION_SLOPE_THRESHOLD
) -> Evaluation | None:
    """Return the first best-so-far point followed by a stable low-gain region.

    The derivative is expressed as normalized performance gain per normalized
    area gain, hence the default 0.10 threshold.  Detection uses the sampled
    running-best curve, including flat segments created by dominated larger
    designs.  Dropping those segments would make a true plateau impossible to
    detect when the last performance record is followed only by equal or worse
    points.
    """
    ordered = sorted(values, key=lambda e: (e.area_ratio, -e.performance_score))
    if len(ordered) < 3:
        return None

    running: list[tuple[float, float, Evaluation]] = []
    best = ordered[0]
    for value in ordered:
        if value.performance_score > best.performance_score + 1e-12:
            best = value
        running.append((value.area_ratio, best.performance_score, best))

    for index in range(1, len(running) - 1):
        left_area, left_performance, _ = running[index - 1]
        area, performance, best_at_area = running[index]
        right_area, right_performance, _ = running[index + 1]
        slope1 = (performance - left_performance) / max(area - left_area, 1e-12)
        slope2 = (right_performance - performance) / max(right_area - area, 1e-12)
        if slope1 < slope_threshold and slope2 < slope_threshold:
            return best_at_area
    return None


def _saturation_svg(evaluations: list[Evaluation]) -> str:
    relevant = _preferred_pareto_values(evaluations)
    plot = _Plot(1000, 700, "Area scaling and saturation (dashed lines = SRAM bounds)")
    technologies = sorted({e.design.technology for e in relevant})
    panels = [
        ((80, 65, 870, 230), [e.performance_score for e in relevant], "Geometric-mean speedup", "performance_score"),
        ((80, 390, 870, 230), [e.power_score for e in relevant], "Normalized average cache power", "power_score"),
    ]
    xs = [e.area_ratio for e in relevant]
    for rect, ys, ylabel, attribute in panels:
        sx, sy = _panel(plot, rect, xs, ys, "Normalized cache-array area", ylabel)
        for technology in technologies:
            values = [e for e in relevant if e.design.technology == technology]
            # Both panels connect the exact same designs selected by the
            # performance envelope.  This preserves the power/performance
            # correspondence instead of drawing an unrelated power trace.
            envelope = _saturation_envelope(values)
            if envelope:
                path = " ".join(("M" if index == 0 else "L") + f" {sx(e.area_ratio):.2f} {sy(getattr(e, attribute)):.2f}" for index, e in enumerate(envelope))
                plot.add(f'<path d="{path}" fill="none" stroke="{_color(technology)}" stroke-width="2"/>')
            for value in values:
                opacity = 1.0 if value.feasible else 0.35
                plot.add(f'<circle cx="{sx(value.area_ratio)}" cy="{sy(getattr(value, attribute))}" r="4" fill="{_color(technology)}" fill-opacity="{opacity}"><title>{html.escape(value.design.id)}</title></circle>')
            if attribute == "performance_score":
                saturation = detect_saturation(values)
                if saturation is not None:
                    px, py = sx(saturation.area_ratio), sy(saturation.performance_score)
                    plot.add(f'<circle cx="{px}" cy="{py}" r="8" fill="none" stroke="{_color(technology)}" stroke-width="2.5"/><text x="{px + 10}" y="{py - 9}" class="small">saturation</text>')
    _legend(plot, technologies, 100, 350)
    plot.add('<text x="620" y="354" class="small">faded points exceed at least one strict objective constraint</text>')
    return plot.finish()


def _summary_rows(optima: dict[tuple[str, str], Evaluation]) -> str:
    rows = []
    for (technology, objective), value in sorted(optima.items()):
        objective_label = {
            MIN_ENERGY: "min energy @ runtime",
            MAX_PERFORMANCE: "max performance @ energy",
            PARETO_FULL_CACHE: "unified 4D Pareto search",
        }.get(objective, objective)
        status = "FEASIBLE" if value.feasible else "NO FEASIBLE POINT (closest shown)"
        rows.append(
            "| " + " | ".join(
                (
                    technology,
                    objective_label,
                    status,
                    f"{value.design.l1_capacity_kib}/{value.design.l1_associativity}",
                    f"{value.design.l2_capacity_kib}/{value.design.l2_associativity}",
                    _fmt(value.area_ratio),
                    _fmt(value.performance_score),
                    _fmt(value.energy_score),
                    _fmt(value.power_score),
                )
            ) + " |"
        )
    return "\n".join(rows)


def _selector_label(selector: str) -> str:
    return {
        MIN_ENERGY_ISO_PERFORMANCE: "min energy @ iso-performance",
        MAX_PERFORMANCE_ISO_ENERGY: "max performance @ iso-energy",
        BALANCED_KNEE: "balanced Pareto knee",
    }.get(selector, selector)


def _power_selection_rows_markdown(
    selections: list[SelectionResult],
) -> str:
    if not selections:
        return "| n/a | n/a | n/a | NOT CONFIGURED | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
    rows = []
    for selection in selections:
        budget_type = str(selection.budget["type"])
        unit = "x" if budget_type == "relative" else "mW"
        limit = f"{_fmt(float(selection.budget['limit']))} {unit}"
        value = selection.evaluation
        if value is None:
            reason = "; ".join(selection.reasons) or "no eligible evaluated point"
            rows.append(
                "| "
                + " | ".join(
                    (
                        selection.technology,
                        str(selection.budget["name"]),
                        limit,
                        _selector_label(selection.selector),
                        selection.status,
                        "n/a",
                        "n/a",
                        "n/a",
                        "n/a",
                        "n/a",
                        "n/a",
                        reason.replace("|", "\\|"),
                    )
                )
                + " |"
            )
            continue
        rows.append(
            "| "
            + " | ".join(
                (
                    selection.technology,
                    str(selection.budget["name"]),
                    limit,
                    _selector_label(selection.selector),
                    selection.status,
                    value.design.candidate_family,
                    f"{value.design.l1_capacity_kib}/{value.design.l1_associativity}; "
                    f"{value.design.l2_capacity_kib}/{value.design.l2_associativity}",
                    _fmt(value.area_ratio),
                    _fmt(value.performance_score),
                    _fmt(value.energy_score),
                    _fmt(selection.observed_power or math.nan),
                    "—",
                )
            )
            + " |"
        )
    return "\n".join(rows)


def _saturation_rows_markdown(evaluations: list[Evaluation]) -> str:
    rows = []
    for technology, value in _saturation_points(evaluations).items():
        if value is None:
            rows.append(f"| {technology} | NOT DETECTED | n/a | n/a | n/a | n/a | n/a |")
            continue
        rows.append(
            "| " + " | ".join(
                (
                    technology,
                    "DETECTED",
                    "FEASIBLE" if value.feasible else "INFEASIBLE",
                    f"{value.design.l1_capacity_kib}/{value.design.l1_associativity}",
                    f"{value.design.l2_capacity_kib}/{value.design.l2_associativity}",
                    _fmt(value.area_ratio),
                    _fmt(value.performance_score),
                )
            ) + " |"
        )
    return "\n".join(rows) or "| n/a | NOT DETECTED | n/a | n/a | n/a | n/a | n/a |"


def _metadata_items(value: Any) -> list[str]:
    if value is None or value == {} or value == []:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    result = []
    for item in values:
        if isinstance(item, str):
            result.append(item)
        else:
            result.append(json.dumps(to_jsonable(item), sort_keys=True))
    return result


def _metadata_mapping_items(value: Any) -> list[tuple[str, str]]:
    if not isinstance(value, dict):
        return [("value", item) for item in _metadata_items(value)]
    result = []
    for key, item in sorted(value.items()):
        if isinstance(item, str):
            rendered = item
        else:
            rendered = json.dumps(to_jsonable(item), sort_keys=True)
        result.append((str(key), rendered))
    return result


def _model_notes(metadata: dict[str, Any]) -> Any:
    for key in ("model_notes", "technology_model_notes"):
        if key in metadata:
            return metadata[key]
    technologies = metadata.get("technologies")
    if isinstance(technologies, dict):
        notes = {
            str(name): spec["model_note"]
            for name, spec in technologies.items()
            if isinstance(spec, dict) and spec.get("model_note")
        }
        if notes:
            return notes
    return {}


def _markdown_metadata(metadata: dict[str, Any]) -> str:
    constraints = _metadata_mapping_items(metadata.get("constraints", {}))
    assumptions = _metadata_mapping_items(metadata.get("assumptions", {}))
    model_notes = _metadata_mapping_items(_model_notes(metadata))
    warnings = _metadata_items(metadata.get("warnings", []))
    errors = _metadata_items(metadata.get("errors", []))

    def bullets(items: list[tuple[str, str]]) -> str:
        return "\n".join(f"- `{key}`: {value}" for key, value in items) or "- None recorded"

    def messages(items: list[str]) -> str:
        return "\n".join(f"- {value}" for value in items) or "- None"

    return f"""## Constraints

{bullets(constraints)}

## Modeling assumptions

{bullets(assumptions)}

## Model notes

{bullets(model_notes)}

## Warnings

{messages(warnings)}

## Errors

{messages(errors)}
"""


def _search_coverage_rows_markdown(metadata: dict[str, Any]) -> str:
    outcomes = metadata.get("outcomes", {})
    if not isinstance(outcomes, dict) or not outcomes:
        return "| n/a | n/a | n/a | not recorded |"
    rows = []
    for label, outcome in sorted(outcomes.items()):
        value = outcome if isinstance(outcome, dict) else {}
        rows.append(
            "| {} | {} | {} | {} |".format(
                label,
                value.get("attempted", "n/a"),
                value.get("evaluations", "n/a"),
                value.get("stopped_reason", "unknown"),
            )
        )
    return "\n".join(rows)


def _summary_markdown(
    optima: dict[tuple[str, str], Evaluation],
    baseline: dict[str, Any],
    evaluations: list[Evaluation],
    metadata: dict[str, Any],
    power_selections: list[SelectionResult],
) -> str:
    return f"""# NS-Cache + Accel-Sim optimization summary

SRAM is the immutable reference: area, performance, energy, and average cache-array power are normalized to 1.0. A candidate marked feasible satisfies every configured hard constraint. Selections below are the best **evaluated** candidates within the recorded search budget, not a proof of the global optimum.

## Best evaluated configurations

| Technology | Objective | Status | L1 KiB/assoc | L2 KiB/assoc | Area | Speedup | Energy | Power |
|---|---|---|---:|---:|---:|---:|---:|---:|
{_summary_rows(optima)}

Evaluated candidates: {len(evaluations)}  
Baseline cache area: {_fmt(float(baseline.get('area_mm2', math.nan)))} mm²  
Run status: {metadata.get('status', 'unknown')}

## Optima by power budget

Each budget is applied after the common Pareto search. `min energy @ iso-performance` searches the SRAM-L1/target-L2 retrofit family and enforces the configured runtime tolerance. `max performance @ iso-energy` searches the target-L1/L2 full-cache family and enforces the configured energy tolerance. `balanced Pareto knee` chooses a cross-family compromise point on the budget-specific frontier.

| Technology | Power budget | Limit | Selector | Status | Family | L1; L2 KiB/assoc | Area | Speedup | Energy | Observed power | Reason |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---|
{_power_selection_rows_markdown(power_selections)}

## Search coverage

| Technology/objective | Attempted | Completed evaluations | Stop reason |
|---|---:|---:|---|
{_search_coverage_rows_markdown(metadata)}

## Area saturation points

The knee is reported independently for each technology from the unified Pareto search when available, otherwise from the legacy max-performance search, using a fixed normalized slope threshold of {SATURATION_SLOPE_THRESHOLD:.2f}. `NOT DETECTED` means the sampled performance envelope never met that rule; it does not prove that no saturation exists outside the sampled points.

| Technology | Status | Feasibility | L1 KiB/assoc | L2 KiB/assoc | Area | Speedup |
|---|---|---|---:|---:|---:|---:|
{_saturation_rows_markdown(evaluations)}

{_markdown_metadata(metadata)}

## Artifacts

- `optimal_configs.csv`: selected configurations and feasibility status
- `all_evaluations.csv`: every simulated candidate and aggregate metric
- `per_application.csv`: application-level runtime, energy, and access counts
- `pareto_front.csv`: nondominated area/energy/performance/power points
- `power_constrained_optima.csv`: per-budget energy, performance, and knee selections
- `saturation_points.csv`: per-technology saturation configuration and metrics
- `optimization_history.svg`: iterative convergence
- `energy_performance_tradeoff.svg`: energy/performance Pareto view
- `power_constrained_optima.svg`: power/performance panels with budget lines and E/P/K selections
- `area_saturation.svg`: performance/power vs area with SRAM dashed bounds and detected knee
- `results.json`: complete machine-readable provenance and results

## Interpretation boundary

Performance is GPU trace runtime (`gpu_tot_sim_cycle`) at the configured clock, not host wall-clock end-to-end latency. Cache energy is NS-Cache dynamic access energy plus leakage and configured refresh energy over that simulated runtime. Reported power is average **cache-array power over the simulated trace**; it is not Jetson module power, whole-GPU power, or TDP. CPU preprocessing/postprocessing and DRAM/system energy are not silently added.
"""


def _html_report(
    optima: dict[tuple[str, str], Evaluation],
    baseline: dict[str, Any],
    evaluations: list[Evaluation],
    metadata: dict[str, Any],
    power_selections: list[SelectionResult],
) -> str:
    rows = []
    for (technology, objective), value in sorted(optima.items()):
        status = "Feasible" if value.feasible else "No feasible point; closest shown"
        rows.append(
            f"<tr><td>{html.escape(technology)}</td><td>{html.escape(objective)}</td><td>{status}</td>"
            f"<td>{value.design.l1_capacity_kib} / {value.design.l1_associativity}</td>"
            f"<td>{value.design.l2_capacity_kib} / {value.design.l2_associativity}</td>"
            f"<td>{_fmt(value.area_ratio)}</td><td>{_fmt(value.performance_score)}</td>"
            f"<td>{_fmt(value.energy_score)}</td><td>{_fmt(value.power_score)}</td></tr>"
        )
    saturation_rows = []
    for technology, value in _saturation_points(evaluations).items():
        if value is None:
            saturation_rows.append(
                f"<tr><td>{html.escape(technology)}</td><td>Not detected</td>"
                "<td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td></tr>"
            )
            continue
        saturation_rows.append(
            f"<tr><td>{html.escape(technology)}</td><td>Detected</td>"
            f"<td>{'Feasible' if value.feasible else 'Infeasible'}</td>"
            f"<td>{value.design.l1_capacity_kib} / {value.design.l1_associativity}</td>"
            f"<td>{value.design.l2_capacity_kib} / {value.design.l2_associativity}</td>"
            f"<td>{_fmt(value.area_ratio)}</td><td>{_fmt(value.performance_score)}</td></tr>"
        )
    coverage_rows = []
    outcomes = metadata.get("outcomes", {})
    if isinstance(outcomes, dict):
        for label, outcome in sorted(outcomes.items()):
            value = outcome if isinstance(outcome, dict) else {}
            coverage_rows.append(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                    html.escape(str(label)),
                    html.escape(str(value.get("attempted", "n/a"))),
                    html.escape(str(value.get("evaluations", "n/a"))),
                    html.escape(str(value.get("stopped_reason", "unknown"))),
                )
            )
    if not coverage_rows:
        coverage_rows.append(
            "<tr><td>n/a</td><td>n/a</td><td>n/a</td><td>not recorded</td></tr>"
        )
    power_rows = []
    for selection in power_selections:
        budget_type = str(selection.budget["type"])
        unit = "x" if budget_type == "relative" else "mW"
        budget_limit = (
            f"{_fmt(float(selection.budget['limit']))} {html.escape(unit)}"
        )
        value = selection.evaluation
        if value is None:
            reason = "; ".join(selection.reasons) or "no eligible evaluated point"
            power_rows.append(
                f"<tr><td>{html.escape(selection.technology)}</td>"
                f"<td>{html.escape(str(selection.budget['name']))}</td>"
                f"<td>{budget_limit}</td>"
                f"<td>{html.escape(_selector_label(selection.selector))}</td>"
                f"<td>{html.escape(selection.status)}</td>"
                "<td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td>"
                f"<td>{html.escape(reason)}</td></tr>"
            )
            continue
        observed = (
            f"{_fmt(float(selection.observed_power))} "
            f"{html.escape(selection.observed_unit)}"
            if selection.observed_power is not None
            else "n/a"
        )
        power_rows.append(
            f"<tr><td>{html.escape(selection.technology)}</td>"
            f"<td>{html.escape(str(selection.budget['name']))}</td>"
            f"<td>{budget_limit}</td>"
            f"<td>{html.escape(_selector_label(selection.selector))}</td>"
            f"<td>{html.escape(selection.status)}</td>"
            f"<td>{html.escape(value.design.candidate_family)}</td>"
            f"<td>{value.design.l1_capacity_kib}/{value.design.l1_associativity}; "
            f"{value.design.l2_capacity_kib}/{value.design.l2_associativity}</td>"
            f"<td>{_fmt(value.area_ratio)}</td>"
            f"<td>{_fmt(value.performance_score)} / {_fmt(value.energy_score)}</td>"
            f"<td>{observed}</td></tr>"
        )
    if not power_rows:
        power_rows.append(
            "<tr><td>n/a</td><td>n/a</td><td>n/a</td>"
            "<td>Not configured</td><td>n/a</td><td>n/a</td>"
            "<td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td></tr>"
        )

    def html_key_values(value: Any) -> str:
        items = _metadata_mapping_items(value)
        if not items:
            return "<li>None recorded</li>"
        return "".join(
            f"<li><code>{html.escape(key)}</code>: {html.escape(rendered)}</li>"
            for key, rendered in items
        )

    def html_messages(value: Any) -> str:
        items = _metadata_items(value)
        if not items:
            return "<li>None</li>"
        return "".join(f"<li>{html.escape(item)}</li>" for item in items)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cache optimization report</title>
<style>body{{font-family:Inter,Arial,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#111827}}h1,h2{{line-height:1.2}}.note{{background:#f3f4f6;border-left:4px solid #4b5563;padding:12px 16px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #d1d5db;padding:7px;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){{text-align:left}}img{{width:100%;height:auto;border:1px solid #e5e7eb;margin:12px 0}}code{{background:#f3f4f6;padding:2px 4px}}</style></head>
<body><h1>NS-Cache + Accel-Sim optimization</h1>
<p><strong>Run status: {html.escape(str(metadata.get('status', 'unknown')))}</strong>. SRAM is immutable and normalized to 1.0. Evaluated {len(evaluations)} candidates. Baseline cache area: {_fmt(float(baseline.get('area_mm2', math.nan)))} mm².</p>
<div class="note">The selections are the best evaluated candidates within the recorded search budget, not proof of a global optimum. An <code>incomplete</code> status means at least one technology/objective produced no completed candidate.</div>
<h2>Best evaluated configurations</h2>
<table><thead><tr><th>Technology</th><th>Objective</th><th>Status</th><th>L1 KiB/assoc</th><th>L2 KiB/assoc</th><th>Area</th><th>Speedup</th><th>Energy</th><th>Power</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Optima by power budget</h2>
<p>Each budget is applied to the common evaluated Pareto candidates. Minimum energy at iso-performance is selected from the SRAM-L1/target-L2 retrofit family; maximum performance at iso-energy is selected from the target-L1/L2 full-cache family; the balanced knee compares both families.</p>
<table><thead><tr><th>Technology</th><th>Budget</th><th>Limit</th><th>Selector</th><th>Status</th><th>Family</th><th>L1; L2 KiB/assoc</th><th>Area</th><th>Speedup / Energy</th><th>Observed power</th></tr></thead><tbody>{''.join(power_rows)}</tbody></table>
<h2>Search coverage</h2>
<table><thead><tr><th>Technology/objective</th><th>Attempted</th><th>Completed</th><th>Stop reason</th></tr></thead><tbody>{''.join(coverage_rows)}</tbody></table>
<h2>Area saturation points</h2>
<p>The knee is detected independently on each technology's unified Pareto candidates when available, otherwise on the legacy max-performance candidates, using normalized slope threshold {SATURATION_SLOPE_THRESHOLD:.2f}. Not detected does not prove that saturation is absent outside the sampled points.</p>
<table><thead><tr><th>Technology</th><th>Status</th><th>Feasibility</th><th>L1 KiB/assoc</th><th>L2 KiB/assoc</th><th>Area</th><th>Speedup</th></tr></thead><tbody>{''.join(saturation_rows)}</tbody></table>
<h2>Constraints</h2><ul>{html_key_values(metadata.get('constraints', {}))}</ul>
<h2>Modeling assumptions</h2><ul>{html_key_values(metadata.get('assumptions', {}))}</ul>
<h2>Model notes</h2><ul>{html_key_values(_model_notes(metadata))}</ul>
<h2>Warnings</h2><ul>{html_messages(metadata.get('warnings', []))}</ul>
<h2>Errors</h2><ul>{html_messages(metadata.get('errors', []))}</ul>
<h2>Optimization history</h2><img src="optimization_history.svg" alt="Optimization history">
<h2>Energy/performance tradeoff</h2><img src="energy_performance_tradeoff.svg" alt="Energy performance tradeoff">
<h2>Power-constrained optima</h2><img src="power_constrained_optima.svg" alt="Power-constrained Pareto selections">
<h2>Area saturation</h2><img src="area_saturation.svg" alt="Area saturation">
<div class="note"><strong>Interpretation boundary.</strong> Performance is GPU trace runtime from <code>gpu_tot_sim_cycle</code>, not CPU-inclusive wall time. Energy is cache-array energy from candidate-specific access counts, leakage, and configured refresh. Power is average cache-array power over the simulated trace; it is not Jetson module power, whole-GPU power, or TDP. See <a href="results.json">results.json</a> for provenance and exact assumptions.</div>
<p>Data: <a href="optimal_configs.csv">optimal configs</a> · <a href="power_constrained_optima.csv">power-constrained optima</a> · <a href="all_evaluations.csv">all evaluations</a> · <a href="per_application.csv">per application</a> · <a href="pareto_front.csv">Pareto front</a> · <a href="saturation_points.csv">saturation points</a></p>
</body></html>"""
