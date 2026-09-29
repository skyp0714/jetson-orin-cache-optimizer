"""Post-process completed optimizer runs with an estimated LPDDR5 model.

This module deliberately does not pretend to be a vendor-calibrated DRAM power
simulator.  It combines the existing Accel-Sim DRAM request counts with explicit
per-request energy and background-power assumptions, then redraws the Pareto
projection using L1+L2 cache power plus estimated LPDDR5 power.  No simulation
is rerun.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .models import CacheDesign


_DRAM_READS = re.compile(r"^total dram reads = (\d+)\s*$", re.MULTILINE)
_DRAM_WRITES = re.compile(r"^total dram writes = (\d+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class DramPowerModel:
    name: str
    read_energy_nj: float
    write_energy_nj: float
    background_power_mw: float
    transaction_bytes: int = 128

    def validate(self) -> None:
        for label, value in (
            ("read_energy_nj", self.read_energy_nj),
            ("write_energy_nj", self.write_energy_nj),
            ("background_power_mw", self.background_power_mw),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        if self.transaction_bytes <= 0:
            raise ValueError("transaction_bytes must be positive")


DEFAULT_MODELS = (
    DramPowerModel("low", 2.5, 2.0, 300.0),
    DramPowerModel("nominal", 4.5, 4.0, 500.0),
    DramPowerModel("high", 6.5, 6.0, 800.0),
)


@dataclass(frozen=True)
class EstimatedPoint:
    candidate_id: str
    technology: str
    feasible: bool
    performance: float
    cache_power_mw: float
    dram_reads: int
    dram_writes: int
    runtime_s: float
    counts_source: str
    scenario_power_mw: dict[str, float]
    scenario_ratio: dict[str, float]


def estimate_dram_power_mw(
    reads: int, writes: int, runtime_s: float, model: DramPowerModel
) -> tuple[float, float]:
    """Return (dynamic, total) estimated DRAM power in mW."""
    model.validate()
    if reads < 0 or writes < 0:
        raise ValueError("DRAM request counts must be non-negative")
    if not math.isfinite(runtime_s) or runtime_s <= 0:
        raise ValueError("runtime_s must be finite and positive")
    dynamic_mw = (
        reads * model.read_energy_nj + writes * model.write_energy_nj
    ) / runtime_s / 1e6
    return dynamic_mw, dynamic_mw + model.background_power_mw


def parse_dram_counts(path: Path) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    reads = _DRAM_READS.findall(text)
    writes = _DRAM_WRITES.findall(text)
    if not reads or not writes:
        raise ValueError(f"DRAM totals are missing from {path}")
    return int(reads[-1]), int(writes[-1])


def _signature(app: dict[str, Any]) -> tuple[int, ...]:
    accesses = app["accesses"]
    return tuple(
        int(accesses[name])
        for name in (
            "l1_read_hit",
            "l1_read_miss",
            "l1_write",
            "l2_read_hit",
            "l2_read_miss",
            "l2_write",
        )
    )


def _only_log(run_dir: Path, candidate_id: str, application: str) -> Path | None:
    matches = sorted(
        (run_dir / "work" / "simulations" / candidate_id).glob(
            f"{application}-*/accelsim.stdout.log"
        )
    )
    return matches[0] if len(matches) == 1 else None


def _frontier(points: Iterable[EstimatedPoint], scenario: str) -> list[EstimatedPoint]:
    ordered = sorted(
        points,
        key=lambda point: (-point.performance, point.scenario_ratio[scenario]),
    )
    result: list[EstimatedPoint] = []
    lowest_power = math.inf
    for point in ordered:
        power = point.scenario_ratio[scenario]
        if power < lowest_power - 1e-12:
            result.append(point)
            lowest_power = power
    return sorted(result, key=lambda point: point.performance)


def estimate_run(
    run_dir: Path, models: tuple[DramPowerModel, ...] = DEFAULT_MODELS
) -> tuple[list[EstimatedPoint], dict[str, Any]]:
    run_dir = run_dir.resolve()
    data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    baseline = data["baseline"]
    applications = baseline.get("applications", {})
    if len(applications) != 1:
        raise ValueError("The estimator currently requires exactly one application")
    application, baseline_app = next(iter(applications.items()))

    baseline_log = _only_log(run_dir, "sram-baseline", application)
    if baseline_log is None:
        raise ValueError(f"Could not resolve the SRAM baseline log for {application}")
    baseline_reads, baseline_writes = parse_dram_counts(baseline_log)
    baseline_signature = _signature(baseline_app)

    known_by_signature: dict[tuple[int, ...], tuple[int, int, str]] = {
        baseline_signature: (baseline_reads, baseline_writes, "sram-baseline log")
    }
    raw_by_id: dict[str, dict[str, Any]] = {}
    for raw in data["evaluations"]:
        design = CacheDesign(**raw["design"])
        raw_by_id[design.id] = raw
        app = raw["app_results"][0]
        log = _only_log(run_dir, design.id, application)
        if log is None:
            continue
        try:
            reads, writes = parse_dram_counts(log)
        except ValueError:
            continue
        known_by_signature[_signature(app)] = (reads, writes, f"{design.id} log")

    baseline_power: dict[str, float] = {}
    baseline_breakdown: dict[str, dict[str, float]] = {}
    for model in models:
        dynamic, dram_total = estimate_dram_power_mw(
            baseline_reads,
            baseline_writes,
            float(baseline_app["runtime_s"]),
            model,
        )
        total = float(baseline_app["cache_power_mw"]) + dram_total
        baseline_power[model.name] = total
        baseline_breakdown[model.name] = {
            "dram_dynamic_power_mw": dynamic,
            "dram_background_power_mw": model.background_power_mw,
            "dram_power_mw": dram_total,
            "cache_plus_dram_power_mw": total,
        }

    points: list[EstimatedPoint] = []
    fallback_write_ratio = baseline_writes / max(
        1, int(baseline_app["accesses"]["l2_write"])
    )
    for candidate_id, raw in raw_by_id.items():
        app = raw["app_results"][0]
        signature = _signature(app)
        log = _only_log(run_dir, candidate_id, application)
        counts_source = "candidate log"
        if log is not None:
            try:
                reads, writes = parse_dram_counts(log)
            except ValueError:
                log = None
        if log is None:
            reused = known_by_signature.get(signature)
            if reused is not None:
                reads, writes, source_id = reused
                counts_source = f"identical access signature: {source_id}"
            else:
                reads = int(app["accesses"]["l2_read_miss"])
                writes = round(int(app["accesses"]["l2_write"]) * fallback_write_ratio)
                counts_source = "fallback from L2 counters"

        scenario_power: dict[str, float] = {}
        scenario_ratio: dict[str, float] = {}
        for model in models:
            _, dram_total = estimate_dram_power_mw(
                reads, writes, float(app["runtime_s"]), model
            )
            total = float(app["cache_power_mw"]) + dram_total
            scenario_power[model.name] = total
            scenario_ratio[model.name] = total / baseline_power[model.name]
        points.append(
            EstimatedPoint(
                candidate_id=candidate_id,
                technology=str(raw["design"]["technology"]),
                feasible=bool(raw["feasible"]),
                performance=float(raw["performance_score"]),
                cache_power_mw=float(app["cache_power_mw"]),
                dram_reads=reads,
                dram_writes=writes,
                runtime_s=float(app["runtime_s"]),
                counts_source=counts_source,
                scenario_power_mw=scenario_power,
                scenario_ratio=scenario_ratio,
            )
        )

    provenance = {
        "status": "ESTIMATED_NOT_VENDOR_CALIBRATED",
        "application": application,
        "formula": "cache_power_mw + background_power_mw + (reads*read_energy_nj + writes*write_energy_nj)/runtime_s/1e6",
        "baseline": {
            "dram_reads": baseline_reads,
            "dram_writes": baseline_writes,
            "runtime_s": baseline_app["runtime_s"],
            "cache_power_mw": baseline_app["cache_power_mw"],
            "scenario_power": baseline_breakdown,
        },
        "models": [model.__dict__ for model in models],
        "limitations": [
            "Aggregate read/write requests do not model ACT/PRE/refresh or row-buffer locality.",
            "Background and per-request energy are engineering assumptions, not an Orin vendor calibration.",
            "Memory-controller, NoC, CPU, GPU-core, board-regulator, and fan power are excluded.",
            "A missing duplicate simulation log reuses counts only when all six recorded cache-access counters match exactly.",
        ],
    }
    return points, provenance


def _write_csv(path: Path, points: list[EstimatedPoint]) -> None:
    fields = [
        "candidate_id",
        "technology",
        "feasible",
        "performance_speedup",
        "runtime_s",
        "dram_reads",
        "dram_writes",
        "counts_source",
        "cache_power_mw",
    ]
    for model in DEFAULT_MODELS:
        fields.extend(
            [
                f"cache_plus_dram_power_mw_{model.name}",
                f"cache_plus_dram_power_ratio_{model.name}",
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in points:
            row: dict[str, Any] = {
                "candidate_id": point.candidate_id,
                "technology": point.technology,
                "feasible": point.feasible,
                "performance_speedup": point.performance,
                "runtime_s": point.runtime_s,
                "dram_reads": point.dram_reads,
                "dram_writes": point.dram_writes,
                "counts_source": point.counts_source,
                "cache_power_mw": point.cache_power_mw,
            }
            for model in DEFAULT_MODELS:
                row[f"cache_plus_dram_power_mw_{model.name}"] = point.scenario_power_mw[model.name]
                row[f"cache_plus_dram_power_ratio_{model.name}"] = point.scenario_ratio[model.name]
            writer.writerow(row)


def _write_optima_csv(path: Path, points: list[EstimatedPoint], scenario: str) -> None:
    fields = [
        "technology",
        "selector",
        "candidate_id",
        "performance_speedup",
        "cache_plus_dram_power_ratio",
        "cache_plus_dram_power_mw",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for technology in sorted({point.technology for point in points}):
            candidates = [
                point
                for point in points
                if point.technology == technology and point.feasible
            ]
            if not candidates:
                continue
            selections = {
                "minimum_total_power": min(
                    candidates, key=lambda point: point.scenario_ratio[scenario]
                ),
                "maximum_performance_under_1x": max(
                    (
                        point
                        for point in candidates
                        if point.scenario_ratio[scenario] <= 1.0
                    ),
                    key=lambda point: point.performance,
                    default=None,
                ),
                "maximum_performance_under_1p5x": max(
                    (
                        point
                        for point in candidates
                        if point.scenario_ratio[scenario] <= 1.5
                    ),
                    key=lambda point: point.performance,
                    default=None,
                ),
                "maximum_performance_unconstrained": max(
                    candidates, key=lambda point: point.performance
                ),
            }
            for selector, point in selections.items():
                if point is None:
                    continue
                writer.writerow(
                    {
                        "technology": technology,
                        "selector": selector,
                        "candidate_id": point.candidate_id,
                        "performance_speedup": point.performance,
                        "cache_plus_dram_power_ratio": point.scenario_ratio[scenario],
                        "cache_plus_dram_power_mw": point.scenario_power_mw[scenario],
                    }
                )


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def render_svg(
    points: list[EstimatedPoint], provenance: dict[str, Any], scenario: str = "nominal"
) -> str:
    width, height = 1280, 720
    left, top, plot_width, plot_height = 120.0, 42.0, 1110.0, 588.0
    feasible = [point for point in points if point.feasible]
    technologies = [
        technology
        for technology in ("gain_cell", "sram_tuned")
        if any(point.technology == technology for point in feasible)
    ]
    frontiers = {
        technology: _frontier(
            (point for point in feasible if point.technology == technology), scenario
        )
        for technology in technologies
    }
    all_frontier = [point for values in frontiers.values() for point in values]
    x_low = max(0.0, min(point.performance for point in all_frontier) - 0.1)
    x_high = max(3.65, max(point.performance for point in all_frontier) + 0.18)
    y_low = max(0.0, min(point.scenario_ratio[scenario] for point in all_frontier) - 0.12)
    y_high = max(1.65, max(point.scenario_ratio[scenario] for point in all_frontier) + 0.18)
    sx = lambda value: left + (value - x_low) / (x_high - x_low) * plot_width
    sy = lambda value: top + plot_height - (value - y_low) / (y_high - y_low) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#252525}'
        '.axis{stroke:#52514e;stroke-width:1.4}.grid{stroke:#e1e0d9;stroke-width:1}'
        '.budget{stroke:#898781;stroke-width:1.6;stroke-dasharray:2 4}'
        '.label{font-size:20px}.tick{font-size:17px;fill:#52514e}'
        '.note{font-size:15px;fill:#898781}.annotation{font-size:18px;font-weight:700}'
        '.direction{font-size:17px;font-weight:700;fill:#6f6d68}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<defs><marker id="direction-arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="strokeWidth"><path d="M 0 0 L 10 5 L 0 10 z" fill="#77746e"/></marker></defs>',
    ]
    add = parts.append
    x_ticks = [0.69, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    for value in x_ticks:
        if x_low <= value <= x_high:
            px = sx(value)
            add(f'<line x1="{px}" y1="{top}" x2="{px}" y2="{top + plot_height}" class="grid"/><text x="{px}" y="{top + plot_height + 28}" text-anchor="middle" class="tick">{_fmt(value)}×</text>')
    step = 0.5
    first_y = math.ceil(y_low / step) * step
    y_value = first_y
    while y_value <= y_high + 1e-9:
        py = sy(y_value)
        add(f'<line x1="{left}" y1="{py}" x2="{left + plot_width}" y2="{py}" class="grid"/><text x="{left - 12}" y="{py + 5}" text-anchor="end" class="tick">{_fmt(y_value)}×</text>')
        y_value += step
    add(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis"/><line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis"/>')
    for ratio, label in ((1.0, "SRAM×1 total-power budget"), (1.5, "SRAM×1.5")):
        if y_low <= ratio <= y_high:
            py = sy(ratio)
            add(f'<line x1="{left}" y1="{py}" x2="{left + plot_width}" y2="{py}" class="budget"/><text x="{left + plot_width - 12}" y="{py - 8}" text-anchor="end" class="note">{html.escape(label)}</text>')

    colors = {
        "gain_cell": ("#ef642f", "#f7a889", "#f8cdbc"),
        "sram_tuned": ("#72a9e5", "#9fc5ef", "#c9ddf4"),
    }
    frontier_ids = {
        technology: {point.candidate_id for point in frontier}
        for technology, frontier in frontiers.items()
    }
    for technology in reversed(technologies):
        solid, line, faint = colors[technology]
        for point in feasible:
            if point.technology != technology:
                continue
            ratio = point.scenario_ratio[scenario]
            if not (y_low <= ratio <= y_high):
                continue
            is_frontier = point.candidate_id in frontier_ids[technology]
            radius = 7 if is_frontier else 5.5
            opacity = 0.78 if is_frontier else 0.38
            add(f'<circle cx="{sx(point.performance)}" cy="{sy(ratio)}" r="{radius}" fill="{faint}" fill-opacity="{opacity}"><title>{html.escape(point.candidate_id)} | speedup={point.performance:.4f} | estimated total power={ratio:.4f}×</title></circle>')
        frontier = frontiers[technology]
        path = " ".join(("M" if index == 0 else "L") + f" {sx(point.performance):.2f} {sy(point.scenario_ratio[scenario]):.2f}" for index, point in enumerate(frontier))
        opacity = 1.0 if technology == "gain_cell" else 0.62
        width_px = 3.0 if technology == "gain_cell" else 2.4
        add(f'<path d="{path}" fill="none" stroke="{line}" stroke-opacity="{opacity}" stroke-width="{width_px}"/>')
        for point in frontier:
            if technology == "gain_cell":
                add(f'<circle cx="{sx(point.performance)}" cy="{sy(point.scenario_ratio[scenario])}" r="8" fill="{solid}"/>')
            else:
                add(f'<circle cx="{sx(point.performance)}" cy="{sy(point.scenario_ratio[scenario])}" r="7" fill="white" fill-opacity="0.82" stroke="{solid}" stroke-opacity="0.72" stroke-width="2.4"/>')

    gain_frontier = frontiers.get("gain_cell", [])
    if gain_frontier:
        selected = [
            ("A", min(gain_frontier, key=lambda point: abs(point.performance - 1.0)), 1.28, 1.12),
            ("B", min(gain_frontier, key=lambda point: point.scenario_ratio[scenario]), 1.02, max(y_low + 0.05, 0.72)),
            ("C", max(gain_frontier, key=lambda point: point.performance), 2.20, min(y_high - 0.18, 2.68)),
        ]
        descriptions = {
            "A": lambda p: f"A — same speed, {p.scenario_ratio[scenario]:.2f}× total power",
            "B": lambda p: f"B — minimum total power: {p.scenario_ratio[scenario]:.2f}×",
            "C": lambda p: f"C — {p.performance:.2f}× speed, {p.scenario_ratio[scenario]:.2f}× total power",
        }
        for label, point, tx, ty in selected:
            px, py = sx(point.performance), sy(point.scenario_ratio[scenario])
            text_x, text_y = sx(tx), sy(ty)
            add(f'<circle cx="{px}" cy="{py}" r="16" fill="none" stroke="#111111" stroke-width="2.6"/><line x1="{px + (10 if text_x >= px else -10)}" y1="{py}" x2="{text_x}" y2="{text_y - 6}" stroke="#898781" stroke-width="1.4"/><text x="{text_x}" y="{text_y}" class="annotation">{html.escape(descriptions[label](point))}</text>')

    baseline_x, baseline_y = sx(1.0), sy(1.0)
    add(f'<rect x="{baseline_x - 9}" y="{baseline_y - 9}" width="18" height="18" fill="#2369b3" stroke="#111111" stroke-width="1.4"/><text x="{baseline_x + 20}" y="{baseline_y - 12}" style="font-size:20px;font-weight:700;fill:#2369b3">SRAM baseline</text>')

    model = next(model for model in DEFAULT_MODELS if model.name == scenario)
    add(f'<text x="{left + 14}" y="{top + 22}" class="note">Estimated LPDDR5 ({scenario}): {model.read_energy_nj:g} nJ/read, {model.write_energy_nj:g} nJ/write, {model.background_power_mw:g} mW background, {model.transaction_bytes} B/request</text>')
    direction_x1, direction_y1 = sx(2.15), sy(min(y_high - 0.38, 1.35))
    direction_x2, direction_y2 = sx(2.65), sy(max(y_low + 0.22, 1.08))
    add(f'<text x="{direction_x1 - 10}" y="{direction_y1 - 12}" text-anchor="end" class="direction">worse</text><line x1="{direction_x1}" y1="{direction_y1}" x2="{direction_x2}" y2="{direction_y2}" stroke="#77746e" stroke-width="2.2" marker-end="url(#direction-arrow)"/><text x="{direction_x2 + 13}" y="{direction_y2 + 6}" class="direction">better</text>')
    add(f'<text x="{left + plot_width / 2}" y="{height - 28}" text-anchor="middle" class="label">Application performance (speedup vs SRAM)</text><text x="30" y="{top + plot_height / 2}" text-anchor="middle" class="label" transform="rotate(-90 30 {top + plot_height / 2})">Estimated cache + LPDDR5 power vs SRAM</text>')
    legend_y = top + plot_height - 70
    add(f'<circle cx="{left + plot_width - 275}" cy="{legend_y}" r="8" fill="#ef642f"/><text x="{left + plot_width - 257}" y="{legend_y + 7}" class="label">Gain-cell frontier</text><circle cx="{left + plot_width - 275}" cy="{legend_y + 36}" r="7" fill="white" stroke="#72a9e5" stroke-width="2.4"/><text x="{left + plot_width - 257}" y="{legend_y + 43}" class="label">SRAM-tuned frontier</text>')
    add('</svg>')
    return "".join(parts)


def write_estimate(run_dir: Path) -> dict[str, Path]:
    points, provenance = estimate_run(run_dir)
    outputs = {
        "csv": run_dir / "estimated_memory_power.csv",
        "optima_csv": run_dir / "estimated_memory_power_optima.csv",
        "provenance": run_dir / "estimated_memory_power_model.json",
        "svg": run_dir / "power_performance_pareto_cache_plus_dram.svg",
    }
    _write_csv(outputs["csv"], points)
    _write_optima_csv(outputs["optima_csv"], points, "nominal")
    outputs["provenance"].write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    outputs["svg"].write_text(
        render_svg(points, provenance, "nominal"), encoding="utf-8"
    )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Estimate cache + LPDDR5 power from an existing optimizer run"
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    for name, path in write_estimate(args.run_dir).items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

