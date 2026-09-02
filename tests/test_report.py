import csv
import re
from pathlib import Path

from cache_optimizer.models import CacheDesign, Evaluation
from cache_optimizer import report
from cache_optimizer.report import detect_saturation, write_reports
from cache_optimizer.search import MAX_PERFORMANCE, MIN_ENERGY, PARETO_FULL_CACHE


def evaluation(
    index: int,
    area: float,
    perf: float,
    energy: float,
    *,
    technology: str = "gain_cell",
    objective: str = MAX_PERFORMANCE,
    power: float = None,
) -> Evaluation:
    design = CacheDesign(
        technology,
        objective,
        "sram" if objective == MIN_ENERGY else technology,
        technology,
        128 << index,
        4,
        2048 << index,
        16,
    )
    power_score = energy / perf if power is None else power
    return Evaluation(design, index, area, area, 0, 0, [], perf, energy, power_score, 1 / perf, energy, area <= 1 and energy <= 1, [])


def test_reports_are_created_without_plotting_dependency(tmp_path: Path):
    values = [evaluation(0, 0.5, 0.9, 0.7), evaluation(1, 1.0, 1.05, 0.95)]
    outputs = write_reports(tmp_path, values, baseline={"area_mm2": 1.2}, metadata={"status": "test"})
    for output in outputs.values():
        assert Path(output).is_file()
    assert "<svg" in (tmp_path / "area_saturation.svg").read_text()
    assert "dashed lines = SRAM bounds" in (tmp_path / "area_saturation.svg").read_text()


def test_saturation_detection_needs_stable_low_slopes():
    values = [
        evaluation(0, 0.5, 0.8, 0.7),
        evaluation(1, 0.8, 1.0, 0.8),
        evaluation(2, 1.0, 1.01, 0.9),
        evaluation(3, 1.2, 1.015, 1.0),
    ]
    assert detect_saturation(values) is values[2]


def test_saturation_detection_keeps_flat_running_best_evidence():
    values = [
        evaluation(0, 0.5, 0.8, 0.7),
        evaluation(1, 0.7, 2.0, 0.8),
        evaluation(2, 1.0, 1.7, 0.9),
        evaluation(3, 1.3, 1.6, 1.0),
    ]
    assert detect_saturation(values) is values[1]


def test_history_bounds_use_only_the_panel_objective(monkeypatch):
    minimum = evaluation(
        0, 0.8, 50.0, 0.8, objective=MIN_ENERGY
    )
    maximum = evaluation(
        1, 0.9, 1.1, 75.0, objective=MAX_PERFORMANCE
    )
    calls = []

    def capture_panel(plot, rect, xs, ys, *args, **kwargs):
        calls.append((list(xs), list(ys)))
        return (lambda value: value, lambda value: value)

    monkeypatch.setattr(report, "_panel", capture_panel)
    report._history_svg([minimum, maximum])

    assert calls[0] == ([0.0], [minimum.energy_score])
    assert calls[1] == ([1.0], [maximum.performance_score])


def test_pareto_is_grouped_by_technology_and_objective(tmp_path: Path):
    # The gain-cell point is globally dominated, but belongs to an independent
    # technology/objective search and must remain on that group's frontier.
    gain = evaluation(0, 0.9, 1.0, 0.9, technology="gain_cell")
    stt = evaluation(0, 0.8, 1.1, 0.8, technology="stt_mram")
    gain_dominated = evaluation(1, 1.0, 0.9, 1.0, technology="gain_cell")

    write_reports(
        tmp_path,
        [gain, stt, gain_dominated],
        baseline={"area_mm2": 1.2},
        metadata={"status": "test"},
    )

    with (tmp_path / "pareto_front.csv").open(newline="", encoding="utf-8") as handle:
        ids = {row["candidate_id"] for row in csv.DictReader(handle)}
    assert gain.design.id in ids
    assert stt.design.id in ids
    assert gain_dominated.design.id not in ids

    tradeoff = (tmp_path / "energy_performance_tradeoff.svg").read_text()
    assert re.search(
        rf'<circle[^>]+stroke="#111827"[^>]*><title>{re.escape(gain.design.id)}',
        tradeoff,
    )


def test_saturation_csv_summary_and_metadata_are_reported(tmp_path: Path):
    values = [
        evaluation(0, 0.5, 0.8, 0.7, power=0.4),
        evaluation(1, 0.8, 1.0, 0.8, power=0.7),
        evaluation(2, 1.0, 1.01, 0.9, power=0.85),
        evaluation(3, 1.2, 1.015, 1.0, power=0.95),
    ]
    metadata = {
        "status": "partial",
        "constraints": {"max_area_ratio": 1.0, "energy_tolerance": 0.0},
        "assumptions": {"performance_model": "hit-latency-only"},
        "model_notes": {"gain_cell": "measured retention pending"},
        "warnings": ["trace scope excludes CPU work"],
        "errors": [{"candidate": "bad-one", "error": "timed out"}],
    }
    outputs = write_reports(
        tmp_path,
        values,
        baseline={"area_mm2": 1.2},
        metadata=metadata,
    )

    assert Path(outputs["saturation_csv"]).name == "saturation_points.csv"
    with (tmp_path / "saturation_points.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["status"] == "DETECTED"
    assert float(rows[0]["slope_threshold"]) == 0.10
    assert rows[0]["candidate_id"] == values[2].design.id
    assert rows[0]["l1_capacity_kib"] == str(values[2].design.l1_capacity_kib)
    assert float(rows[0]["area_ratio"]) == values[2].area_ratio
    assert float(rows[0]["performance_score"]) == values[2].performance_score

    summary = (tmp_path / "summary.md").read_text()
    html_report = (tmp_path / "report.html").read_text()
    for expected in (
        "Area saturation points",
        "measured retention pending",
        "trace scope excludes CPU work",
        "timed out",
        "max_area_ratio",
        "hit-latency-only",
        "Run status: partial",
    ):
        assert expected in summary
        assert expected in html_report


def test_saturation_power_line_uses_performance_envelope(monkeypatch):
    values = [
        evaluation(0, 0.5, 0.8, 0.7, power=0.3),
        evaluation(1, 0.7, 0.75, 0.8, power=9.0),  # not on performance envelope
        evaluation(2, 1.0, 1.0, 0.9, power=0.8),
    ]
    captured_paths = []
    original_add = report._Plot.add

    def capture_add(self, value):
        if value.startswith('<path d="'):
            captured_paths.append(value)
        original_add(self, value)

    monkeypatch.setattr(report._Plot, "add", capture_add)
    report._saturation_svg(values)

    # One connected trace per panel, each with the two designs selected by the
    # performance envelope; the high-power non-envelope design is only a dot.
    assert len(captured_paths) == 2
    assert all(path.count(" L ") == 1 for path in captured_paths)


def test_saturation_csv_marks_technology_without_performance_points(tmp_path: Path):
    only_min_energy = evaluation(
        0,
        0.8,
        1.0,
        0.7,
        technology="stt_mram",
        objective=MIN_ENERGY,
    )
    write_reports(
        tmp_path,
        [only_min_energy],
        baseline={"area_mm2": 1.2},
        metadata={"status": "partial"},
    )
    with (tmp_path / "saturation_points.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "technology": "stt_mram",
            "status": "NOT_DETECTED",
            "slope_threshold": "0.1",
            "candidate_id": "",
            "l1_technology": "",
            "l2_technology": "",
            "l1_capacity_kib": "",
            "l1_associativity": "",
            "l2_capacity_kib": "",
            "l2_associativity": "",
            "area_mm2": "",
            "area_ratio": "",
            "performance_score": "",
            "energy_score": "",
            "power_score": "",
            "feasible": "",
        }
    ]


def test_power_budget_outputs_keep_each_budget_and_selector_separate(tmp_path: Path):
    low = evaluation(
        0, 0.8, 1.05, 0.9, objective=PARETO_FULL_CACHE, power=0.9
    )
    mid = evaluation(
        1, 0.9, 1.20, 0.9, objective=PARETO_FULL_CACHE, power=1.4
    )
    fast = evaluation(
        2, 0.95, 1.30, 0.9, objective=PARETO_FULL_CACHE, power=1.6
    )
    outputs = write_reports(
        tmp_path,
        [low, mid, fast],
        baseline={"area_mm2": 1.2},
        metadata={"status": "test"},
        power_constraints=[
            {"name": "sram_1x", "type": "relative", "limit": 1.0},
            {"name": "sram_1p5x", "type": "relative", "limit": 1.5},
        ],
        constraints={
            "max_area_ratio": 1.0,
            "runtime_tolerance": 0.02,
            "energy_tolerance": 0.0,
            "constraint_scope": "per_application",
        },
    )
    assert Path(outputs["power_optima_csv"]).name == "power_constrained_optima.csv"
    assert Path(outputs["power_svg"]).name == "power_constrained_optima.svg"
    with (tmp_path / "power_constrained_optima.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    max_performance = {
        row["budget_name"]: row
        for row in rows
        if row["selector"] == "max_performance_iso_energy"
    }
    assert max_performance["sram_1x"]["candidate_id"] == low.design.id
    assert max_performance["sram_1p5x"]["candidate_id"] == mid.design.id

    svg = (tmp_path / "power_constrained_optima.svg").read_text()
    summary = (tmp_path / "summary.md").read_text()
    html_report = (tmp_path / "report.html").read_text()
    for expected in ("sram_1x", "sram_1p5x", "max performance"):
        assert expected in svg or expected in summary
        assert expected in summary
        assert expected in html_report
