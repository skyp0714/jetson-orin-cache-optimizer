"""Calibrate array-only cache energy to a cache-subsystem power envelope.

NS-Cache reports data/tag array energy, leakage, and refresh.  A hardware-
calibrated GPU power model also attributes energy to the L2 controller and
interconnect.  This post-processor preserves every candidate's NS-Cache energy
and adds a technology-independent per-L2-access overhead inferred from an
external effective-energy anchor.  It never reruns Accel-Sim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .models import CacheDesign


@dataclass(frozen=True)
class L2Anchor:
    name: str
    effective_energy_pj: float

    def validate(self) -> None:
        if not math.isfinite(self.effective_energy_pj) or self.effective_energy_pj <= 0:
            raise ValueError("effective_energy_pj must be finite and positive")


# Interquartile range and median calculated from the AccelWattch Volta_HW
# validation artifact after normalizing L2CP by L2 accesses and kernel time.
# Only kernels with at least 200,000 L2 accesses are included (n=22).
DEFAULT_ANCHORS = (
    L2Anchor("low", 635.6333986333665),
    L2Anchor("nominal", 736.0643952512144),
    L2Anchor("high", 981.9438775256672),
)

ANCHOR_SOURCE = (
    "AccelWattch MICRO'21 Volta_HW validation artifact; L2CP includes the "
    "artifact's L2+NoC grouping and is transferred to Orin as a sensitivity anchor"
)
ANCHOR_URL = "https://github.com/accel-sim/accel-sim-framework"


def l2_accesses(app: dict[str, Any]) -> int:
    accesses = app["accesses"]
    return sum(
        int(accesses[name])
        for name in ("l2_read_hit", "l2_read_miss", "l2_write")
    )


def baseline_l2_array_dynamic_pj(
    baseline: dict[str, Any], app: dict[str, Any]
) -> float:
    accesses = app["accesses"]
    total = l2_accesses(app)
    if total <= 0:
        raise ValueError("baseline must contain at least one L2 access")
    ppa = baseline["l2_ppa"]
    dynamic_nj = (
        int(accesses["l2_read_hit"]) * float(ppa["hit_energy_nj"])
        + int(accesses["l2_read_miss"]) * float(ppa["miss_energy_nj"])
        + int(accesses["l2_write"]) * float(ppa["write_energy_nj"])
    )
    return dynamic_nj / total * 1_000.0


def calibrate_app(
    app: dict[str, Any], *, array_dynamic_pj: float, anchor: L2Anchor
) -> tuple[float, float, float]:
    """Return overhead pJ/access, calibrated energy nJ, and power mW."""
    anchor.validate()
    runtime_s = float(app["runtime_s"])
    if not math.isfinite(runtime_s) or runtime_s <= 0:
        raise ValueError("runtime_s must be finite and positive")
    overhead_pj = max(0.0, anchor.effective_energy_pj - array_dynamic_pj)
    energy_nj = float(app["cache_energy_nj"]) + l2_accesses(app) * overhead_pj / 1_000.0
    power_mw = energy_nj / runtime_s / 1e6
    return overhead_pj, energy_nj, power_mw


def _candidate_id(design: dict[str, Any]) -> str:
    return CacheDesign(**design).id


def _points(data: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any], dict[str, Any]]]:
    baseline = data["baseline"]
    for application, app in baseline["applications"].items():
        yield "sram-baseline", {"technology": "sram", "objective": "baseline"}, app
    for evaluation in data["evaluations"]:
        design = evaluation["design"]
        candidate_id = _candidate_id(design)
        for app in evaluation["app_results"]:
            yield candidate_id, design, app


def calibrate_run(
    run_dir: Path, anchors: tuple[L2Anchor, ...] = DEFAULT_ANCHORS
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = run_dir.resolve()
    data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    baseline = data["baseline"]
    baseline_apps = baseline["applications"]
    if not baseline_apps:
        raise ValueError("baseline contains no applications")
    for anchor in anchors:
        anchor.validate()

    raw_by_app = {
        name: baseline_l2_array_dynamic_pj(baseline, app)
        for name, app in baseline_apps.items()
    }
    rows: list[dict[str, Any]] = []
    for candidate_id, design, app in _points(data):
        application = str(app["application"])
        raw_pj = raw_by_app[application]
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "technology": design["technology"],
            "objective": design["objective"],
            "application": application,
            "runtime_s": float(app["runtime_s"]),
            "l2_accesses": l2_accesses(app),
            "raw_cache_energy_mj": float(app["cache_energy_nj"]) / 1e6,
            "raw_cache_power_w": float(app["cache_power_mw"]) / 1e3,
            "baseline_l2_array_dynamic_pj_per_access": raw_pj,
        }
        for anchor in anchors:
            overhead_pj, energy_nj, power_mw = calibrate_app(
                app, array_dynamic_pj=raw_pj, anchor=anchor
            )
            row[f"{anchor.name}_overhead_pj_per_l2_access"] = overhead_pj
            row[f"{anchor.name}_cache_subsystem_energy_mj"] = energy_nj / 1e6
            row[f"{anchor.name}_cache_subsystem_power_w"] = power_mw / 1e3
        rows.append(row)

    model = {
        "scope": "L1/L2 NS-Cache array energy plus transferred L2 controller/NoC envelope",
        "formula": (
            "calibrated_cache_energy = raw_cache_energy + L2_accesses * "
            "max(0, anchor_effective_energy - baseline_L2_array_dynamic_energy)"
        ),
        "anchor_source": ANCHOR_SOURCE,
        "anchor_url": ANCHOR_URL,
        "anchor_sample_filter": "AccelWattch Volta_HW kernels with >=200000 L2 accesses",
        "anchor_sample_count": 22,
        "anchors_pj_per_l2_access": {
            anchor.name: anchor.effective_energy_pj for anchor in anchors
        },
        "baseline_l2_array_dynamic_pj_per_access": raw_by_app,
        "limitations": [
            "This is not an Orin rail measurement or an Orin-specific AccelWattch fit.",
            "The anchor transfers a GV100 L2+NoC effective energy distribution to Orin.",
            "The added overhead is technology-independent and scales with simulated L2 accesses.",
            "NS-Cache array leakage and refresh remain unchanged.",
        ],
    }
    return rows, model


def write_outputs(run_dir: Path, rows: list[dict[str, Any]], model: dict[str, Any]) -> None:
    if not rows:
        raise ValueError("no calibrated rows to write")
    csv_path = run_dir / "estimated_cache_subsystem_power.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "estimated_cache_subsystem_power_model.json").write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add an access-normalized L2 controller/NoC anchor to array-only cache power"
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    rows, model = calibrate_run(args.run_dir)
    write_outputs(args.run_dir, rows, model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
