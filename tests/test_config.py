import json
from pathlib import Path

import pytest

from cache_optimizer.cli import main
from cache_optimizer.config import ConfigError, load_config


def _raw_config(tmp_path: Path) -> dict:
    kernelslist = tmp_path / "kernelslist.g"
    kernelslist.write_text("kernel-1.traceg\n", encoding="utf-8")
    return {
        "paths": {
            "nscache_root": "missing-nscache",
            "nsc_binary": "missing-nsc",
            "accelsim_root": "missing-accelsim",
            "accelsim_binary": "missing-accel-sim.out",
            "accelsim_config": "missing-gpgpusim.config",
            "trace_config": "missing-trace.config",
            "output_dir": "results",
        },
        "baseline": {
            "l1_cfg": "missing-l1.cfg",
            "l2_cfg": "missing-l2.cfg",
            "l1_instances": 16,
            "l2_instances": 1,
        },
        "technologies": {
            "gain_cell": {
                "l1_template": "missing-gain-l1.cfg",
                "l2_template": "missing-gain-l2.cfg",
            }
        },
        "applications": [
            {"name": "app", "trace": str(kernelslist), "weight": 1.0}
        ],
        "search_space": {
            "l1_capacity_kib": [256],
            "l1_associativity": [4],
            "l2_capacity_kib": [4096],
            "l2_associativity": [16],
        },
    }


def _write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_unsafe_technology_name_is_rejected(tmp_path):
    raw = _raw_config(tmp_path)
    raw["technologies"]["../escape"] = raw["technologies"].pop("gain_cell")
    with pytest.raises(ConfigError, match="Technology names"):
        load_config(_write(tmp_path, raw))


def test_boolean_integer_and_nonfinite_number_are_rejected(tmp_path):
    raw = _raw_config(tmp_path)
    raw["baseline"]["l1_instances"] = True
    with pytest.raises(ConfigError, match="positive integer"):
        load_config(_write(tmp_path, raw))

    raw = _raw_config(tmp_path)
    raw["optimizer"] = {"saturation_area_ratio_max": float("nan")}
    with pytest.raises(ConfigError, match="finite number"):
        load_config(_write(tmp_path, raw))


def test_cli_reports_missing_paths_without_traceback(tmp_path, capsys):
    config = _write(tmp_path, _raw_config(tmp_path))
    assert main(["--config", str(config), "--validate-only"]) == 2
    stderr = capsys.readouterr().err
    assert "does not exist" in stderr
    assert "Traceback" not in stderr


def test_relative_and_absolute_power_constraints_are_normalized(tmp_path):
    raw = _raw_config(tmp_path)
    raw["objectives"] = ["pareto_full_cache"]
    raw["power_constraints"] = [
        {"name": "sram_1x", "max_power_ratio": 1.0},
        {"name": "cache_300mw", "max_power_mw": 300},
    ]
    config = load_config(_write(tmp_path, raw))
    assert config["power_constraints"] == [
        {"name": "sram_1x", "type": "relative", "limit": 1.0},
        {"name": "cache_300mw", "type": "absolute_mw", "limit": 300.0},
    ]


@pytest.mark.parametrize(
    "budgets, message",
    [
        (
            [{"name": "both", "max_power_ratio": 1.0, "max_power_mw": 300}],
            "exactly one",
        ),
        ([{"name": "none"}], "exactly one"),
        ([{"name": "../escape", "max_power_ratio": 1.0}], "name must use"),
        (
            [
                {"name": "same", "max_power_ratio": 1.0},
                {"name": "same", "max_power_ratio": 1.5},
            ],
            "Duplicate power constraint",
        ),
        ([{"name": "zero", "max_power_mw": 0}], "must be > 0"),
    ],
)
def test_invalid_power_constraints_are_rejected(tmp_path, budgets, message):
    raw = _raw_config(tmp_path)
    raw["power_constraints"] = budgets
    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path, raw))
