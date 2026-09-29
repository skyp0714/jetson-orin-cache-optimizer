import pytest

from cache_optimizer.cache_subsystem_power import (
    L2Anchor,
    baseline_l2_array_dynamic_pj,
    calibrate_app,
)


def _baseline():
    return {
        "l2_ppa": {
            "hit_energy_nj": 0.2,
            "miss_energy_nj": 0.2,
            "write_energy_nj": 0.1,
        }
    }


def _app():
    return {
        "runtime_s": 0.001,
        "cache_energy_nj": 1_000.0,
        "accesses": {
            "l2_read_hit": 2,
            "l2_read_miss": 1,
            "l2_write": 1,
        },
    }


def test_baseline_dynamic_energy_is_access_weighted():
    assert baseline_l2_array_dynamic_pj(_baseline(), _app()) == pytest.approx(175.0)


def test_calibration_adds_only_missing_per_access_envelope():
    overhead, energy, power = calibrate_app(
        _app(), array_dynamic_pj=175.0, anchor=L2Anchor("nominal", 675.0)
    )

    assert overhead == pytest.approx(500.0)
    assert energy == pytest.approx(1_002.0)
    assert power == pytest.approx(1.002)


def test_anchor_below_array_energy_does_not_subtract_energy():
    overhead, energy, _ = calibrate_app(
        _app(), array_dynamic_pj=175.0, anchor=L2Anchor("low", 100.0)
    )

    assert overhead == 0.0
    assert energy == 1_000.0
