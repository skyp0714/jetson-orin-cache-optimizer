import math

import pytest

from cache_optimizer.models import CacheDesign, weighted_geomean


def test_design_id_is_stable_and_descriptive():
    design = CacheDesign("gain_cell", "min_energy_runtime_bound", "sram", "gain_cell", 256, 4, 4096, 16)
    assert design.id == CacheDesign(**design.__dict__).id
    assert design.id.startswith("gain_cell-min_energy_runtime_bound-")


def test_weighted_geomean():
    assert weighted_geomean([1.0, 4.0]) == pytest.approx(2.0)
    assert weighted_geomean([1.0, 4.0], [3.0, 1.0]) == pytest.approx(math.sqrt(2.0))


def test_weighted_geomean_rejects_nonpositive():
    with pytest.raises(ValueError):
        weighted_geomean([1.0, 0.0])

