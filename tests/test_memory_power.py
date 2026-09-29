from pathlib import Path

import pytest

from cache_optimizer.memory_power import (
    DramPowerModel,
    estimate_dram_power_mw,
    parse_dram_counts,
)


def test_dram_power_formula_separates_dynamic_and_background():
    model = DramPowerModel("test", 4.0, 2.0, 500.0)

    dynamic, total = estimate_dram_power_mw(1000, 500, 0.001, model)

    assert dynamic == pytest.approx(5.0)
    assert total == pytest.approx(505.0)


def test_parse_dram_counts_uses_completed_totals(tmp_path: Path):
    log = tmp_path / "accelsim.stdout.log"
    log.write_text(
        "total dram reads = 10\ntotal dram writes = 4\n"
        "total dram reads = 12\ntotal dram writes = 5\n",
        encoding="utf-8",
    )

    assert parse_dram_counts(log) == (12, 5)


def test_invalid_runtime_is_rejected():
    with pytest.raises(ValueError, match="runtime_s"):
        estimate_dram_power_mw(1, 1, 0.0, DramPowerModel("test", 1, 1, 1))

