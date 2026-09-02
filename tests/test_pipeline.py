from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cache_optimizer import accelsim, nscache
from cache_optimizer.models import AccessCounts
from cache_optimizer.pipeline import CacheOptimizationPipeline, PipelineError
from cache_optimizer.search import PARETO_FULL_CACHE
from cache_optimizer.selection import (
    BALANCED_KNEE,
    MAX_PERFORMANCE_ISO_ENERGY,
    MIN_ENERGY_ISO_PERFORMANCE,
)


def _cache_config(capacity_kib: int, associativity: int, cell: Path) -> str:
    return (
        "-DesignTarget: cache\n"
        f"-Associativity (for cache only): {associativity}\n"
        f"-Capacity (KB): {capacity_kib}\n"
        "-WordWidth (bit): 256\n"
        f'-MemoryCellInputFile: "{cell}"\n'
        "-Temperature (K): 323\n"
    )


def _fixture_config(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    nsc_root = tmp_path / "nscache"
    accel_root = tmp_path / "accelsim"
    nsc_root.mkdir()
    accel_root.mkdir()
    runtime_lib = accel_root / "lib"
    runtime_lib.mkdir()
    (runtime_lib / "libcudart.so").write_text(
        "fixture gpgpu-sim runtime\n", encoding="utf-8"
    )

    nsc_binary = nsc_root / "nsc"
    accel_binary = accel_root / "accel-sim.out"
    for binary in (nsc_binary, accel_binary):
        binary.write_text("fixture binary\n", encoding="utf-8")
        binary.chmod(0o755)

    sram_cell = nsc_root / "sram.cell"
    gain_cell = nsc_root / "gain.cell"
    sram_cell.write_text("-MemCellType: SRAM\n", encoding="utf-8")
    gain_cell.write_text("-MemCellType: gcDRAM\n", encoding="utf-8")
    baseline_l1 = nsc_root / "baseline_l1.cfg"
    baseline_l2 = nsc_root / "baseline_l2.cfg"
    gain_l1 = nsc_root / "gain_l1.cfg"
    gain_l2 = nsc_root / "gain_l2.cfg"
    baseline_l1.write_text(_cache_config(256, 4, sram_cell), encoding="utf-8")
    baseline_l2.write_text(_cache_config(4096, 16, sram_cell), encoding="utf-8")
    gain_l1.write_text(_cache_config(256, 4, gain_cell), encoding="utf-8")
    gain_l2.write_text(_cache_config(4096, 16, gain_cell), encoding="utf-8")

    gpu_config = accel_root / "gpgpusim.config"
    gpu_config.write_text(
        "-gpgpu_n_clusters 16\n"
        "-gpgpu_n_cores_per_cluster 1\n"
        "-gpgpu_n_mem 16\n"
        "-gpgpu_n_sub_partition_per_mchannel 1\n"
        "-gpgpu_clock_domains 1300:1300:1300:1600\n"
        "-gpgpu_adaptive_cache_config 1\n"
        "-gpgpu_shmem_option 0,8,16,32,64,100,132,164\n"
        "-gpgpu_unified_l1d_size 192\n"
        "-gpgpu_shmem_size 167936\n"
        "-gpgpu_shmem_sizeDefault 167936\n"
        "-gpgpu_cache:dl1 S:4:128:64,L:T:m:L:L,A:384:48,16:0,32\n"
        "-gpgpu_l1_latency 38\n"
        "-gpgpu_cache:dl2 S:128:128:16,L:B:m:L:P,A:192:32,32:0,32\n"
        "-gpgpu_l2_rop_latency 146\n",
        encoding="utf-8",
    )
    trace_config = accel_root / "trace.config"
    trace_config.write_text("# trace config\n", encoding="utf-8")
    traces = tmp_path / "application" / "traces"
    traces.mkdir(parents=True)
    kernelslist = traces / "kernelslist.g"
    kernelslist.write_text("kernel-1.traceg\n", encoding="utf-8")
    (traces / "kernel-1.traceg").write_text("trace fixture\n", encoding="utf-8")

    output = tmp_path / "results"
    config = {
        "config_path": str(tmp_path / "pipeline.json"),
        "paths": {
            "nscache_root": str(nsc_root),
            "nsc_binary": str(nsc_binary),
            "accelsim_root": str(accel_root),
            "accelsim_binary": str(accel_binary),
            "accelsim_config": str(gpu_config),
            "trace_config": str(trace_config),
            "output_dir": str(output),
        },
        "baseline": {
            "l1_cfg": str(baseline_l1),
            "l2_cfg": str(baseline_l2),
            "l1_instances": 16,
            "l2_instances": 1,
            "l1_banks_per_instance": 1,
            "l2_banks_per_instance": 1,
            "l1_counts_scope": "all_instances",
            "l2_counts_scope": "all_instances",
        },
        "technologies": {
            "gain_cell": {
                "l1_template": str(gain_l1),
                "l2_template": str(gain_l2),
                "cell_template": str(gain_cell),
                "retention_time_us": 315000.0,
                "retention_reference_temperature_k": 300.0,
                "model_note": "test gain-cell model",
            }
        },
        "applications": [
            {"name": "app name", "trace": str(kernelslist), "weight": 1.0}
        ],
        "search_space": {
            "l1_capacity_kib": [256],
            "l1_associativity": [4],
            "l2_capacity_kib": [4096],
            "l2_associativity": [16],
        },
        "objectives": ["min_energy_runtime_bound"],
        "constraints": {
            "max_area_ratio": 1.0,
            "runtime_tolerance": 0.02,
            "energy_tolerance": 0.0,
            "constraint_scope": "per_application",
        },
        "optimizer": {
            "max_evaluations_per_objective": 1,
            "max_iterations": 1,
            "beam_width": 1,
            "max_workers": 1,
            "nsc_timeout_s": 10,
            "sim_timeout_s": 10,
            "resume": True,
            "saturation_area_ratio_max": 1.5,
        },
        "latency_mapping": {
            "mode": "relative_delta",
            "rounding": "ceil_away_from_zero",
            "minimum_cycles": 1,
        },
    }
    return config, {
        "output": output,
        "baseline_l1": baseline_l1,
        "baseline_l2": baseline_l2,
    }


def _install_fake_adapters(monkeypatch, paths):
    baseline_l1 = nscache.CachePPA(0.1, 1, 1, 1, 1, 1, 1, 0.1)
    baseline_l2 = nscache.CachePPA(0.4, 1, 1, 1, 1, 1, 1, 0.1)
    candidate = nscache.CachePPA(
        0.3, 1.2, 1, 1.5, 0.5, 0.5, 0.5, 0.05,
        refresh_energy_nj=1.0,
    )

    def fake_nsc(cache, binary, config_path, *, cwd, timeout_s):
        path = Path(config_path)
        if path == paths["baseline_l1"]:
            return baseline_l1, False
        if path == paths["baseline_l2"]:
            return baseline_l2, False
        return candidate, False

    stats = SimpleNamespace(
        cycles=100,
        instructions=1000,
        ipc=10.0,
        accesses=AccessCounts(10, 10, 10, 10, 10, 10),
    )

    monkeypatch.setattr(nscache, "cached_run_nscache", fake_nsc)
    monkeypatch.setattr(
        accelsim,
        "build_accelsim_environment",
        lambda root: {"LD_LIBRARY_PATH": str(Path(root) / "lib")},
    )
    monkeypatch.setattr(
        CacheOptimizationPipeline,
        "_run_sim_cached",
        lambda self, **kwargs: stats,
    )


def test_mocked_pipeline_writes_reports_and_self_contained_optimum(tmp_path, monkeypatch):
    config, paths = _fixture_config(tmp_path)
    _install_fake_adapters(monkeypatch, paths)
    pipeline = CacheOptimizationPipeline(config)

    assert pipeline.validate() == []
    first = pipeline.run()
    assert first["evaluations"] == 1
    assert first["metadata"]["status"] == "complete"
    assert len(pipeline.warnings) == 2

    optimum = (
        paths["output"]
        / "optimal_configs"
        / "gain_cell"
        / "min_energy_runtime_bound"
    )
    assert (optimum / "selection.json").is_file()
    assert (optimum / "cells" / "l1.cell").is_file()
    assert (optimum / "cells" / "l2.cell").is_file()
    assert 'MemoryCellInputFile: "cells/l2.cell"' in (
        optimum / "nsc_l2.cfg"
    ).read_text(encoding="utf-8")
    assert (paths["output"] / "saturation_points.csv").is_file()

    # A rerun may reuse simulator/PPA work, but current-run logs and selected
    # artifacts must not accumulate stale duplicates.
    pipeline.run()
    partial = (paths["output"] / "partial_evaluations.jsonl").read_text(
        encoding="utf-8"
    )
    assert len([line for line in partial.splitlines() if line]) == 1
    assert len(pipeline.warnings) == 2


def test_zero_completed_group_is_reported_as_incomplete(tmp_path, monkeypatch):
    config, paths = _fixture_config(tmp_path)
    _install_fake_adapters(monkeypatch, paths)
    pipeline = CacheOptimizationPipeline(config)

    def reject(*args, **kwargs):
        raise RuntimeError("candidate rejected")

    monkeypatch.setattr(pipeline, "_prepare_candidate", reject)
    with pytest.raises(PipelineError, match="No candidate completed"):
        pipeline.run()
    manifest = json.loads((paths["output"] / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert (paths["output"] / "report.html").is_file()
    assert "Run status: incomplete" in (paths["output"] / "report.html").read_text()


def test_unified_pareto_packages_each_power_budget_selection(
    tmp_path, monkeypatch
):
    config, paths = _fixture_config(tmp_path)
    config["objectives"] = [PARETO_FULL_CACHE]
    config["power_constraints"] = [
        {"name": "wide_budget", "type": "relative", "limit": 100.0}
    ]
    config["optimizer"].update(
        {
            "max_evaluations_per_objective": 2,
            "max_iterations": 1,
            "beam_width": 2,
        }
    )
    _install_fake_adapters(monkeypatch, paths)
    result = CacheOptimizationPipeline(config).run()
    assert result["evaluations"] == 1

    selection_root = (
        paths["output"]
        / "optimal_configs"
        / "gain_cell"
        / "power_constraints"
        / "wide_budget"
    )
    expected_status = {
        MIN_ENERGY_ISO_PERFORMANCE: "FEASIBLE",
        MAX_PERFORMANCE_ISO_ENERGY: "NO_FEASIBLE_CANDIDATE",
        BALANCED_KNEE: "FEASIBLE",
    }
    for selector, status in expected_status.items():
        payload = json.loads(
            (selection_root / selector / "selection.json").read_text()
        )
        assert payload["status"] == status
        assert payload["power_budget"]["limit"] == 100.0
