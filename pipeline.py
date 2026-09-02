"""End-to-end orchestration for NS-Cache and Accel-Sim."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Mapping

from . import __version__
from . import accelsim
from . import nscache
from .config import validate_paths
from .energy import build_evaluation, build_raw_app_result
from .models import AppResult, CacheDesign, Evaluation, to_jsonable
from .report import write_reports
from .selection import select_power_budget_optima
from .search import (
    AdaptiveParetoSearch,
    DiscreteSpace,
    MAX_PERFORMANCE,
    MIN_ENERGY,
    select_optimum,
)


class PipelineError(RuntimeError):
    pass


@dataclasses.dataclass
class PreparedCandidate:
    design: CacheDesign
    iteration: int
    l1_ppa: nscache.CachePPA
    l2_ppa: nscache.CachePPA
    gpu_config: Path
    generated_files: dict[str, str]
    l1_retention_time_us: float | None
    l2_retention_time_us: float | None
    area_ratio: float


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return result or "application"


def _application_slug(name: str, trace: Path) -> str:
    identity = _sha256_bytes(
        (name + "\0" + str(trace.resolve())).encode("utf-8")
    )[:10]
    return "{}-{}".format(_safe_name(name), identity)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _write_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except OSError:
        pass
    path.write_text(text, encoding="utf-8")


def _nscache_config_hashes(config_path: Path, nscache_root: Path) -> dict[str, str]:
    text = config_path.read_text(encoding="utf-8", errors="replace")
    result = {"config_sha256": _sha256_file(config_path)}
    raw_cell = nscache.memory_cell_input(text)
    if raw_cell is not None:
        cell = Path(raw_cell).expanduser()
        if not cell.is_absolute():
            cell = nscache_root / cell
        if cell.is_file():
            result["memory_cell_path"] = str(cell.resolve())
            result["memory_cell_sha256"] = _sha256_file(cell)
    return result


def _read_temperature_k(config_text: str) -> float:
    match = re.search(
        r"^\s*-Temperature\s*\(K\)\s*:\s*([0-9.eE+-]+)",
        config_text,
        flags=re.MULTILINE,
    )
    return float(match.group(1)) if match else 300.0


def _effective_retention_us(spec: Mapping[str, Any], config_text: str) -> float | None:
    retention = spec.get("retention_time_us")
    if retention is None:
        return None
    reference = spec.get("retention_reference_temperature_k")
    if reference is None:
        return float(retention)
    target = _read_temperature_k(config_text)
    # Matches MemCell::ApplyPVT in the bundled NS-Cache implementation.
    return float(retention) * math.exp(-0.0268 * (target - float(reference)))


def _git_revision(directory: Path) -> str | None:
    try:
        result = subprocess.run(
            ("git", "-C", str(directory), "rev-parse", "HEAD"),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


class CacheOptimizationPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.paths = config["paths"]
        self.baseline_cfg = config["baseline"]
        self.output_dir = Path(self.paths["output_dir"])
        self.work_dir = self.output_dir / "work"
        self.generated_dir = self.work_dir / "generated"
        self.sim_dir = self.work_dir / "simulations"
        self.cache_dir = self.work_dir / "cache"
        self.ppa_cache = nscache.JsonPPACache(self.cache_dir / "ppa.json")
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self._error_lock = threading.Lock()
        self._simulation_cache_lock = threading.Lock()
        self._shared_simulation_cache: dict[str, Path] | None = None
        self._baseline_l1_config: nscache.CacheConfig | None = None
        self._baseline_l2_config: nscache.CacheConfig | None = None
        self._accel_baseline_text: str | None = None
        self._orin_config: accelsim.OrinConfig | None = None
        self._environment: dict[str, str] | None = None
        self._accelsim_runtime_library: dict[str, str] | None = None
        self._trace_identities: dict[str, list[dict[str, Any]]] = {}
        self._baseline_l1_ppa: nscache.CachePPA | None = None
        self._baseline_l2_ppa: nscache.CachePPA | None = None
        self._baseline_apps: dict[str, AppResult] = {}
        self._baseline_area_mm2: float | None = None
        self._all_evaluations: list[Evaluation] = []

    def validate(self) -> list[str]:
        errors = validate_paths(self.config)
        if errors:
            return errors

        try:
            if self._baseline_l1_config is None:
                self._baseline_l1_config = nscache.read_config(
                    self.baseline_cfg["l1_cfg"]
                )
            if self._baseline_l2_config is None:
                self._baseline_l2_config = nscache.read_config(
                    self.baseline_cfg["l2_cfg"]
                )
            if self._accel_baseline_text is None:
                self._accel_baseline_text = Path(
                    self.paths["accelsim_config"]
                ).read_text(encoding="utf-8", errors="replace")
            if self._orin_config is None:
                self._orin_config = accelsim.parse_config_text(
                    self._accel_baseline_text
                )
        except (OSError, nscache.NSCacheError, accelsim.AccelSimError) as exc:
            errors.append(str(exc))
            return errors

        assert self._baseline_l1_config is not None
        assert self._baseline_l2_config is not None
        assert self._orin_config is not None

        if self._orin_config.l1_instances != self.baseline_cfg["l1_instances"]:
            errors.append(
                "baseline.l1_instances={} but Accel-Sim config contains {} shader cores".format(
                    self.baseline_cfg["l1_instances"], self._orin_config.l1_instances
                )
            )
        simulator_l2_kib = self._orin_config.l2_total_capacity_kib
        if simulator_l2_kib.denominator != 1 or simulator_l2_kib.numerator != self._baseline_l2_config.capacity_kib:
            errors.append(
                "NS-Cache L2 baseline is {} KiB but Accel-Sim L2 totals {} KiB".format(
                    self._baseline_l2_config.capacity_kib, simulator_l2_kib
                )
            )
        if self._baseline_l1_config.capacity_kib != self._orin_config.unified_l1d_size_kib:
            warning = (
                "NS-Cache L1 template is {} KiB per modeled instance while Accel-Sim's adaptive unified L1/shared pool is {} KiB per SM; candidate mapping is therefore ratio-based."
            ).format(
                self._baseline_l1_config.capacity_kib,
                self._orin_config.unified_l1d_size_kib,
            )
            if warning not in self.warnings:
                self.warnings.append(warning)
        if (
            self.baseline_cfg["l2_instances"] == 1
            and self._orin_config.l2_instances > 1
        ):
            warning = (
                "The immutable NS-Cache L2 baseline models one monolithic {} KiB array, "
                "whereas Accel-Sim partitions L2 across {} subpartitions. PPA is kept "
                "monolithic to preserve the supplied SRAM baseline; absolute slice-level "
                "latency/energy may differ."
            ).format(
                self._baseline_l2_config.capacity_kib,
                self._orin_config.l2_instances,
            )
            if warning not in self.warnings:
                self.warnings.append(warning)

        for app in self.config["applications"]:
            trace = Path(app["trace"])
            try:
                text = trace.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                errors.append("cannot read trace list for '{}': {}".format(app["name"], exc))
                continue
            if not text.strip():
                errors.append("trace list for '{}' is empty: {}".format(app["name"], trace))
                continue
            try:
                self._trace_identities[str(trace.resolve())] = self._resolve_trace_identities(
                    trace, text
                )
            except PipelineError as exc:
                errors.append("application '{}': {}".format(app["name"], exc))

        try:
            self._environment = accelsim.build_accelsim_environment(
                self.paths["accelsim_root"]
            )
            self._accelsim_runtime_library = self._runtime_library_identity(
                self._environment
            )
        except accelsim.AccelSimError as exc:
            errors.append(str(exc))
        except PipelineError as exc:
            errors.append(str(exc))
        return errors

    @staticmethod
    def _runtime_library_identity(environment: Mapping[str, str]) -> dict[str, str]:
        for directory in environment.get("LD_LIBRARY_PATH", "").split(os.pathsep):
            if not directory:
                continue
            library = Path(directory) / "libcudart.so"
            if library.is_file():
                return {
                    "path": str(library.resolve()),
                    "sha256": _sha256_file(library),
                }
        raise PipelineError(
            "Accel-Sim environment contains no loadable libcudart.so in LD_LIBRARY_PATH"
        )

    def run(self) -> dict[str, Any]:
        errors = self.validate()
        if errors:
            raise PipelineError("Configuration validation failed:\n- " + "\n- ".join(errors))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.errors = []
        self._all_evaluations = []
        self._baseline_apps = {}
        _atomic_json(self.output_dir / "errors.json", [])
        (self.output_dir / "partial_evaluations.jsonl").write_text("", encoding="utf-8")

        assert self._baseline_l1_config is not None
        assert self._baseline_l2_config is not None

        print("[baseline] Evaluating immutable SRAM PPA and traces", flush=True)
        self._prepare_baseline()
        assert self._baseline_area_mm2 is not None

        space = DiscreteSpace.from_config(self.config["search_space"])
        outcomes: dict[str, Any] = {}
        for technology in sorted(self.config["technologies"]):
            for objective in self.config["objectives"]:
                label = "{}::{}".format(technology, objective)
                print("[search] {}".format(label), flush=True)
                search = AdaptiveParetoSearch(
                    technology=technology,
                    objective=objective,
                    space=space,
                    baseline_l1_capacity_kib=self._baseline_l1_config.capacity_kib,
                    baseline_l1_associativity=self._baseline_l1_config.associativity,
                    baseline_l2_capacity_kib=self._baseline_l2_config.capacity_kib,
                    baseline_l2_associativity=self._baseline_l2_config.associativity,
                    max_evaluations=self.config["optimizer"]["max_evaluations_per_objective"],
                    max_iterations=self.config["optimizer"]["max_iterations"],
                    beam_width=self.config["optimizer"]["beam_width"],
                    power_constraints=self.config.get("power_constraints", []),
                    constraints=self.config["constraints"],
                )
                outcome = search.run(
                    lambda designs, iteration, technology=technology: self._evaluate_batch(
                        technology, designs, iteration
                    )
                )
                self._all_evaluations.extend(outcome.evaluations)
                outcomes[label] = {
                    "stopped_reason": outcome.stopped_reason,
                    "attempted": outcome.attempted,
                    "evaluations": len(outcome.evaluations),
                    "errors": outcome.errors,
                    "optimum": to_jsonable(outcome.optimum),
                }
                self._write_checkpoint(outcomes)

        missing_groups = sorted(
            label for label, outcome in outcomes.items() if outcome["evaluations"] == 0
        )
        status = "incomplete" if missing_groups else "complete"
        metadata = self._metadata(outcomes, status=status)
        self._copy_optimal_configs()
        artifacts = write_reports(
            self.output_dir,
            self._all_evaluations,
            baseline=self._baseline_payload(),
            metadata=metadata,
            power_constraints=self.config.get("power_constraints", []),
            constraints=self.config["constraints"],
        )
        _atomic_json(self.output_dir / "manifest.json", metadata)
        print("[done] Report: {}".format(artifacts["report_html"]), flush=True)
        if missing_groups:
            raise PipelineError(
                "No candidate completed for: {}. Partial report: {}".format(
                    ", ".join(missing_groups), artifacts["report_html"]
                )
            )
        return {
            "artifacts": artifacts,
            "metadata": metadata,
            "evaluations": len(self._all_evaluations),
        }

    def _prepare_baseline(self) -> None:
        assert self._orin_config is not None
        nsc_root = self.paths["nscache_root"]
        timeout = self.config["optimizer"]["nsc_timeout_s"]
        self._baseline_l1_ppa, _ = nscache.cached_run_nscache(
            self.ppa_cache,
            self.paths["nsc_binary"],
            self.baseline_cfg["l1_cfg"],
            cwd=nsc_root,
            timeout_s=timeout,
        )
        self._baseline_l2_ppa, _ = nscache.cached_run_nscache(
            self.ppa_cache,
            self.paths["nsc_binary"],
            self.baseline_cfg["l2_cfg"],
            cwd=nsc_root,
            timeout_s=timeout,
        )
        self._baseline_area_mm2 = (
            self._baseline_l1_ppa.area_mm2 * self.baseline_cfg["l1_instances"]
            + self._baseline_l2_ppa.area_mm2 * self.baseline_cfg["l2_instances"]
        )

        assert self._environment is not None
        clock_hz = self._orin_config.clocks.core_hz
        for app in self.config["applications"]:
            stats = self._run_sim_cached(
                cache_id="sram-baseline",
                application=app,
                gpu_config=Path(self.paths["accelsim_config"]),
            )
            self._baseline_apps[app["name"]] = build_raw_app_result(
                app["name"],
                stats,
                clock_hz=clock_hz,
                l1_ppa=self._baseline_l1_ppa,
                l2_ppa=self._baseline_l2_ppa,
                l1_instances=self.baseline_cfg["l1_instances"],
                l2_instances=self.baseline_cfg["l2_instances"],
                l1_counts_scope=self.baseline_cfg["l1_counts_scope"],
                l2_counts_scope=self.baseline_cfg["l2_counts_scope"],
                l1_bank_count=self.baseline_cfg["l1_banks_per_instance"],
                l2_bank_count=self.baseline_cfg["l2_banks_per_instance"],
                l1_retention_time_us=None,
                l2_retention_time_us=None,
            )
        _atomic_json(self.output_dir / "baseline.json", self._baseline_payload())

    def _prepare_generated_cell(self, technology: str) -> Path | None:
        spec = self.config["technologies"][technology]
        source = spec.get("cell_template")
        if source is None:
            return None
        text = Path(source).read_text(encoding="utf-8", errors="replace")
        text = nscache.patch_cell_text(
            text,
            retention_time_us=spec.get("retention_time_us"),
            temperature_k=spec.get("retention_reference_temperature_k"),
        )
        path = self.generated_dir / "cells" / (technology + ".cell")
        _write_if_changed(path, text)
        return path

    def _candidate_ppa(
        self,
        technology: str,
        level: str,
        capacity_kib: int,
        associativity: int,
        candidate_dir: Path,
    ) -> tuple[nscache.CachePPA, Path, float | None]:
        spec = self.config["technologies"][technology]
        template_path = Path(spec[level + "_template"])
        config_text = nscache.render_config(
            template_path,
            capacity_kib=capacity_kib,
            associativity=associativity,
        )
        cell_path = self._prepare_generated_cell(technology)
        if cell_path is not None:
            config_text = nscache.patch_memory_cell_input(config_text, cell_path)
        generated = candidate_dir / ("nsc_" + level + ".cfg")
        _write_if_changed(generated, config_text)
        ppa, _ = nscache.cached_run_nscache(
            self.ppa_cache,
            self.paths["nsc_binary"],
            generated,
            cwd=self.paths["nscache_root"],
            timeout_s=self.config["optimizer"]["nsc_timeout_s"],
        )
        effective_retention = _effective_retention_us(spec, config_text)
        if effective_retention is not None and ppa.refresh_energy_nj is None:
            raise nscache.NSCacheRunError(
                "{} {} has configured retention but NS-Cache reported no refresh energy".format(
                    technology, level.upper()
                )
            )
        return ppa, generated, effective_retention

    def _prepare_candidate(
        self, technology: str, design: CacheDesign, iteration: int
    ) -> PreparedCandidate | None:
        assert self._baseline_l1_ppa is not None
        assert self._baseline_l2_ppa is not None
        assert self._baseline_area_mm2 is not None
        assert self._baseline_l1_config is not None
        assert self._baseline_l2_config is not None
        assert self._accel_baseline_text is not None
        candidate_dir = self.generated_dir / "candidates" / design.id
        generated_files: dict[str, str] = {}

        # Reject fractional set counts, unsupported IPOLY sizes, and invalid
        # adaptive L1/shared mappings before spending tens of seconds on PPA.
        accelsim.generate_candidate_config(
            self._accel_baseline_text,
            baseline_l1_capacity_kib=self._baseline_l1_config.capacity_kib,
            baseline_l1_associativity=self._baseline_l1_config.associativity,
            baseline_l2_capacity_kib=self._baseline_l2_config.capacity_kib,
            baseline_l2_associativity=self._baseline_l2_config.associativity,
            l1_capacity_kib=design.l1_capacity_kib,
            l1_associativity=design.l1_associativity,
            l2_capacity_kib=design.l2_capacity_kib,
            l2_associativity=design.l2_associativity,
        )

        if design.l1_technology == "sram":
            l1_ppa = self._baseline_l1_ppa
            l1_cfg = Path(self.baseline_cfg["l1_cfg"])
            l1_retention = None
        else:
            l1_ppa, l1_cfg, l1_retention = self._candidate_ppa(
                technology,
                "l1",
                design.l1_capacity_kib,
                design.l1_associativity,
                candidate_dir,
            )
        l2_ppa, l2_cfg, l2_retention = self._candidate_ppa(
            technology,
            "l2",
            design.l2_capacity_kib,
            design.l2_associativity,
            candidate_dir,
        )
        generated_files["nsc_l1_config"] = str(l1_cfg)
        generated_files["nsc_l2_config"] = str(l2_cfg)

        area = (
            l1_ppa.area_mm2 * self.baseline_cfg["l1_instances"]
            + l2_ppa.area_mm2 * self.baseline_cfg["l2_instances"]
        )
        area_ratio = area / self._baseline_area_mm2
        if area_ratio > self.config["optimizer"]["saturation_area_ratio_max"]:
            self._record_error(
                design,
                iteration,
                "PPA pre-screen: area ratio {:.6g} exceeds saturation probe limit {:.6g}".format(
                    area_ratio, self.config["optimizer"]["saturation_area_ratio_max"]
                ),
            )
            return None

        gpu_text = accelsim.generate_candidate_config(
            self._accel_baseline_text,
            baseline_l1_capacity_kib=self._baseline_l1_config.capacity_kib,
            baseline_l1_associativity=self._baseline_l1_config.associativity,
            baseline_l2_capacity_kib=self._baseline_l2_config.capacity_kib,
            baseline_l2_associativity=self._baseline_l2_config.associativity,
            l1_capacity_kib=design.l1_capacity_kib,
            l1_associativity=design.l1_associativity,
            l2_capacity_kib=design.l2_capacity_kib,
            l2_associativity=design.l2_associativity,
            baseline_l1_latency_ns=self._baseline_l1_ppa.hit_latency_ns,
            l1_latency_ns=l1_ppa.hit_latency_ns,
            baseline_l2_latency_ns=self._baseline_l2_ppa.hit_latency_ns,
            l2_latency_ns=l2_ppa.hit_latency_ns,
            minimum_latency_cycles=self.config["latency_mapping"]["minimum_cycles"],
        )
        gpu_config = candidate_dir / "gpgpusim.config"
        _write_if_changed(gpu_config, gpu_text)
        generated_files["gpgpusim_config"] = str(gpu_config)
        return PreparedCandidate(
            design=design,
            iteration=iteration,
            l1_ppa=l1_ppa,
            l2_ppa=l2_ppa,
            gpu_config=gpu_config,
            generated_files=generated_files,
            l1_retention_time_us=l1_retention,
            l2_retention_time_us=l2_retention,
            area_ratio=area_ratio,
        )

    def _evaluate_batch(
        self, technology: str, designs: list[CacheDesign], iteration: int
    ) -> list[Evaluation]:
        prepared: list[PreparedCandidate] = []
        for design in designs:
            print("  [PPA] {}".format(design.id), flush=True)
            try:
                candidate = self._prepare_candidate(technology, design, iteration)
                if candidate is not None:
                    prepared.append(candidate)
            except Exception as exc:
                self._record_error(design, iteration, "PPA/config: {}".format(exc))

        workers = min(self.config["optimizer"]["max_workers"], max(len(prepared), 1))
        results: list[Evaluation] = []
        if workers == 1:
            for candidate in prepared:
                value = self._evaluate_prepared(candidate)
                if value is not None:
                    results.append(value)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(self._evaluate_prepared, candidate): candidate
                    for candidate in prepared
                }
                for future in concurrent.futures.as_completed(future_map):
                    value = future.result()
                    if value is not None:
                        results.append(value)
        results.sort(key=lambda evaluation: evaluation.design.id)
        self._write_partial_evaluations(results)
        return results

    def _evaluate_prepared(self, candidate: PreparedCandidate) -> Evaluation | None:
        assert self._orin_config is not None
        print("  [SIM] {}".format(candidate.design.id), flush=True)
        try:
            stats_by_app = {
                app["name"]: self._run_sim_cached(
                    cache_id=candidate.design.id,
                    application=app,
                    gpu_config=candidate.gpu_config,
                )
                for app in self.config["applications"]
            }
            evaluation = build_evaluation(
                design=candidate.design,
                iteration=candidate.iteration,
                stats_by_app=stats_by_app,
                baseline_by_app=self._baseline_apps,
                weights_by_app={app["name"]: app["weight"] for app in self.config["applications"]},
                clock_hz=self._orin_config.clocks.core_hz,
                l1_ppa=candidate.l1_ppa,
                l2_ppa=candidate.l2_ppa,
                baseline_area_mm2=float(self._baseline_area_mm2),
                l1_instances=self.baseline_cfg["l1_instances"],
                l2_instances=self.baseline_cfg["l2_instances"],
                l1_counts_scope=self.baseline_cfg["l1_counts_scope"],
                l2_counts_scope=self.baseline_cfg["l2_counts_scope"],
                l1_bank_count=self.baseline_cfg["l1_banks_per_instance"],
                l2_bank_count=self.baseline_cfg["l2_banks_per_instance"],
                l1_retention_time_us=candidate.l1_retention_time_us,
                l2_retention_time_us=candidate.l2_retention_time_us,
                constraints=self.config["constraints"],
                generated_files=candidate.generated_files,
            )
            print(
                "    area={:.3f} perf={:.3f} energy={:.3f} feasible={}".format(
                    evaluation.area_ratio,
                    evaluation.performance_score,
                    evaluation.energy_score,
                    evaluation.feasible,
                ),
                flush=True,
            )
            return evaluation
        except Exception as exc:
            self._record_error(candidate.design, candidate.iteration, "simulation/evaluation: {}".format(exc))
            return None

    def _run_sim_cached(
        self,
        *,
        cache_id: str,
        application: Mapping[str, Any],
        gpu_config: Path,
    ) -> accelsim.AccelSimOutput:
        trace = Path(str(application["trace"]))
        run_dir = self.sim_dir / cache_id / _application_slug(
            str(application["name"]), trace
        )
        log_path = run_dir / "accelsim.stdout.log"
        metadata_path = run_dir / "cache_key.json"
        key = self._simulation_cache_key(trace, gpu_config)

        if self.config["optimizer"]["resume"] and log_path.is_file() and metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("key") == key:
                    output = accelsim.parse_output(
                        log_path.read_text(encoding="utf-8", errors="replace")
                    )
                    output.require_success()
                    print("    [cache] {}".format(application["name"]), flush=True)
                    return output
            except (OSError, ValueError, accelsim.AccelSimError):
                pass

        if self.config["optimizer"]["resume"]:
            shared_log = self._find_shared_simulation(key, exclude=log_path)
            if shared_log is not None:
                try:
                    output = accelsim.parse_output(
                        shared_log.read_text(encoding="utf-8", errors="replace")
                    )
                    output.require_success()
                    print(
                        "    [cache:physical] {}".format(application["name"]),
                        flush=True,
                    )
                    return output
                except (OSError, ValueError, accelsim.AccelSimError):
                    with self._simulation_cache_lock:
                        if self._shared_simulation_cache is not None:
                            self._shared_simulation_cache.pop(key, None)

        assert self._environment is not None
        result = accelsim.run_accelsim(
            self.paths["accelsim_binary"],
            trace,
            gpu_config,
            self.paths["trace_config"],
            work_dir=run_dir,
            environment=self._environment,
            timeout_s=self.config["optimizer"]["sim_timeout_s"],
            overwrite_log=True,
            check=True,
        )
        _atomic_json(
            metadata_path,
            {
                "key": key,
                "application": application["name"],
                "trace": str(trace),
                "gpu_config": str(gpu_config),
                "elapsed_s": result.elapsed_s,
            },
        )
        with self._simulation_cache_lock:
            if self._shared_simulation_cache is not None:
                self._shared_simulation_cache[key] = log_path
        return result.output

    def _find_shared_simulation(self, key: str, *, exclude: Path) -> Path | None:
        """Find a completed simulation of the same physical GPU config.

        Candidate IDs include the search policy for provenance.  A unified
        Pareto run can nevertheless reuse a legacy objective's expensive
        Accel-Sim result when the binary, trace, and generated GPU config hashes
        are identical.
        """
        with self._simulation_cache_lock:
            if self._shared_simulation_cache is None:
                cache: dict[str, Path] = {}
                for metadata_path in self.sim_dir.glob("*/*/cache_key.json"):
                    try:
                        metadata = json.loads(
                            metadata_path.read_text(encoding="utf-8")
                        )
                        cached_key = metadata.get("key")
                        candidate_log = metadata_path.with_name(
                            "accelsim.stdout.log"
                        )
                        if isinstance(cached_key, str) and candidate_log.is_file():
                            cache.setdefault(cached_key, candidate_log)
                    except (OSError, ValueError):
                        continue
                self._shared_simulation_cache = cache
            found = self._shared_simulation_cache.get(key)
            if found is None or found == exclude:
                return None
            return found

    def _simulation_cache_key(self, trace: Path, gpu_config: Path) -> str:
        trace_text = trace.read_text(encoding="utf-8", errors="replace")
        trace_key = str(trace.resolve())
        trace_entries = self._trace_identities.get(trace_key)
        if trace_entries is None:
            trace_entries = self._resolve_trace_identities(trace, trace_text)
            self._trace_identities[trace_key] = trace_entries
        payload = {
            "schema": 2,
            "binary_sha256": _sha256_file(Path(self.paths["accelsim_binary"])),
            "runtime_library": self._accelsim_runtime_library,
            "gpu_config_sha256": _sha256_file(gpu_config),
            "trace_config_sha256": _sha256_file(Path(self.paths["trace_config"])),
            "kernelslist_sha256": _sha256_bytes(trace_text.encode("utf-8")),
            "trace_entries": trace_entries,
        }
        return _sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))

    @staticmethod
    def _resolve_trace_identities(
        kernelslist: Path, text: str
    ) -> list[dict[str, Any]]:
        identities: list[dict[str, Any]] = []
        kernel_lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip().startswith("kernel")
        ]
        if not kernel_lines:
            raise PipelineError("kernelslist contains no kernel trace entries: {}".format(kernelslist))
        for entry in kernel_lines:
            target = kernelslist.parent / entry
            alternatives = (target, Path(str(target) + ".xz"))
            existing = next((path for path in alternatives if path.is_file()), None)
            if existing is None:
                raise PipelineError(
                    "referenced trace is missing (also checked .xz): {}".format(target)
                )
            stat = existing.stat()
            identities.append(
                {
                    "path": str(existing.resolve()),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        return identities

    def _record_error(self, design: CacheDesign, iteration: int, message: str) -> None:
        row = {
            "candidate": design.id,
            "technology": design.technology,
            "objective": design.objective,
            "iteration": iteration,
            "error": message,
        }
        with self._error_lock:
            self.errors.append(row)
            _atomic_json(self.output_dir / "errors.json", self.errors)
        print("    [skip] {}".format(message), flush=True)

    def _write_partial_evaluations(self, values: list[Evaluation]) -> None:
        path = self.output_dir / "partial_evaluations.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(to_jsonable(value), sort_keys=True) + "\n")

    def _write_checkpoint(self, outcomes: Mapping[str, Any]) -> None:
        _atomic_json(
            self.output_dir / "checkpoint.json",
            {
                "outcomes": outcomes,
                "errors": self.errors,
                "evaluations": self._all_evaluations,
            },
        )

    def _copy_optimal_configs(self) -> None:
        groups: dict[tuple[str, str], list[Evaluation]] = {}
        for evaluation in self._all_evaluations:
            groups.setdefault(
                (evaluation.design.technology, evaluation.design.objective), []
            ).append(evaluation)
        root = self.output_dir / "optimal_configs"
        if root.exists():
            shutil.rmtree(root)
        for (technology, objective), values in groups.items():
            optimum = select_optimum(values, objective)
            if optimum is None:
                continue
            target = root / technology / objective
            target.mkdir(parents=True, exist_ok=True)
            for key, level in (
                ("nsc_l1_config", "l1"),
                ("nsc_l2_config", "l2"),
            ):
                source = optimum.generated_files.get(key)
                if source and Path(source).is_file():
                    self._copy_self_contained_nsc_config(
                        Path(source), target / ("nsc_" + level + ".cfg"), level
                    )
            gpu_source = optimum.generated_files.get("gpgpusim_config")
            if gpu_source and Path(gpu_source).is_file():
                shutil.copy2(gpu_source, target / "gpgpusim.config")
            _atomic_json(
                target / "selection.json",
                {
                    "status": "FEASIBLE" if optimum.feasible else "NO_FEASIBLE_CANDIDATE",
                    "evaluation": optimum,
                },
            )

        for selection in select_power_budget_optima(
            self._all_evaluations,
            self.config.get("power_constraints", []),
            self.config["constraints"],
        ):
            target = (
                root
                / selection.technology
                / "power_constraints"
                / selection.budget["name"]
                / selection.selector
            )
            target.mkdir(parents=True, exist_ok=True)
            optimum = selection.evaluation
            if optimum is not None:
                for key, level in (
                    ("nsc_l1_config", "l1"),
                    ("nsc_l2_config", "l2"),
                ):
                    source = optimum.generated_files.get(key)
                    if source and Path(source).is_file():
                        self._copy_self_contained_nsc_config(
                            Path(source),
                            target / ("nsc_" + level + ".cfg"),
                            level,
                        )
                gpu_source = optimum.generated_files.get("gpgpusim_config")
                if gpu_source and Path(gpu_source).is_file():
                    shutil.copy2(gpu_source, target / "gpgpusim.config")
            _atomic_json(
                target / "selection.json",
                {
                    "status": selection.status,
                    "technology": selection.technology,
                    "selector": selection.selector,
                    "power_budget": selection.budget,
                    "observed_power": selection.observed_power,
                    "observed_unit": selection.observed_unit,
                    "reasons": selection.reasons,
                    "evaluation": optimum,
                },
            )

    def _copy_self_contained_nsc_config(
        self, source: Path, destination: Path, level: str
    ) -> None:
        text = source.read_text(encoding="utf-8", errors="replace")
        raw_cell = nscache.memory_cell_input(text)
        if raw_cell is not None:
            cell_source = Path(raw_cell).expanduser()
            if not cell_source.is_absolute():
                cell_source = Path(self.paths["nscache_root"]) / cell_source
            if not cell_source.is_file():
                raise PipelineError(
                    "cannot package optimal {} cell file: {}".format(level, cell_source)
                )
            cells_dir = destination.parent / "cells"
            cells_dir.mkdir(parents=True, exist_ok=True)
            suffix = cell_source.suffix or ".cell"
            cell_destination = cells_dir / (level + suffix)
            shutil.copy2(cell_source, cell_destination)
            text = nscache.patch_memory_cell_input(
                text, Path("cells") / cell_destination.name
            )
        _write_if_changed(destination, text)

    def _baseline_payload(self) -> dict[str, Any]:
        assert self._baseline_l1_config is not None
        assert self._baseline_l2_config is not None
        return {
            "technology": "sram",
            "immutable": True,
            "l1_config": self.baseline_cfg["l1_cfg"],
            "l2_config": self.baseline_cfg["l2_cfg"],
            "accelsim_config": self.paths["accelsim_config"],
            "l1_capacity_kib": self._baseline_l1_config.capacity_kib,
            "l1_associativity": self._baseline_l1_config.associativity,
            "l2_capacity_kib": self._baseline_l2_config.capacity_kib,
            "l2_associativity": self._baseline_l2_config.associativity,
            "l1_instances": self.baseline_cfg["l1_instances"],
            "l2_instances": self.baseline_cfg["l2_instances"],
            "area_mm2": self._baseline_area_mm2,
            "l1_ppa": self._baseline_l1_ppa,
            "l2_ppa": self._baseline_l2_ppa,
            "applications": self._baseline_apps,
        }

    def _metadata(
        self, outcomes: Mapping[str, Any], *, status: str
    ) -> dict[str, Any]:
        return {
            "status": status,
            "pipeline_version": __version__,
            "config_path": self.config["config_path"],
            "warnings": self.warnings,
            "errors": self.errors,
            "outcomes": outcomes,
            "assumptions": {
                "performance_metric": "GPU trace cycles (gpu_tot_sim_cycle)",
                "energy_metric": "NS-Cache access dynamic + leakage*time + refresh*time",
                "power_metric": "workload-average modeled L1+L2 cache-array power; not Jetson board/GPU TDP or peak power",
                "suite_aggregation": "weighted geometric mean of per-application normalized ratios",
                "power_constraint_scope": "per_application uses the worst application; suite uses the weighted geometric mean",
                "pareto_dimensions": "minimize area, cache energy, and cache power; maximize application performance",
                "latency_mapping": "calibrated Accel-Sim baseline plus SRAM-relative NS-Cache hit-latency delta",
                "performance_model": "hit-latency delta only; write-latency and refresh-stall effects are not injected into Accel-Sim",
                "adaptive_l1_mapping": "unified L1/shared pool scales with target L1 ratio while Orin shared-memory limits remain fixed; candidates below the fixed shared-memory limit are rejected",
                "l2_physical_model": "supplied immutable 4 MiB monolithic NS-Cache array; Accel-Sim access behavior remains 16-subpartition",
                "tag_technology": "same as data technology in the supplied NS-Cache templates",
                "l1_dynamic_counts": self.baseline_cfg["l1_counts_scope"],
                "l2_dynamic_counts": self.baseline_cfg["l2_counts_scope"],
            },
            "constraints": self.config["constraints"],
            "power_constraints": self.config.get("power_constraints", []),
            "search_space": self.config["search_space"],
            "objectives": self.config["objectives"],
            "technology_model_notes": {
                name: spec.get("model_note", "")
                for name, spec in self.config["technologies"].items()
            },
            "technologies": {
                name: {
                    "retention_time_us": spec.get("retention_time_us"),
                    "retention_reference_temperature_k": spec.get(
                        "retention_reference_temperature_k"
                    ),
                    "model_note": spec.get("model_note", ""),
                }
                for name, spec in self.config["technologies"].items()
            },
            "applications": [
                {
                    "name": app["name"],
                    "weight": app["weight"],
                    "kernelslist": app["trace"],
                    "kernelslist_sha256": _sha256_file(Path(app["trace"])),
                    "trace_files": self._trace_identities.get(
                        str(Path(app["trace"]).resolve()), []
                    ),
                }
                for app in self.config["applications"]
            ],
            "revisions": {
                "nscache": _git_revision(Path(self.paths["nscache_root"])),
                "accelsim": _git_revision(Path(self.paths["accelsim_root"])),
            },
            "input_hashes": {
                "nsc_binary": _sha256_file(Path(self.paths["nsc_binary"])),
                "accelsim_binary": _sha256_file(Path(self.paths["accelsim_binary"])),
                "accelsim_runtime_library": self._accelsim_runtime_library,
                "accelsim_config": _sha256_file(Path(self.paths["accelsim_config"])),
                "trace_config": _sha256_file(Path(self.paths["trace_config"])),
                "baseline_l1_cfg": _sha256_file(Path(self.baseline_cfg["l1_cfg"])),
                "baseline_l2_cfg": _sha256_file(Path(self.baseline_cfg["l2_cfg"])),
                "nscache_configs": {
                    "baseline_l1": _nscache_config_hashes(
                        Path(self.baseline_cfg["l1_cfg"]),
                        Path(self.paths["nscache_root"]),
                    ),
                    "baseline_l2": _nscache_config_hashes(
                        Path(self.baseline_cfg["l2_cfg"]),
                        Path(self.paths["nscache_root"]),
                    ),
                    **{
                        "{}_{}".format(name, level): _nscache_config_hashes(
                            Path(spec[level + "_template"]),
                            Path(self.paths["nscache_root"]),
                        )
                        for name, spec in self.config["technologies"].items()
                        for level in ("l1", "l2")
                    },
                },
                "technology_inputs": {
                    name: {
                        key: _sha256_file(Path(spec[key]))
                        for key in ("l1_template", "l2_template", "cell_template")
                        if spec.get(key) is not None
                    }
                    for name, spec in self.config["technologies"].items()
                },
            },
        }
