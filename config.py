"""JSON configuration loading and validation."""

from __future__ import annotations

import copy
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    pass


OBJECTIVES = (
    "min_energy_runtime_bound",
    "max_performance_energy_bound",
    "pareto_full_cache",
)

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


DEFAULTS: dict[str, Any] = {
    "constraints": {
        "max_area_ratio": 1.0,
        "runtime_tolerance": 0.02,
        "energy_tolerance": 0.0,
        "constraint_scope": "per_application",
    },
    "optimizer": {
        "max_evaluations_per_objective": 24,
        "max_iterations": 12,
        "beam_width": 5,
        "max_workers": 1,
        "nsc_timeout_s": 1200,
        "sim_timeout_s": 86400,
        "resume": True,
        "saturation_area_ratio_max": 2.0,
    },
    "latency_mapping": {
        "mode": "relative_delta",
        "rounding": "ceil_away_from_zero",
        "minimum_cycles": 1,
    },
    "objectives": [
        "min_energy_runtime_bound",
        "max_performance_energy_bound",
    ],
    "power_constraints": [
        {
            "name": "sram_1x",
            "max_power_ratio": 1.0,
        },
        {
            "name": "sram_1p5x",
            "max_power_ratio": 1.5,
        },
    ],
}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_path(raw: str, base: Path) -> str:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required key '{context}.{key}'")
    return mapping[key]


def _positive_int_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{label} must be a non-empty list")
    result: list[int] = []
    for item in value:
        result.append(_positive_int(item, f"{label} value"))
    if len(set(result)) != len(result):
        raise ConfigError(f"{label} contains duplicate values")
    return sorted(result)


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{label} must be a finite number")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a positive integer")
    number = _finite_float(value, label)
    if number <= 0 or not number.is_integer():
        raise ConfigError(f"{label} must be a positive integer")
    return int(number)


def _normalize_power_constraints(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ConfigError("power_constraints must be a non-empty list")

    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    allowed_keys = {"name", "max_power_ratio", "max_power_mw"}
    for index, entry in enumerate(value):
        label = f"power_constraints[{index}]"
        if not isinstance(entry, dict):
            raise ConfigError(f"{label} must be an object")
        unknown_keys = sorted(set(entry) - allowed_keys)
        if unknown_keys:
            raise ConfigError(
                f"{label} contains unknown keys: {', '.join(unknown_keys)}"
            )

        name = entry.get("name")
        if not isinstance(name, str) or not _SAFE_NAME_RE.fullmatch(name):
            raise ConfigError(
                f"{label}.name must use only letters, digits, '.', '_' or '-' "
                "and cannot start with punctuation"
            )
        if name in seen_names:
            raise ConfigError(f"Duplicate power constraint name: {name}")

        limit_keys = [
            key for key in ("max_power_ratio", "max_power_mw") if key in entry
        ]
        if len(limit_keys) != 1:
            raise ConfigError(
                f"{label} must define exactly one of max_power_ratio or max_power_mw"
            )
        limit_key = limit_keys[0]
        limit = _finite_float(entry[limit_key], f"{label}.{limit_key}")
        if limit <= 0:
            raise ConfigError(f"{label}.{limit_key} must be > 0")

        normalized.append(
            {
                "name": name,
                "type": (
                    "relative" if limit_key == "max_power_ratio" else "absolute_mw"
                ),
                "limit": limit,
            }
        )
        seen_names.add(name)

    return normalized


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"Cannot read config {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be an object")

    cfg = _deep_merge(DEFAULTS, raw)
    cfg["config_path"] = str(path)
    cfg["config_dir"] = str(path.parent)

    paths = _require(cfg, "paths", "config")
    if not isinstance(paths, dict):
        raise ConfigError("paths must be an object")
    for key in ("nscache_root", "nsc_binary", "accelsim_root", "accelsim_binary", "accelsim_config", "trace_config", "output_dir"):
        paths[key] = _resolve_path(str(_require(paths, key, "paths")), path.parent)

    baseline = _require(cfg, "baseline", "config")
    if not isinstance(baseline, dict):
        raise ConfigError("baseline must be an object")
    for key in ("l1_cfg", "l2_cfg"):
        baseline[key] = _resolve_path(str(_require(baseline, key, "baseline")), path.parent)
    baseline.setdefault("l1_instances", 1)
    baseline.setdefault("l2_instances", 1)
    baseline.setdefault("l1_banks_per_instance", 1)
    baseline.setdefault("l2_banks_per_instance", 1)
    for key in (
        "l1_instances",
        "l2_instances",
        "l1_banks_per_instance",
        "l2_banks_per_instance",
    ):
        baseline[key] = _positive_int(baseline[key], f"baseline.{key}")
    baseline.setdefault("l1_counts_scope", "all_instances")
    baseline.setdefault("l2_counts_scope", "all_instances")
    for key in ("l1_counts_scope", "l2_counts_scope"):
        if baseline[key] not in {"all_instances", "per_instance"}:
            raise ConfigError(
                f"baseline.{key} must be 'all_instances' or 'per_instance'"
            )

    technologies = _require(cfg, "technologies", "config")
    if not isinstance(technologies, dict) or not technologies:
        raise ConfigError("technologies must be a non-empty object")
    for name, spec in technologies.items():
        if not isinstance(name, str) or not _SAFE_NAME_RE.fullmatch(name):
            raise ConfigError(
                "Technology names must use only letters, digits, '.', '_' or '-' "
                f"and cannot start with punctuation: {name!r}"
            )
        if name.lower() == "sram":
            raise ConfigError("Do not put SRAM in technologies; SRAM is the immutable baseline")
        if not isinstance(spec, dict):
            raise ConfigError(f"technologies.{name} must be an object")
        for key in ("l1_template", "l2_template"):
            spec[key] = _resolve_path(str(_require(spec, key, f"technologies.{name}")), path.parent)
        if spec.get("cell_template") is not None:
            spec["cell_template"] = _resolve_path(str(spec["cell_template"]), path.parent)
        retention = spec.get("retention_time_us")
        if retention is not None:
            retention = _finite_float(
                retention, f"technologies.{name}.retention_time_us"
            )
            if retention <= 0:
                raise ConfigError(f"technologies.{name}.retention_time_us must be > 0")
        spec["retention_time_us"] = retention
        reference_temperature = spec.get("retention_reference_temperature_k")
        if reference_temperature is not None:
            reference_temperature = _finite_float(
                reference_temperature,
                f"technologies.{name}.retention_reference_temperature_k",
            )
            if reference_temperature <= 0:
                raise ConfigError(
                    f"technologies.{name}.retention_reference_temperature_k must be > 0"
                )
        spec["retention_reference_temperature_k"] = reference_temperature
        if spec["retention_time_us"] is not None and not spec.get("cell_template"):
            raise ConfigError(
                f"technologies.{name}.cell_template is required when retention_time_us is set"
            )

    space = _require(cfg, "search_space", "config")
    if not isinstance(space, dict):
        raise ConfigError("search_space must be an object")
    for key in ("l1_capacity_kib", "l1_associativity", "l2_capacity_kib", "l2_associativity"):
        space[key] = _positive_int_list(_require(space, key, "search_space"), f"search_space.{key}")

    objectives = cfg.get("objectives")
    if not isinstance(objectives, list) or not objectives:
        raise ConfigError("objectives must be a non-empty list")
    if any(not isinstance(objective, str) for objective in objectives):
        raise ConfigError("objectives values must be strings")
    if len(set(objectives)) != len(objectives):
        raise ConfigError("objectives contains duplicate values")
    unknown = sorted(set(objectives) - set(OBJECTIVES))
    if unknown:
        raise ConfigError(f"Unknown objectives: {', '.join(unknown)}")

    cfg["power_constraints"] = _normalize_power_constraints(
        cfg.get("power_constraints")
    )

    constraints = cfg["constraints"]
    if not isinstance(constraints, dict):
        raise ConfigError("constraints must be an object")
    if constraints["constraint_scope"] not in {"per_application", "suite"}:
        raise ConfigError("constraints.constraint_scope must be 'per_application' or 'suite'")
    for key in ("max_area_ratio", "runtime_tolerance", "energy_tolerance"):
        constraints[key] = _finite_float(constraints.get(key), f"constraints.{key}")
    if constraints["max_area_ratio"] <= 0:
        raise ConfigError("constraints.max_area_ratio must be > 0")
    if constraints["runtime_tolerance"] < 0 or constraints["energy_tolerance"] < 0:
        raise ConfigError("constraint tolerances must be >= 0")

    optimizer = cfg["optimizer"]
    if not isinstance(optimizer, dict):
        raise ConfigError("optimizer must be an object")
    for key in ("max_evaluations_per_objective", "max_iterations", "beam_width", "max_workers", "nsc_timeout_s", "sim_timeout_s"):
        optimizer[key] = _positive_int(optimizer.get(key), f"optimizer.{key}")
    if not isinstance(optimizer.get("resume"), bool):
        raise ConfigError("optimizer.resume must be true or false")
    optimizer["saturation_area_ratio_max"] = _finite_float(
        optimizer.get("saturation_area_ratio_max"),
        "optimizer.saturation_area_ratio_max",
    )
    if optimizer["saturation_area_ratio_max"] < constraints["max_area_ratio"]:
        raise ConfigError("optimizer.saturation_area_ratio_max cannot be below max_area_ratio")

    latency_mapping = cfg.get("latency_mapping")
    if not isinstance(latency_mapping, dict):
        raise ConfigError("latency_mapping must be an object")
    if latency_mapping.get("mode") != "relative_delta":
        raise ConfigError("latency_mapping.mode must be 'relative_delta'")
    if latency_mapping.get("rounding") != "ceil_away_from_zero":
        raise ConfigError(
            "latency_mapping.rounding must be 'ceil_away_from_zero'"
        )
    latency_mapping["minimum_cycles"] = _positive_int(
        latency_mapping.get("minimum_cycles"), "latency_mapping.minimum_cycles"
    )

    cfg["applications"] = _resolve_applications(cfg, path.parent)
    if not cfg["applications"]:
        raise ConfigError(
            "No application traces found. Add applications or make an application_trace_glob match kernelslist.g files."
        )
    return cfg


def _resolve_applications(cfg: dict[str, Any], base: Path) -> list[dict[str, Any]]:
    apps_raw = cfg.get("applications", [])
    if apps_raw is None:
        apps_raw = []
    if not isinstance(apps_raw, list):
        raise ConfigError("applications must be a list")
    apps: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_traces: set[str] = set()

    for index, raw in enumerate(apps_raw):
        if not isinstance(raw, dict):
            raise ConfigError(f"applications[{index}] must be an object")
        name = str(_require(raw, "name", f"applications[{index}]")).strip()
        trace = _resolve_path(str(_require(raw, "trace", f"applications[{index}]")), base)
        if not name:
            raise ConfigError(f"applications[{index}].name cannot be empty")
        if name in seen_names:
            raise ConfigError(f"Duplicate application name: {name}")
        if trace in seen_traces:
            raise ConfigError(f"Duplicate application trace: {trace}")
        weight = _finite_float(raw.get("weight", 1.0), f"applications[{index}].weight")
        if weight <= 0:
            raise ConfigError(f"applications[{index}].weight must be > 0")
        apps.append({"name": name, "trace": trace, "weight": weight})
        seen_names.add(name)
        seen_traces.add(trace)

    globs_raw = cfg.get("application_trace_globs", [])
    if isinstance(globs_raw, str):
        globs_raw = [globs_raw]
    if not isinstance(globs_raw, list):
        raise ConfigError("application_trace_globs must be a string or list")
    for raw_pattern in globs_raw:
        pattern_path = Path(str(raw_pattern)).expanduser()
        if not pattern_path.is_absolute():
            pattern_path = base / pattern_path
        for match in sorted(glob.glob(str(pattern_path), recursive=True)):
            trace = str(Path(match).resolve())
            if trace in seen_traces:
                continue
            parent = Path(trace).parent
            name = parent.parent.name if parent.name == "traces" else parent.name
            base_name = name or f"application_{len(apps) + 1}"
            name = base_name
            suffix = 2
            while name in seen_names:
                name = f"{base_name}_{suffix}"
                suffix += 1
            apps.append({"name": name, "trace": trace, "weight": 1.0})
            seen_names.add(name)
            seen_traces.add(trace)
    return apps


def validate_paths(cfg: dict[str, Any]) -> list[str]:
    """Return all filesystem validation errors instead of stopping at the first."""
    errors: list[str] = []
    file_keys = ("nsc_binary", "accelsim_binary", "accelsim_config", "trace_config")
    for key in file_keys:
        if not Path(cfg["paths"][key]).is_file():
            errors.append(f"paths.{key} does not exist as a file: {cfg['paths'][key]}")
    for key in ("nsc_binary", "accelsim_binary"):
        path = Path(cfg["paths"][key])
        if path.is_file() and not os.access(str(path), os.X_OK):
            errors.append(f"paths.{key} is not executable: {path}")
    for key in ("nscache_root", "accelsim_root"):
        if not Path(cfg["paths"][key]).is_dir():
            errors.append(f"paths.{key} does not exist as a directory: {cfg['paths'][key]}")
    for key in ("l1_cfg", "l2_cfg"):
        if not Path(cfg["baseline"][key]).is_file():
            errors.append(f"baseline.{key} does not exist: {cfg['baseline'][key]}")
    for name, spec in cfg["technologies"].items():
        for key in ("l1_template", "l2_template", "cell_template"):
            if spec.get(key) is None:
                continue
            if not Path(spec[key]).is_file():
                errors.append(f"technologies.{name}.{key} does not exist: {spec[key]}")
    for app in cfg["applications"]:
        path = Path(app["trace"])
        if not path.is_file():
            errors.append(f"application '{app['name']}' trace does not exist: {path}")
        elif path.name != "kernelslist.g":
            errors.append(f"application '{app['name']}' trace should be a kernelslist.g file: {path}")
    return errors
