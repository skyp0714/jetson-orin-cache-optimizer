"""Accel-Sim adapter for the Jetson Orin cache optimization pipeline.

The adapter has three deliberately separate responsibilities:

* parse a completed (or failed) trace-driven simulation log;
* render cache candidates without mutating the calibrated Orin template; and
* run one candidate in its own working directory without sourcing a shell.

Cache access accounting follows the aggregate counters used by AccelWattch's
``power_stat.h``.  In particular, reservation failures are retries and are not
charged as cache-array accesses.
"""

from __future__ import annotations

import dataclasses
import math
import os
import re
import shlex
import signal
import subprocess
import time
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence, Union

from .models import AccessCounts


PathLike = Union[str, os.PathLike]


class AccelSimError(RuntimeError):
    """Base class for Accel-Sim adapter failures."""


class AccelSimParseError(AccelSimError):
    """A simulator log is incomplete or internally inconsistent."""


class AccelSimConfigError(AccelSimError):
    """An Orin configuration or requested mapping is unsupported."""


class AccelSimRunError(AccelSimError):
    """An Accel-Sim subprocess did not complete successfully."""


@dataclasses.dataclass(frozen=True)
class CacheString:
    """The geometry and untouched policy suffix of a GPGPU-Sim cache string."""

    cache_type: str
    sets: int
    line_bytes: int
    associativity: int
    policy_suffix: str

    def __post_init__(self) -> None:
        if self.cache_type not in {"N", "S"}:
            raise AccelSimConfigError("cache type must be N or S")
        for name in ("sets", "line_bytes", "associativity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise AccelSimConfigError(f"cache {name} must be a positive integer")
        if not self.policy_suffix.startswith(","):
            raise AccelSimConfigError("cache policy suffix must begin with ','")

    @property
    def capacity_bytes(self) -> int:
        return self.sets * self.line_bytes * self.associativity

    @property
    def capacity_kib(self) -> Fraction:
        return Fraction(self.capacity_bytes, 1024)

    @property
    def set_index_function(self) -> str:
        policy_group = self.policy_suffix[1:].split(",", 1)[0]
        fields = policy_group.split(":")
        if len(fields) != 5 or len(fields[-1]) != 1:
            raise AccelSimConfigError(
                f"malformed cache policy group in {self.render()!r}"
            )
        return fields[-1]

    def with_geometry(self, *, sets: int, associativity: int) -> "CacheString":
        return dataclasses.replace(self, sets=sets, associativity=associativity)

    def render(self) -> str:
        return (
            f"{self.cache_type}:{self.sets}:{self.line_bytes}:"
            f"{self.associativity}{self.policy_suffix}"
        )


_CACHE_HEAD_RE = re.compile(
    r"^(?P<kind>[NS]):(?P<sets>\d+):(?P<line>\d+):(?P<assoc>\d+)(?P<suffix>,.+)$"
)


def parse_cache_string(value: str) -> CacheString:
    """Parse a cache string while retaining every non-geometry character."""

    match = _CACHE_HEAD_RE.fullmatch(value.strip())
    if match is None:
        raise AccelSimConfigError(f"malformed GPGPU-Sim cache string: {value!r}")
    return CacheString(
        cache_type=match.group("kind"),
        sets=int(match.group("sets")),
        line_bytes=int(match.group("line")),
        associativity=int(match.group("assoc")),
        policy_suffix=match.group("suffix"),
    )


@dataclasses.dataclass(frozen=True)
class ClockDomains:
    """GPGPU-Sim clock domains, in MHz and in config-file order."""

    core_mhz: float
    interconnect_mhz: float
    l2_mhz: float
    dram_mhz: float

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = float(getattr(self, field.name))
            if not math.isfinite(value) or value <= 0:
                raise AccelSimConfigError(f"{field.name} must be finite and > 0")
            object.__setattr__(self, field.name, value)

    @property
    def core_hz(self) -> float:
        return self.core_mhz * 1_000_000.0

    @property
    def l2_hz(self) -> float:
        return self.l2_mhz * 1_000_000.0

    def render(self) -> str:
        return ":".join(
            format(value, "g")
            for value in (
                self.core_mhz,
                self.interconnect_mhz,
                self.l2_mhz,
                self.dram_mhz,
            )
        )


def parse_clock_domains(value: str) -> ClockDomains:
    fields = value.strip().split(":")
    if len(fields) != 4:
        raise AccelSimConfigError(
            "-gpgpu_clock_domains must contain core:icnt:l2:dram"
        )
    try:
        return ClockDomains(*(float(field) for field in fields))
    except ValueError as exc:
        raise AccelSimConfigError(f"malformed clock domains: {value!r}") from exc


@dataclasses.dataclass(frozen=True)
class _DirectiveLine:
    indent: str
    key: str
    spacing: str
    value: str
    comment: str
    newline: str

    def render(self, value: str | None = None) -> str:
        replacement = self.value if value is None else value
        return (
            f"{self.indent}{self.key}{self.spacing}{replacement}"
            f"{self.comment}{self.newline}"
        )


_DIRECTIVE_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<key>-\S+)(?P<spacing>[ \t]+)(?P<rest>.*)$"
)
_INLINE_COMMENT_RE = re.compile(r"^(?P<value>.*?)(?P<comment>[ \t]+#.*)?$")


def _split_newline(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n") or line.endswith("\r"):
        return line[:-1], line[-1]
    return line, ""


def _parse_directive_line(line: str) -> _DirectiveLine | None:
    body, newline = _split_newline(line)
    match = _DIRECTIVE_RE.match(body)
    if match is None:
        return None
    rest = _INLINE_COMMENT_RE.fullmatch(match.group("rest"))
    assert rest is not None
    return _DirectiveLine(
        indent=match.group("indent"),
        key=match.group("key"),
        spacing=match.group("spacing"),
        value=rest.group("value").strip(),
        comment=rest.group("comment") or "",
        newline=newline,
    )


def parse_directives(config_text: str) -> dict[str, str]:
    """Return active directives, rejecting ambiguous duplicate definitions."""

    result: dict[str, str] = {}
    for raw_line in config_text.splitlines(keepends=True):
        directive = _parse_directive_line(raw_line)
        if directive is None:
            continue
        if directive.key in result:
            raise AccelSimConfigError(
                f"multiple active {directive.key} directives in configuration"
            )
        result[directive.key] = directive.value
    return result


def patch_directives(config_text: str, updates: Mapping[str, object]) -> str:
    """Patch active directives in place, preserving comments and line endings."""

    requested = {str(key): str(value) for key, value in updates.items()}
    found: set[str] = set()
    rendered: list[str] = []
    for raw_line in config_text.splitlines(keepends=True):
        directive = _parse_directive_line(raw_line)
        if directive is None or directive.key not in requested:
            rendered.append(raw_line)
            continue
        if directive.key in found:
            raise AccelSimConfigError(
                f"multiple active {directive.key} directives in configuration"
            )
        found.add(directive.key)
        rendered.append(directive.render(requested[directive.key]))
    missing = sorted(set(requested) - found)
    if missing:
        raise AccelSimConfigError(
            "missing active directive(s): {}".format(", ".join(missing))
        )
    return "".join(rendered)


def _parse_positive_int(value: str, directive: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise AccelSimConfigError(f"{directive} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise AccelSimConfigError(f"{directive} must be > 0")
    return parsed


@dataclasses.dataclass(frozen=True)
class OrinConfig:
    """A parsed view over a calibrated Orin ``gpgpusim.config`` template."""

    text: str
    directives: Mapping[str, str]

    def get(self, name: str) -> str:
        try:
            return self.directives[name]
        except KeyError as exc:
            raise AccelSimConfigError(f"missing active {name} directive") from exc

    @property
    def l1(self) -> CacheString:
        return parse_cache_string(self.get("-gpgpu_cache:dl1"))

    @property
    def l2(self) -> CacheString:
        return parse_cache_string(self.get("-gpgpu_cache:dl2"))

    @property
    def clocks(self) -> ClockDomains:
        return parse_clock_domains(self.get("-gpgpu_clock_domains"))

    @property
    def l1_latency_cycles(self) -> int:
        return _parse_positive_int(self.get("-gpgpu_l1_latency"), "-gpgpu_l1_latency")

    @property
    def l2_latency_cycles(self) -> int:
        return _parse_positive_int(
            self.get("-gpgpu_l2_rop_latency"), "-gpgpu_l2_rop_latency"
        )

    @property
    def adaptive_l1(self) -> bool:
        raw = self.get("-gpgpu_adaptive_cache_config")
        if raw not in {"0", "1"}:
            raise AccelSimConfigError("-gpgpu_adaptive_cache_config must be 0 or 1")
        return raw == "1"

    @property
    def unified_l1d_size_kib(self) -> int:
        return _parse_positive_int(
            self.get("-gpgpu_unified_l1d_size"), "-gpgpu_unified_l1d_size"
        )

    @property
    def shmem_options_kib(self) -> tuple[int, ...]:
        raw = self.get("-gpgpu_shmem_option")
        try:
            values = tuple(int(value.strip()) for value in raw.split(","))
        except ValueError as exc:
            raise AccelSimConfigError(f"malformed -gpgpu_shmem_option: {raw!r}") from exc
        if not values or min(values) < 0:
            raise AccelSimConfigError("shared-memory options must be non-negative")
        return values

    @property
    def shmem_size_bytes(self) -> int:
        return _parse_positive_int(
            self.get("-gpgpu_shmem_size"), "-gpgpu_shmem_size"
        )

    @property
    def shmem_size_default_bytes(self) -> int:
        return _parse_positive_int(
            self.get("-gpgpu_shmem_sizeDefault"), "-gpgpu_shmem_sizeDefault"
        )

    @property
    def l1_instances(self) -> int:
        return _parse_positive_int(
            self.get("-gpgpu_n_clusters"), "-gpgpu_n_clusters"
        ) * _parse_positive_int(
            self.get("-gpgpu_n_cores_per_cluster"), "-gpgpu_n_cores_per_cluster"
        )

    @property
    def l2_instances(self) -> int:
        return _parse_positive_int(self.get("-gpgpu_n_mem"), "-gpgpu_n_mem") * _parse_positive_int(
            self.get("-gpgpu_n_sub_partition_per_mchannel"),
            "-gpgpu_n_sub_partition_per_mchannel",
        )

    @property
    def l2_total_capacity_kib(self) -> Fraction:
        return self.l2.capacity_kib * self.l2_instances

    def patched(self, updates: Mapping[str, object]) -> str:
        return patch_directives(self.text, updates)


_REQUIRED_ORIN_DIRECTIVES = (
    "-gpgpu_cache:dl1",
    "-gpgpu_cache:dl2",
    "-gpgpu_clock_domains",
    "-gpgpu_l1_latency",
    "-gpgpu_l2_rop_latency",
    "-gpgpu_adaptive_cache_config",
    "-gpgpu_n_clusters",
    "-gpgpu_n_cores_per_cluster",
    "-gpgpu_n_mem",
    "-gpgpu_n_sub_partition_per_mchannel",
)


def parse_config_text(config_text: str) -> OrinConfig:
    directives = parse_directives(config_text)
    missing = [name for name in _REQUIRED_ORIN_DIRECTIVES if name not in directives]
    if missing:
        raise AccelSimConfigError(
            "missing required Orin directive(s): {}".format(", ".join(missing))
        )
    config = OrinConfig(config_text, directives)
    # Eager validation gives a candidate-local failure before an expensive run.
    config.l1
    config.l2
    config.clocks
    config.l1_latency_cycles
    config.l2_latency_cycles
    config.l1_instances
    config.l2_instances
    if config.adaptive_l1:
        config.unified_l1d_size_kib
        config.shmem_options_kib
        config.shmem_size_bytes
        config.shmem_size_default_bytes
    return config


def read_config(path: PathLike) -> OrinConfig:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise AccelSimConfigError(f"cannot read Accel-Sim config {path!s}: {exc}") from exc
    return parse_config_text(text)



def _positive_architecture_int(value: int, label: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise AccelSimConfigError(f"{label} must be a positive integer")
    return int(value)


def _integral(value: Fraction, label: str) -> int:
    if value.denominator != 1:
        raise AccelSimConfigError(
            f"{label} maps to non-integral simulator geometry ({value})"
        )
    if value.numerator <= 0:
        raise AccelSimConfigError(f"{label} must map to a positive integer")
    return value.numerator


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


_IPOLY_SET_COUNTS = frozenset({4, 8, 16, 32, 64, 128, 256})


def _validate_index_mapping(cache: CacheString, label: str) -> None:
    if not _is_power_of_two(cache.sets):
        raise AccelSimConfigError(
            f"{label} maps to {cache.sets} sets; GPGPU-Sim cache sets must be a power of two"
        )
    indexing = cache.set_index_function
    if indexing == "P" and cache.sets not in _IPOLY_SET_COUNTS:
        supported = ", ".join(str(value) for value in sorted(_IPOLY_SET_COUNTS))
        raise AccelSimConfigError(
            f"{label} preserves IPOLY indexing but {cache.sets} sets are unsupported "
            f"(supported: {supported})"
        )
    if indexing == "H" and cache.sets not in {32, 64}:
        raise AccelSimConfigError(
            f"{label} preserves Fermi hashing, which supports only 32 or 64 sets"
        )


def map_relative_cache_geometry(
    cache: CacheString,
    *,
    baseline_capacity_kib: int,
    baseline_associativity: int,
    capacity_kib: int,
    associativity: int,
    label: str = "cache",
) -> CacheString:
    """Map architecture-level capacity/ways onto a calibrated cache string.

    Both dimensions are relative to the architecture baseline.  Thus a target
    with twice the capacity and unchanged associativity doubles the number of
    simulator sets, while a target with unchanged capacity and twice the ways
    doubles simulator ways and halves its sets.
    """

    baseline_capacity_kib = _positive_architecture_int(
        baseline_capacity_kib, f"baseline {label} capacity"
    )
    baseline_associativity = _positive_architecture_int(
        baseline_associativity, f"baseline {label} associativity"
    )
    capacity_kib = _positive_architecture_int(capacity_kib, f"target {label} capacity")
    associativity = _positive_architecture_int(
        associativity, f"target {label} associativity"
    )
    capacity_ratio = Fraction(capacity_kib, baseline_capacity_kib)
    associativity_ratio = Fraction(associativity, baseline_associativity)
    mapped = cache.with_geometry(
        sets=_integral(
            Fraction(cache.sets) * capacity_ratio / associativity_ratio,
            f"{label} set count",
        ),
        associativity=_integral(
            Fraction(cache.associativity) * associativity_ratio,
            f"{label} associativity",
        ),
    )
    _validate_index_mapping(mapped, label)
    return mapped


def map_relative_delta_latency(
    *,
    baseline_cycles: int,
    baseline_latency_ns: float,
    candidate_latency_ns: float,
    clock_mhz: float,
    minimum_cycles: int = 1,
) -> int:
    """Preserve calibrated pipeline delay and map only the NS-Cache delta.

    Deltas are rounded away from zero.  This avoids systematically hiding a
    sub-cycle slowdown and treats an equally sized speedup symmetrically.
    """

    baseline_cycles = _positive_architecture_int(baseline_cycles, "baseline cycles")
    minimum_cycles = _positive_architecture_int(minimum_cycles, "minimum cycles")
    values = {
        "baseline_latency_ns": baseline_latency_ns,
        "candidate_latency_ns": candidate_latency_ns,
        "clock_mhz": clock_mhz,
    }
    for label, raw in values.items():
        value = float(raw)
        if not math.isfinite(value) or (label == "clock_mhz" and value <= 0) or (
            label != "clock_mhz" and value < 0
        ):
            raise AccelSimConfigError(f"{label} has invalid value {raw!r}")
    delta_cycles = (float(candidate_latency_ns) - float(baseline_latency_ns)) * float(
        clock_mhz
    ) / 1000.0
    rounded_delta = math.ceil(delta_cycles) if delta_cycles >= 0 else math.floor(delta_cycles)
    return max(minimum_cycles, baseline_cycles + rounded_delta)



def _latency_pair(
    baseline_ns: float | None,
    candidate_ns: float | None,
    label: str,
) -> tuple[float, float] | None:
    if baseline_ns is None and candidate_ns is None:
        return None
    if baseline_ns is None or candidate_ns is None:
        raise AccelSimConfigError(
            f"{label} baseline and candidate latency must be provided together"
        )
    return float(baseline_ns), float(candidate_ns)


def generate_candidate_config(
    baseline_config_text: str,
    *,
    baseline_l1_capacity_kib: int,
    baseline_l1_associativity: int,
    baseline_l2_capacity_kib: int,
    baseline_l2_associativity: int,
    l1_capacity_kib: int,
    l1_associativity: int,
    l2_capacity_kib: int,
    l2_associativity: int,
    baseline_l1_latency_ns: float | None = None,
    l1_latency_ns: float | None = None,
    baseline_l2_latency_ns: float | None = None,
    l2_latency_ns: float | None = None,
    minimum_latency_cycles: int = 1,
    extra_directives: Mapping[str, object] | None = None,
) -> str:
    """Render one architecture candidate from the immutable Orin baseline."""

    config = parse_config_text(baseline_config_text)
    mapped_l1 = map_relative_cache_geometry(
        config.l1,
        baseline_capacity_kib=baseline_l1_capacity_kib,
        baseline_associativity=baseline_l1_associativity,
        capacity_kib=l1_capacity_kib,
        associativity=l1_associativity,
        label="L1",
    )
    mapped_l2 = map_relative_cache_geometry(
        config.l2,
        baseline_capacity_kib=baseline_l2_capacity_kib,
        baseline_associativity=baseline_l2_associativity,
        capacity_kib=l2_capacity_kib,
        associativity=l2_associativity,
        label="L2",
    )
    updates: dict[str, object] = {
        "-gpgpu_cache:dl1": mapped_l1.render(),
        "-gpgpu_cache:dl2": mapped_l2.render(),
    }

    if config.adaptive_l1:
        # Scale the unified pool with the requested cache capacity, but retain
        # Orin's shared-memory options and size. This changes cache capacity
        # without changing the shared-memory/CTA-occupancy limits. Candidates
        # smaller than the fixed shared-memory allocation are therefore invalid.
        ratio = Fraction(
            _positive_architecture_int(l1_capacity_kib, "target L1 capacity"),
            _positive_architecture_int(
                baseline_l1_capacity_kib, "baseline L1 capacity"
            ),
        )
        unified_kib = _integral(
            Fraction(config.unified_l1d_size_kib) * ratio,
            "adaptive unified L1/shared capacity",
        )
        nominal_kib = mapped_l1.capacity_kib
        if nominal_kib.denominator != 1 or unified_kib % nominal_kib.numerator:
            raise AccelSimConfigError(
                "adaptive L1 mapping requires unified size to be an integral "
                "multiple of the base dl1 geometry"
            )
        fixed_shared_kib = max(
            max(config.shmem_options_kib),
            math.ceil(config.shmem_size_bytes / 1024),
            math.ceil(config.shmem_size_default_bytes / 1024),
        )
        if unified_kib < fixed_shared_kib:
            raise AccelSimConfigError(
                "scaled unified L1/shared pool is {} KiB, below Orin's fixed "
                "{} KiB shared-memory limit".format(unified_kib, fixed_shared_kib)
            )
        updates["-gpgpu_unified_l1d_size"] = unified_kib

    l1_pair = _latency_pair(baseline_l1_latency_ns, l1_latency_ns, "L1")
    if l1_pair is not None:
        updates["-gpgpu_l1_latency"] = map_relative_delta_latency(
            baseline_cycles=config.l1_latency_cycles,
            baseline_latency_ns=l1_pair[0],
            candidate_latency_ns=l1_pair[1],
            clock_mhz=config.clocks.core_mhz,
            minimum_cycles=minimum_latency_cycles,
        )
    l2_pair = _latency_pair(baseline_l2_latency_ns, l2_latency_ns, "L2")
    if l2_pair is not None:
        updates["-gpgpu_l2_rop_latency"] = map_relative_delta_latency(
            baseline_cycles=config.l2_latency_cycles,
            baseline_latency_ns=l2_pair[0],
            candidate_latency_ns=l2_pair[1],
            # gpgpu_l2_rop_latency is added to gpu_sim_cycle in l2cache.cc,
            # hence it is expressed in the core-cycle domain.
            clock_mhz=config.clocks.core_mhz,
            minimum_cycles=minimum_latency_cycles,
        )

    if extra_directives:
        conflicts = sorted(set(updates) & set(extra_directives))
        if conflicts:
            raise AccelSimConfigError(
                "extra_directives cannot override generated fields: {}".format(
                    ", ".join(conflicts)
                )
            )
        updates.update(extra_directives)
    return config.patched(updates)



_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_CYCLE_RE = re.compile(r"^\s*gpu_tot_sim_cycle\s*=\s*(\d+)\s*$", re.MULTILINE)
_INSN_RE = re.compile(r"^\s*gpu_tot_sim_insn\s*=\s*(\d+)\s*$", re.MULTILINE)
_IPC_RE = re.compile(rf"^\s*gpu_tot_ipc\s*=\s*({_NUMBER})\s*$", re.MULTILINE)
_STREAM_RE = re.compile(r"^\s*kernel_stream_id\s*=\s*(\d+)\s*$")
_BREAKDOWN_RE = re.compile(
    r"^\s*(?P<level>Total_core_cache_stats_breakdown|L2_cache_stats_breakdown)"
    r"\[(?P<access>[A-Z0-9_]+)\]\[(?P<status>[A-Z0-9_]+)\]\s*=\s*(?P<value>\d+)\s*$"
)

_EXIT_MARKER = "GPGPU-Sim: *** exit detected ***"
_LIMIT_PATTERNS = (
    re.compile(r"break due to reaching the maximum cycles", re.IGNORECASE),
    re.compile(r"Maximum cycle, instruction, or CTA count hit", re.IGNORECASE),
)
_ERROR_PATTERNS = (
    re.compile(r"\bERROR(?:\s*\*\*|:)", re.IGNORECASE),
    re.compile(r"deadlock detected", re.IGNORECASE),
    re.compile(r"\b(?:fatal error|segmentation fault|core dumped)\b", re.IGNORECASE),
    re.compile(r"\bassertion\b.*\bfailed\b", re.IGNORECASE),
    re.compile(r"\bassert(?: failed| failure|:)\b", re.IGNORECASE),
    re.compile(r"undefined symbol", re.IGNORECASE),
    re.compile(r"terminate called after throwing", re.IGNORECASE),
    re.compile(r"\bstd::bad_alloc\b", re.IGNORECASE),
    re.compile(r"^\s*(?:aborted|killed)(?:\s|$)", re.IGNORECASE),
)


@dataclasses.dataclass(frozen=True)
class AccelSimOutput:
    """Final cumulative metrics plus termination diagnostics from one log."""

    cycles: int | None
    instructions: int | None
    ipc: float | None
    accesses: AccessCounts
    completed: bool
    reached_limit: bool
    error_messages: tuple[str, ...]
    missing_metrics: tuple[str, ...]

    @property
    def success(self) -> bool:
        return (
            self.completed
            and not self.reached_limit
            and not self.error_messages
            and not self.missing_metrics
        )

    def runtime_s(self, core_clock_mhz: float) -> float:
        if self.cycles is None:
            raise AccelSimParseError("gpu_tot_sim_cycle is unavailable")
        if not math.isfinite(core_clock_mhz) or core_clock_mhz <= 0:
            raise ValueError("core_clock_mhz must be finite and > 0")
        return self.cycles / (core_clock_mhz * 1_000_000.0)

    def require_success(self) -> "AccelSimOutput":
        if self.success:
            return self
        reasons: list[str] = []
        if not self.completed:
            reasons.append("missing Accel-Sim exit marker")
        if self.reached_limit:
            reasons.append("maximum simulation limit reached")
        reasons.extend(self.error_messages)
        if self.missing_metrics:
            reasons.append("missing metrics: " + ", ".join(self.missing_metrics))
        raise AccelSimParseError("; ".join(reasons) or "Accel-Sim output is incomplete")


def _last_int(pattern: re.Pattern[str], text: str) -> int | None:
    matches = pattern.findall(text)
    return int(matches[-1]) if matches else None


def _last_float(pattern: re.Pattern[str], text: str) -> float | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    value = float(matches[-1])
    return value if math.isfinite(value) else None


def _collect_breakdowns(
    text: str,
) -> dict[str, dict[int, dict[tuple[str, str], int]]]:
    """Keep the final cumulative snapshot for every CUDA stream."""

    current_stream = -1
    values: dict[str, dict[int, dict[tuple[str, str], int]]] = {
        "l1": {},
        "l2": {},
    }
    for line in text.splitlines():
        stream = _STREAM_RE.match(line)
        if stream is not None:
            current_stream = int(stream.group(1))
            continue
        breakdown = _BREAKDOWN_RE.match(line)
        if breakdown is None:
            continue
        level = "l1" if breakdown.group("level").startswith("Total_core") else "l2"
        stream_values = values[level].setdefault(current_stream, {})
        stream_values[(breakdown.group("access"), breakdown.group("status"))] = int(
            breakdown.group("value")
        )
    return values


def _sum_stats(
    streams: Mapping[int, Mapping[tuple[str, str], int]],
    access_types: Sequence[str],
    statuses: Sequence[str],
) -> int:
    return sum(
        counters.get((access_type, status), 0)
        for counters in streams.values()
        for access_type in access_types
        for status in statuses
    )


def _access_counts(
    breakdowns: Mapping[str, Mapping[int, Mapping[tuple[str, str], int]]]
) -> tuple[AccessCounts, tuple[str, ...]]:
    missing: list[str] = []
    l1 = breakdowns["l1"]
    l2 = breakdowns["l2"]
    if not l1:
        missing.append("L1 cache breakdown")
    if not l2:
        missing.append("L2 cache breakdown")

    l1_reads = ("GLOBAL_ACC_R", "LOCAL_ACC_R")
    l1_writes = ("GLOBAL_ACC_W", "LOCAL_ACC_W")
    l1_read_access = _sum_stats(l1, l1_reads, ("HIT", "MISS", "SECTOR_MISS"))
    l1_read_hit = _sum_stats(l1, l1_reads, ("HIT", "MSHR_HIT"))
    if l1_read_hit > l1_read_access:
        missing.append("consistent L1 read counters")
        l1_read_miss = 0
    else:
        # This is exactly get_l1d_read_misses() in AccelWattch power_stat.h.
        l1_read_miss = l1_read_access - l1_read_hit
    l1_write = _sum_stats(l1, l1_writes, ("HIT", "MISS", "SECTOR_MISS"))

    l2_reads = (
        "GLOBAL_ACC_R",
        "LOCAL_ACC_R",
        "CONST_ACC_R",
        "TEXTURE_ACC_R",
        "INST_ACC_R",
    )
    l2_writes = ("GLOBAL_ACC_W", "LOCAL_ACC_W", "L1_WRBK_ACC")
    l2_read_hit = _sum_stats(l2, l2_reads, ("HIT", "HIT_RESERVED"))
    l2_read_miss = _sum_stats(l2, l2_reads, ("MISS", "SECTOR_MISS"))
    l2_write = _sum_stats(
        l2, l2_writes, ("HIT", "HIT_RESERVED", "MISS", "SECTOR_MISS")
    )

    accesses = AccessCounts(
        l1_read_hit=l1_read_hit,
        l1_read_miss=l1_read_miss,
        l1_write=l1_write,
        l2_read_hit=l2_read_hit,
        l2_read_miss=l2_read_miss,
        l2_write=l2_write,
    )
    accesses.validate()
    return accesses, tuple(missing)


def _diagnostic_lines(text: str) -> tuple[str, ...]:
    result: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line and any(pattern.search(line) for pattern in _ERROR_PATTERNS):
            if line not in result:
                result.append(line)
    return tuple(result)


def parse_output(text: str) -> AccelSimOutput:
    """Parse the last cumulative metrics and all stream-local cache counters."""

    cycles = _last_int(_CYCLE_RE, text)
    instructions = _last_int(_INSN_RE, text)
    ipc = _last_float(_IPC_RE, text)
    if ipc is None and cycles not in {None, 0} and instructions is not None:
        ipc = instructions / cycles

    accesses, cache_missing = _access_counts(_collect_breakdowns(text))
    missing = list(cache_missing)
    if cycles is None:
        missing.append("gpu_tot_sim_cycle")
    if instructions is None:
        missing.append("gpu_tot_sim_insn")
    if ipc is None:
        missing.append("gpu_tot_ipc")

    return AccelSimOutput(
        cycles=cycles,
        instructions=instructions,
        ipc=ipc,
        accesses=accesses,
        completed=_EXIT_MARKER in text,
        reached_limit=any(pattern.search(text) for pattern in _LIMIT_PATTERNS),
        error_messages=_diagnostic_lines(text),
        missing_metrics=tuple(missing),
    )



@dataclasses.dataclass(frozen=True)
class AccelSimRunResult:
    command: tuple[str, ...]
    work_dir: Path
    stdout_log: Path
    returncode: int | None
    elapsed_s: float
    timed_out: bool
    output: AccelSimOutput

    @property
    def success(self) -> bool:
        return not self.timed_out and self.returncode == 0 and self.output.success

    def raise_for_status(self) -> "AccelSimRunResult":
        if self.success:
            return self
        if self.timed_out:
            reason = "timed out"
        elif self.output.reached_limit:
            reason = "stopped after reaching the maximum simulation limit"
            if self.returncode not in {None, 0}:
                reason += f" (exit status {self.returncode})"
        elif self.output.error_messages:
            reason = "reported an error: " + " | ".join(self.output.error_messages[:3])
            if self.returncode not in {None, 0}:
                reason += f" (exit status {self.returncode})"
        elif self.returncode != 0:
            reason = f"exited with status {self.returncode}"
        else:
            try:
                self.output.require_success()
            except AccelSimParseError as exc:
                reason = str(exc)
            else:  # pragma: no cover - guarded by success property
                reason = "unknown simulation failure"
        raise AccelSimRunError(
            f"Accel-Sim {reason}. Log: {self.stdout_log}. "
            f"Command: {shlex.join(self.command)}"
        )


def _resolved_input(path: PathLike, label: str, *, executable: bool = False) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise AccelSimRunError(f"{label} does not exist as a file: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise AccelSimRunError(f"{label} is not executable: {resolved}")
    return resolved


def run_accelsim(
    binary: PathLike,
    trace: PathLike,
    gpu_config: PathLike,
    trace_config: PathLike,
    *,
    work_dir: PathLike,
    environment: Mapping[str, str],
    timeout_s: float,
    extra_args: Sequence[str] = (),
    log_name: str = "accelsim.stdout.log",
    overwrite_log: bool = False,
    check: bool = True,
) -> AccelSimRunResult:
    """Run one trace in an isolated cwd with no shell or inherited env magic."""

    if environment is None:
        raise AccelSimRunError("environment must be supplied explicitly")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise AccelSimRunError("timeout_s must be finite and > 0")
    binary_path = _resolved_input(binary, "Accel-Sim binary", executable=True)
    trace_path = _resolved_input(trace, "kernels list")
    gpu_config_path = _resolved_input(gpu_config, "GPGPU-Sim config")
    trace_config_path = _resolved_input(trace_config, "trace config")

    run_directory = Path(work_dir).expanduser().resolve()
    run_directory.mkdir(parents=True, exist_ok=True)
    if not run_directory.is_dir():
        raise AccelSimRunError(f"work_dir is not a directory: {run_directory}")
    if Path(log_name).name != log_name:
        raise AccelSimRunError("log_name must be a plain file name")
    stdout_log = run_directory / log_name
    mode = "wb" if overwrite_log else "xb"

    command = (
        str(binary_path),
        "-trace",
        str(trace_path),
        "-config",
        str(gpu_config_path),
        "-config",
        str(trace_config_path),
        *(str(argument) for argument in extra_args),
    )
    explicit_environment = {str(key): str(value) for key, value in environment.items()}
    started = time.monotonic()
    timed_out = False
    returncode: int | None = None
    try:
        with stdout_log.open(mode) as output_file:
            process = subprocess.Popen(
                command,
                cwd=run_directory,
                env=explicit_environment,
                stdout=output_file,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=float(timeout_s))
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    returncode = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    returncode = process.wait()
            except KeyboardInterrupt:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                raise
    except FileExistsError as exc:
        raise AccelSimRunError(
            f"stdout log already exists (use a fresh work_dir or overwrite_log): {stdout_log}"
        ) from exc
    except OSError as exc:
        raise AccelSimRunError(f"could not execute Accel-Sim: {exc}") from exc

    elapsed = time.monotonic() - started
    try:
        text = stdout_log.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise AccelSimRunError(f"cannot read simulator log {stdout_log}: {exc}") from exc
    result = AccelSimRunResult(
        command=command,
        work_dir=run_directory,
        stdout_log=stdout_log,
        returncode=returncode,
        elapsed_s=elapsed,
        timed_out=timed_out,
        output=parse_output(text),
    )
    if check:
        result.raise_for_status()
    return result


def _prepend_path(value: str, existing: str | None) -> str:
    return value if not existing else value + os.pathsep + existing


def build_accelsim_environment(
    accelsim_root: PathLike,
    *,
    cuda_install_path: PathLike = "/usr/local/cuda",
    gpgpusim_lib_dir: PathLike | None = None,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Construct the setup-script environment without invoking a shell.

    ``accelsim_root`` may be either the framework checkout or its
    ``gpu-simulator`` directory.  Exactly one compiled GPGPU-Sim release must
    be discoverable unless ``gpgpusim_lib_dir`` is supplied explicitly.
    """

    supplied_root = Path(accelsim_root).expanduser().resolve()
    if (supplied_root / "gpu-simulator" / "gpgpu-sim").is_dir():
        simulator_root = supplied_root / "gpu-simulator"
    elif (supplied_root / "gpgpu-sim").is_dir():
        simulator_root = supplied_root
    else:
        raise AccelSimConfigError(
            f"cannot locate gpu-simulator/gpgpu-sim below {supplied_root}"
        )
    gpgpusim_root = simulator_root / "gpgpu-sim"
    cuda_root = Path(cuda_install_path).expanduser().resolve()
    if not (cuda_root / "bin").is_dir():
        raise AccelSimConfigError(f"CUDA bin directory does not exist: {cuda_root / 'bin'}")

    if gpgpusim_lib_dir is None:
        releases = sorted((gpgpusim_root / "lib").glob("gcc-*/cuda-*/release"))
        releases = [path.resolve() for path in releases if path.is_dir()]
        if len(releases) != 1:
            raise AccelSimConfigError(
                f"expected exactly one compiled GPGPU-Sim release below {gpgpusim_root / 'lib'}, "
                f"found {len(releases)}"
            )
        library = releases[0]
    else:
        library = Path(gpgpusim_lib_dir).expanduser().resolve()
        if not library.is_dir():
            raise AccelSimConfigError(f"GPGPU-Sim library directory does not exist: {library}")

    try:
        relative_library = library.relative_to(gpgpusim_root / "lib")
    except ValueError as exc:
        raise AccelSimConfigError(
            f"GPGPU-Sim library must be below {gpgpusim_root / 'lib'}: {library}"
        ) from exc

    env = dict(os.environ if base_environment is None else base_environment)
    env.update(
        {
            "ACCELSIM_ROOT": str(simulator_root),
            "ACCELSIM_CONFIG": "release",
            "ACCELSIM_SETUP_ENVIRONMENT_WAS_RUN": "1",
            "GPGPUSIM_ROOT": str(gpgpusim_root),
            "GPGPUSIM_CONFIG": str(relative_library),
            "GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN": "1",
            "CUDA_INSTALL_PATH": str(cuda_root),
            "PTXAS_CUDA_INSTALL_PATH": str(cuda_root),
        }
    )
    env["PATH"] = _prepend_path(str(cuda_root / "bin"), env.get("PATH"))
    env["LD_LIBRARY_PATH"] = _prepend_path(str(library), env.get("LD_LIBRARY_PATH"))
    build_python = simulator_root / "build" / "release"
    if build_python.is_dir():
        env["PYTHONPATH"] = _prepend_path(str(build_python), env.get("PYTHONPATH"))
    return {str(key): str(value) for key, value in env.items()}


__all__ = [
    "AccelSimConfigError",
    "AccelSimError",
    "AccelSimOutput",
    "AccelSimParseError",
    "AccelSimRunError",
    "AccelSimRunResult",
    "CacheString",
    "ClockDomains",
    "OrinConfig",
    "build_accelsim_environment",
    "generate_candidate_config",
    "map_relative_cache_geometry",
    "map_relative_delta_latency",
    "parse_cache_string",
    "parse_clock_domains",
    "parse_config_text",
    "parse_directives",
    "parse_output",
    "patch_directives",
    "read_config",
    "run_accelsim",
]
