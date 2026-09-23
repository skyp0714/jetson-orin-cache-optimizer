"""Small, dependency-free adapter for invoking and parsing NS-Cache.

The functions in this module deliberately render configuration variants in
memory.  A caller can write the returned text to a run directory without ever
mutating the source template.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union


PathLike = Union[str, os.PathLike]


class NSCacheError(RuntimeError):
    """Base class for adapter failures."""


class NSCacheParseError(NSCacheError):
    """NS-Cache output did not contain a usable summary."""


class NSCacheConfigError(NSCacheError):
    """An NS-Cache configuration is missing or malformed."""


class NSCacheRunError(NSCacheError):
    """NS-Cache could not be executed successfully."""


class PPACacheError(NSCacheError):
    """A JSON PPA cache is malformed or cannot be updated."""


@dataclass(frozen=True)
class CachePPA:
    """Technology-level cache PPA parsed from an NS-Cache summary.

    NS-Cache labels refresh quantities as ``per bank``; the canonical
    ``refresh_*`` fields keep compact names but follow that convention.
    """

    area_mm2: float
    hit_latency_ns: float
    miss_latency_ns: float
    write_latency_ns: float
    hit_energy_nj: float
    miss_energy_nj: float
    write_energy_nj: float
    leakage_power_mw: float
    refresh_latency_us: Optional[float] = None
    refresh_energy_nj: Optional[float] = None
    refresh_power_mw: Optional[float] = None
    availability_percent: Optional[float] = None

    def __post_init__(self) -> None:
        required = (
            "area_mm2",
            "hit_latency_ns",
            "miss_latency_ns",
            "write_latency_ns",
            "hit_energy_nj",
            "miss_energy_nj",
            "write_energy_nj",
            "leakage_power_mw",
        )
        optional = (
            "refresh_latency_us",
            "refresh_energy_nj",
            "refresh_power_mw",
            "availability_percent",
        )
        for name in required:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError("{} must be a finite, non-negative number".format(name))
            object.__setattr__(self, name, value)
        for name in optional:
            raw = getattr(self, name)
            if raw is None:
                continue
            value = float(raw)
            if not math.isfinite(value) or value < 0:
                raise ValueError("{} must be a finite, non-negative number".format(name))
            object.__setattr__(self, name, value)
        if self.availability_percent is not None and self.availability_percent > 100:
            raise ValueError("availability_percent must be between 0 and 100")

    def to_dict(self) -> Dict[str, Optional[float]]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CachePPA":
        if not isinstance(value, Mapping):
            raise ValueError("CachePPA JSON value must be an object")
        names = {field.name for field in dataclasses.fields(cls)}
        unknown = set(value) - names
        if unknown:
            raise ValueError("unknown CachePPA fields: {}".format(", ".join(sorted(unknown))))
        return cls(**dict(value))


@dataclass(frozen=True)
class CacheConfig:
    """Architectural fields read from an NS-Cache config.

    ``force_bank_count`` retains the historical adapter name, but it is A*B
    subarrays inside the single modeled NS-Cache bank, not a count of replicated
    physical cache banks.  It must never be used as a refresh multiplier.
    """

    capacity_kib: int
    associativity: int
    force_bank_count: Optional[int] = None

    def __post_init__(self) -> None:
        if self.capacity_kib <= 0:
            raise ValueError("capacity_kib must be > 0")
        if self.associativity <= 0:
            raise ValueError("associativity must be > 0")
        if self.force_bank_count is not None and self.force_bank_count <= 0:
            raise ValueError("force_bank_count must be > 0")


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_SUMMARY_MARKER_RE = re.compile(
    r"^[ \t]*CACHE[ \t]+DESIGN[ \t]*--[ \t]*SUMMARY[ \t]*$", re.IGNORECASE | re.MULTILINE
)
_SUMMARY_END_RE = re.compile(
    r"^[ \t]*(?:CACHE[ \t]+DATA[ \t]+ARRAY[ \t]+DETAILS|Finished!)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

_AREA_TO_MM2 = {
    "m^2": 1e6,
    "cm^2": 1e2,
    "mm^2": 1.0,
    "um^2": 1e-6,
    "nm^2": 1e-12,
}
_TIME_TO_NS = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0, "ps": 1e-3, "fs": 1e-6}
_ENERGY_TO_NJ = {"j": 1e9, "mj": 1e6, "uj": 1e3, "nj": 1.0, "pj": 1e-3, "fj": 1e-6}
_POWER_TO_MW = {"w": 1e3, "mw": 1.0, "uw": 1e-3, "nw": 1e-6, "pw": 1e-9, "fw": 1e-12}

_AREA_UNIT = r"(?:mm|cm|um|nm|m)[ \t]*(?:\^[ \t]*2|²)"
_TIME_UNIT = r"(?:ms|us|ns|ps|fs|s)"
_ENERGY_UNIT = r"(?:mJ|uJ|nJ|pJ|fJ|J)"
_POWER_UNIT = r"(?:mW|uW|nW|pW|fW|W)"


def _normalise_unit(unit: str) -> str:
    return (
        unit.strip()
        .replace("µ", "u")
        .replace("μ", "u")
        .replace("²", "^2")
        .replace(" ", "")
        .lower()
    )


def _extract_summary(output: str) -> str:
    if not isinstance(output, str):
        raise TypeError("NS-Cache output must be text")
    text = output.replace("\r\n", "\n").replace("\r", "\n")
    markers = list(_SUMMARY_MARKER_RE.finditer(text))
    if not markers:
        raise NSCacheParseError("could not find 'CACHE DESIGN -- SUMMARY' in NS-Cache output")
    start = markers[-1].end()
    tail = text[start:]
    end = _SUMMARY_END_RE.search(tail)
    return tail[: end.start()] if end else tail


def _quantity(
    block: str,
    label_pattern: str,
    unit_pattern: str,
    conversions: Mapping[str, float],
    field_name: str,
    required: bool = True,
) -> Optional[float]:
    pattern = re.compile(
        r"^[ \t]*-[ \t]*"
        + label_pattern
        + r"[ \t]*=[ \t]*(?P<value>"
        + _NUMBER
        + r")[ \t]*(?P<unit>"
        + unit_pattern
        + r")",
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(block)
    if not match:
        if required:
            raise NSCacheParseError("missing '{}' in NS-Cache summary".format(field_name))
        return None
    unit = _normalise_unit(match.group("unit"))
    try:
        factor = conversions[unit]
    except KeyError as exc:
        raise NSCacheParseError(
            "unsupported unit '{}' for '{}'".format(match.group("unit"), field_name)
        ) from exc
    return float(match.group("value")) * factor


def _availability(block: str) -> Optional[float]:
    match = re.search(
        r"^[ \t]*-[ \t]*Cache[ \t]+Availability[ \t]*=[ \t]*(?P<value>"
        + _NUMBER
        + r")[ \t]*%",
        block,
        re.IGNORECASE | re.MULTILINE,
    )
    return float(match.group("value")) if match else None


def parse_summary(output: str) -> CachePPA:
    """Parse the final NS-Cache summary and convert values to canonical units."""

    block = _extract_summary(output)
    refresh_latency_ns = _quantity(
        block,
        r"Cache[ \t]+Refresh[ \t]+Latency",
        _TIME_UNIT,
        _TIME_TO_NS,
        "refresh latency",
        required=False,
    )
    return CachePPA(
        area_mm2=_quantity(block, r"Total[ \t]+Area", _AREA_UNIT, _AREA_TO_MM2, "total area"),
        hit_latency_ns=_quantity(
            block, r"Cache[ \t]+Hit[ \t]+Latency", _TIME_UNIT, _TIME_TO_NS, "cache hit latency"
        ),
        miss_latency_ns=_quantity(
            block, r"Cache[ \t]+Miss[ \t]+Latency", _TIME_UNIT, _TIME_TO_NS, "cache miss latency"
        ),
        write_latency_ns=_quantity(
            block, r"Cache[ \t]+Write[ \t]+Latency", _TIME_UNIT, _TIME_TO_NS, "cache write latency"
        ),
        hit_energy_nj=_quantity(
            block,
            r"Cache[ \t]+Hit[ \t]+Dynamic[ \t]+Energy",
            _ENERGY_UNIT,
            _ENERGY_TO_NJ,
            "cache hit dynamic energy",
        ),
        miss_energy_nj=_quantity(
            block,
            r"Cache[ \t]+Miss[ \t]+Dynamic[ \t]+Energy",
            _ENERGY_UNIT,
            _ENERGY_TO_NJ,
            "cache miss dynamic energy",
        ),
        write_energy_nj=_quantity(
            block,
            r"Cache[ \t]+Write[ \t]+Dynamic[ \t]+Energy",
            _ENERGY_UNIT,
            _ENERGY_TO_NJ,
            "cache write dynamic energy",
        ),
        leakage_power_mw=_quantity(
            block,
            r"Cache[ \t]+Total[ \t]+Leakage[ \t]+Power",
            _POWER_UNIT,
            _POWER_TO_MW,
            "cache total leakage power",
        ),
        refresh_latency_us=None if refresh_latency_ns is None else refresh_latency_ns / 1e3,
        refresh_energy_nj=_quantity(
            block,
            r"Cache[ \t]+Refresh[ \t]+Dynamic[ \t]+Energy",
            _ENERGY_UNIT,
            _ENERGY_TO_NJ,
            "cache refresh dynamic energy",
            required=False,
        ),
        refresh_power_mw=_quantity(
            block,
            r"Cache[ \t]+Refresh[ \t]+Power",
            _POWER_UNIT,
            _POWER_TO_MW,
            "cache refresh power",
            required=False,
        ),
        availability_percent=_availability(block),
    )


_CAPACITY_RE = re.compile(
    r"^(?P<indent>[ \t]*)-Capacity[ \t]*\([ \t]*(?P<unit>B|KB|KiB|MB|MiB)[ \t]*\)"
    r"[ \t]*:[ \t]*(?P<value>\d+)(?P<tail>[ \t]*(?:(?://|#).*)?)$",
    re.IGNORECASE | re.MULTILINE,
)
_ASSOCIATIVITY_RE = re.compile(
    r"^(?P<indent>[ \t]*)-Associativity(?:[ \t]*\([^\n)]*\))?[ \t]*:[ \t]*"
    r"(?P<value>\d+)(?P<tail>[ \t]*(?:(?://|#).*)?)$",
    re.IGNORECASE | re.MULTILINE,
)
_FORCE_BANK_A_RE = re.compile(
    r"^[ \t]*-ForceBankA(?:[ \t]*\([^\n)]*\))?[ \t]*:[ \t]*"
    r"(?P<rows>\d+)[ \t]*(?:x|×)[ \t]*(?P<cols>\d+)",
    re.IGNORECASE | re.MULTILINE,
)
_MEMORY_CELL_RE = re.compile(
    r"^[ \t]*-MemoryCellInputFile[ \t]*:[ \t]*(?:\"(?P<double>[^\"]+)\"|'(?P<single>[^']+)'|(?P<plain>\S+))",
    re.IGNORECASE | re.MULTILINE,
)
_MEMORY_CELL_LINE_RE = re.compile(
    r"^(?P<indent>[ \t]*)-MemoryCellInputFile[ \t]*:[ \t]*.*$",
    re.IGNORECASE | re.MULTILINE,
)
_RETENTION_RE = re.compile(
    r"^(?P<indent>[ \t]*)-RetentionTime[ \t]*\([ \t]*us[ \t]*\)[ \t]*:[ \t]*.*$",
    re.IGNORECASE | re.MULTILINE,
)
_CELL_TEMPERATURE_RE = re.compile(
    r"^(?P<indent>[ \t]*)-Temperature[ \t]*\([ \t]*K[ \t]*\)[ \t]*:[ \t]*.*$",
    re.IGNORECASE | re.MULTILINE,
)


def _single_match(pattern: re.Pattern, text: str, name: str) -> re.Match:
    matches = list(pattern.finditer(text))
    if not matches:
        raise NSCacheConfigError("missing {} directive".format(name))
    if len(matches) > 1:
        raise NSCacheConfigError("multiple active {} directives".format(name))
    return matches[0]


def _capacity_kib(value: int, unit: str) -> int:
    normal = unit.lower()
    if normal == "b":
        if value % 1024:
            raise NSCacheConfigError("capacity in bytes is not an integral number of KiB")
        return value // 1024
    if normal in ("kb", "kib"):
        return value
    if normal in ("mb", "mib"):
        return value * 1024
    raise NSCacheConfigError("unsupported capacity unit '{}'".format(unit))


def derive_force_bank_count(config_text: str) -> Optional[int]:
    """Return the A*B subarray product from ``-ForceBankA``, if present.

    Despite the directive name, this does not return replicated cache banks.
    """

    matches = list(_FORCE_BANK_A_RE.finditer(config_text))
    if not matches:
        return None
    if len(matches) > 1:
        raise NSCacheConfigError("multiple active ForceBankA directives")
    rows = int(matches[0].group("rows"))
    cols = int(matches[0].group("cols"))
    if rows <= 0 or cols <= 0:
        raise NSCacheConfigError("ForceBankA dimensions must be > 0")
    return rows * cols


def parse_config_text(config_text: str) -> CacheConfig:
    """Read capacity, associativity, and optional forced bank count from text."""

    capacity = _single_match(_CAPACITY_RE, config_text, "Capacity")
    associativity = _single_match(_ASSOCIATIVITY_RE, config_text, "Associativity")
    return CacheConfig(
        capacity_kib=_capacity_kib(int(capacity.group("value")), capacity.group("unit")),
        associativity=int(associativity.group("value")),
        force_bank_count=derive_force_bank_count(config_text),
    )


def read_config(path: PathLike) -> CacheConfig:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise NSCacheConfigError("could not read config '{}': {}".format(path, exc)) from exc
    return parse_config_text(text)


def patch_config_text(
    template_text: str,
    capacity_kib: Optional[int] = None,
    associativity: Optional[int] = None,
) -> str:
    """Return a config variant; the input string and source file remain unchanged."""

    if capacity_kib is not None:
        if isinstance(capacity_kib, bool) or int(capacity_kib) != capacity_kib or capacity_kib <= 0:
            raise NSCacheConfigError("capacity_kib must be a positive integer")
        _single_match(_CAPACITY_RE, template_text, "Capacity")

        def capacity_replacement(match: re.Match) -> str:
            return "{}-Capacity (KB): {}{}".format(
                match.group("indent"), int(capacity_kib), match.group("tail") or ""
            )

        template_text = _CAPACITY_RE.sub(capacity_replacement, template_text, count=1)

    if associativity is not None:
        if isinstance(associativity, bool) or int(associativity) != associativity or associativity <= 0:
            raise NSCacheConfigError("associativity must be a positive integer")
        _single_match(_ASSOCIATIVITY_RE, template_text, "Associativity")

        def associativity_replacement(match: re.Match) -> str:
            return "{}-Associativity (for cache only): {}{}".format(
                match.group("indent"), int(associativity), match.group("tail") or ""
            )

        template_text = _ASSOCIATIVITY_RE.sub(associativity_replacement, template_text, count=1)

    return template_text


def patch_memory_cell_input(config_text: str, memory_cell_path: PathLike) -> str:
    """Point a generated config at a generated cell file."""

    _single_match(_MEMORY_CELL_LINE_RE, config_text, "MemoryCellInputFile")
    path = os.fspath(memory_cell_path)
    if not path or "\n" in path or "\r" in path:
        raise NSCacheConfigError("memory_cell_path must be a non-empty single-line path")
    return _MEMORY_CELL_LINE_RE.sub(
        lambda match: '{}-MemoryCellInputFile: "{}"'.format(
            match.group("indent"), path.replace('"', '\\"')
        ),
        config_text,
        count=1,
    )


def memory_cell_input(config_text: str) -> Optional[str]:
    """Return the configured memory-cell path, without surrounding quotes."""

    matches = list(_MEMORY_CELL_RE.finditer(config_text))
    if not matches:
        return None
    if len(matches) > 1:
        raise NSCacheConfigError("multiple active MemoryCellInputFile directives")
    match = matches[0]
    return match.group("double") or match.group("single") or match.group("plain")


def patch_cell_text(
    template_text: str,
    retention_time_us: Optional[float] = None,
    temperature_k: Optional[float] = None,
) -> str:
    """Return a cell variant with explicit retention calibration metadata."""

    result = template_text.rstrip() + "\n"
    directives = (
        (_RETENTION_RE, retention_time_us, "-RetentionTime (us)"),
        (_CELL_TEMPERATURE_RE, temperature_k, "-Temperature (K)"),
    )
    for pattern, raw_value, label in directives:
        if raw_value is None:
            continue
        value = float(raw_value)
        if not math.isfinite(value) or value <= 0:
            raise NSCacheConfigError("{} must be finite and > 0".format(label))
        matches = list(pattern.finditer(result))
        if len(matches) > 1:
            raise NSCacheConfigError("multiple active {} directives".format(label))
        replacement = "{}: {:.12g}".format(label, value)
        if matches:
            result = pattern.sub(replacement, result, count=1)
        else:
            result += replacement + "\n"
    return result


def render_config(
    template_path: PathLike,
    capacity_kib: Optional[int] = None,
    associativity: Optional[int] = None,
) -> str:
    """Read a template and return a patched variant without writing the template."""

    try:
        text = Path(template_path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise NSCacheConfigError("could not read template '{}': {}".format(template_path, exc)) from exc
    return patch_config_text(text, capacity_kib=capacity_kib, associativity=associativity)




def _working_directory(cwd: Optional[PathLike]) -> Path:
    path = Path.cwd() if cwd is None else Path(cwd).expanduser()
    path = path.resolve()
    if not path.is_dir():
        raise NSCacheRunError("working directory does not exist: {}".format(path))
    return path


def _resolve_binary(binary: PathLike, cwd: Path) -> Path:
    raw = os.fspath(binary)
    candidate = Path(raw).expanduser()
    candidate = candidate if candidate.is_absolute() else cwd / candidate
    if candidate.is_file():
        return candidate.resolve()
    located = shutil.which(raw)
    if located:
        return Path(located).resolve()
    raise NSCacheRunError("NS-Cache binary not found: {}".format(binary))


def _resolve_config(config: PathLike, cwd: Path) -> Path:
    candidate = Path(config).expanduser()
    candidate = candidate if candidate.is_absolute() else cwd / candidate
    if not candidate.is_file():
        raise NSCacheRunError("NS-Cache config not found: {}".format(candidate))
    return candidate.resolve()


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _output_tail(stdout: str, stderr: str, limit: int = 4000) -> str:
    combined = stdout
    if stderr:
        combined += ("\n" if combined else "") + "[stderr]\n" + stderr
    if len(combined) > limit:
        combined = "...<truncated>...\n" + combined[-limit:]
    return combined.rstrip()


def run_nscache_output(
    nsc_binary: PathLike,
    config_path: PathLike,
    cwd: Optional[PathLike] = None,
    timeout_s: float = 300.0,
) -> str:
    """Run NS-Cache and return combined output, with bounded execution time."""

    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")
    workdir = _working_directory(cwd)
    binary = _resolve_binary(nsc_binary, workdir)
    config = _resolve_config(config_path, workdir)
    command = [str(binary), str(config)]
    display = shlex.join(command)
    try:
        result = subprocess.run(
            command,
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        tail = _output_tail(_as_text(exc.stdout), _as_text(exc.stderr))
        message = "NS-Cache timed out after {:.3f}s\nCommand: {}".format(timeout_s, display)
        if tail:
            message += "\nOutput:\n{}".format(tail)
        raise NSCacheRunError(message) from exc
    except OSError as exc:
        raise NSCacheRunError("could not execute NS-Cache\nCommand: {}\nError: {}".format(display, exc)) from exc

    if result.returncode != 0:
        tail = _output_tail(result.stdout, result.stderr)
        message = "NS-Cache exited with status {}\nCommand: {}".format(result.returncode, display)
        if tail:
            message += "\nOutput:\n{}".format(tail)
        raise NSCacheRunError(message)

    if result.stderr:
        return result.stdout + ("\n" if result.stdout else "") + result.stderr
    return result.stdout


def run_nscache(
    nsc_binary: PathLike,
    config_path: PathLike,
    cwd: Optional[PathLike] = None,
    timeout_s: float = 300.0,
) -> CachePPA:
    """Run NS-Cache and parse its final cache summary."""

    output = run_nscache_output(nsc_binary, config_path, cwd=cwd, timeout_s=timeout_s)
    failure_markers = (
        "No valid solutions for tags.",
        "No valid solutions.",
        "This is not a valid cache configuration.",
        "ERROR: DATA capacity violation",
        "[ERROR]",
    )
    for marker in failure_markers:
        if marker in output:
            raise NSCacheRunError(
                "NS-Cache reported an invalid design: {}\nNS-Cache output:\n{}".format(
                    marker, _output_tail(output, "")
                )
            )
    solution_matches = re.findall(
        r"numSolutions\s*=\s*(\d+)\s*/\s*numDesigns\s*=\s*(\d+)", output
    )
    if solution_matches and int(solution_matches[-1][0]) <= 0:
        raise NSCacheRunError(
            "NS-Cache found zero valid solutions\nNS-Cache output:\n{}".format(
                _output_tail(output, "")
            )
        )
    if "Finished!" not in output:
        raise NSCacheRunError(
            "NS-Cache output is incomplete (missing 'Finished!')\nNS-Cache output:\n{}".format(
                _output_tail(output, "")
            )
        )
    try:
        return parse_summary(output)
    except NSCacheParseError as exc:
        tail = _output_tail(output, "")
        raise NSCacheParseError("{}\nNS-Cache output:\n{}".format(exc, tail)) from exc


def _hash_bytes(digest: "hashlib._Hash", label: str, value: bytes) -> None:
    label_bytes = label.encode("utf-8")
    digest.update(len(label_bytes).to_bytes(8, "big"))
    digest.update(label_bytes)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _hash_file(digest: "hashlib._Hash", label: str, path: Path) -> None:
    try:
        with path.open("rb") as handle:
            label_bytes = label.encode("utf-8")
            digest.update(len(label_bytes).to_bytes(8, "big"))
            digest.update(label_bytes)
            size = path.stat().st_size
            digest.update(size.to_bytes(8, "big"))
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise PPACacheError("could not hash '{}': {}".format(path, exc)) from exc


def build_ppa_cache_key(
    nsc_binary: PathLike,
    config_path: PathLike,
    cwd: Optional[PathLike] = None,
) -> str:
    """Hash the adapter schema, binary, config, and referenced memory-cell file."""

    workdir = _working_directory(cwd)
    binary = _resolve_binary(nsc_binary, workdir)
    config = _resolve_config(config_path, workdir)
    try:
        config_bytes = config.read_bytes()
        config_text = config_bytes.decode("utf-8", errors="replace")
    except OSError as exc:
        raise PPACacheError("could not read config '{}': {}".format(config, exc)) from exc

    digest = hashlib.sha256()
    _hash_bytes(digest, "adapter-schema", b"nscache-ppa-v1")
    _hash_file(digest, "nsc-binary", binary)
    _hash_bytes(digest, "config", config_bytes)

    cell_matches = list(_MEMORY_CELL_RE.finditer(config_text))
    if len(cell_matches) > 1:
        raise NSCacheConfigError("multiple active MemoryCellInputFile directives")
    if cell_matches:
        match = cell_matches[0]
        raw_cell = match.group("double") or match.group("single") or match.group("plain")
        cell = Path(raw_cell).expanduser()
        cell = cell if cell.is_absolute() else workdir / cell
        if not cell.is_file():
            raise NSCacheConfigError("memory-cell file not found: {}".format(cell))
        _hash_file(digest, "memory-cell", cell.resolve())
    return digest.hexdigest()


class JsonPPACache:
    """A compact, atomic JSON store keyed by SHA-256 PPA fingerprints."""

    SCHEMA_VERSION = 1

    def __init__(self, path: PathLike):
        self.path = Path(path)

    def _empty(self) -> Dict[str, Any]:
        return {"schema_version": self.SCHEMA_VERSION, "entries": {}}

    def _read(self) -> Dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PPACacheError("could not read PPA cache '{}': {}".format(self.path, exc)) from exc
        if not isinstance(data, dict):
            raise PPACacheError("PPA cache root must be a JSON object")
        if data.get("schema_version") != self.SCHEMA_VERSION:
            raise PPACacheError("unsupported PPA cache schema version")
        if not isinstance(data.get("entries"), dict):
            raise PPACacheError("PPA cache 'entries' must be a JSON object")
        return data

    @staticmethod
    def _validate_key(key: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise PPACacheError("PPA cache key must be a lowercase SHA-256 hex digest")

    def get(self, key: str) -> Optional[CachePPA]:
        self._validate_key(key)
        raw = self._read()["entries"].get(key)
        if raw is None:
            return None
        try:
            return CachePPA.from_dict(raw)
        except (TypeError, ValueError) as exc:
            raise PPACacheError("invalid CachePPA entry for key {}: {}".format(key, exc)) from exc

    def put(self, key: str, ppa: CachePPA) -> None:
        self._validate_key(key)
        if not isinstance(ppa, CachePPA):
            raise TypeError("ppa must be a CachePPA")
        data = self._read()
        data["entries"][key] = ppa.to_dict()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self.path.parent),
                prefix=self.path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
            temporary_name = None
        except OSError as exc:
            raise PPACacheError("could not update PPA cache '{}': {}".format(self.path, exc)) from exc
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink()
                except OSError:
                    pass




def cached_run_nscache(
    cache: Union[JsonPPACache, PathLike],
    nsc_binary: PathLike,
    config_path: PathLike,
    cwd: Optional[PathLike] = None,
    timeout_s: float = 300.0,
) -> Tuple[CachePPA, bool]:
    """Return ``(ppa, cache_hit)`` for an NS-Cache configuration."""

    store = cache if isinstance(cache, JsonPPACache) else JsonPPACache(cache)
    key = build_ppa_cache_key(nsc_binary, config_path, cwd=cwd)
    cached = store.get(key)
    if cached is not None:
        return cached, True
    ppa = run_nscache(nsc_binary, config_path, cwd=cwd, timeout_s=timeout_s)
    store.put(key, ppa)
    return ppa, False


__all__ = [
    "CacheConfig",
    "CachePPA",
    "JsonPPACache",
    "NSCacheConfigError",
    "NSCacheError",
    "NSCacheParseError",
    "NSCacheRunError",
    "PPACacheError",
    "build_ppa_cache_key",
    "cached_run_nscache",
    "derive_force_bank_count",
    "parse_config_text",
    "parse_summary",
    "patch_cell_text",
    "patch_config_text",
    "patch_memory_cell_input",
    "read_config",
    "render_config",
    "run_nscache",
    "run_nscache_output",
]
