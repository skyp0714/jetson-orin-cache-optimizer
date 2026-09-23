"""NS-Cache + Accel-Sim cache optimization pipeline."""

__version__ = "0.1.0"

from .config import ConfigError, load_config, validate_paths
from .models import AccessCounts, AppResult, CacheDesign, Evaluation
from .pipeline import CacheOptimizationPipeline, PipelineError
from .report import write_reports

__all__ = [
    "AccessCounts",
    "AppResult",
    "CacheDesign",
    "CacheOptimizationPipeline",
    "ConfigError",
    "Evaluation",
    "PipelineError",
    "__version__",
    "load_config",
    "validate_paths",
    "write_reports",
]
