"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import accelsim, nscache
from .config import ConfigError, load_config
from .pipeline import CacheOptimizationPipeline, PipelineError


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Iteratively optimize Jetson Orin L1/L2 cell technology with NS-Cache + Accel-Sim."
    )
    parser.add_argument("--config", type=Path, required=True, help="Pipeline JSON configuration")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate paths, traces, and baseline mappings without running NS-Cache/Accel-Sim",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=None,
        help="Override maximum simulated candidates per technology/objective",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Override concurrent candidate simulations",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore matching completed simulation logs (does not delete them)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        if args.max_evaluations is not None:
            if args.max_evaluations <= 0:
                raise ConfigError("--max-evaluations must be > 0")
            config["optimizer"]["max_evaluations_per_objective"] = args.max_evaluations
        if args.max_workers is not None:
            if args.max_workers <= 0:
                raise ConfigError("--max-workers must be > 0")
            config["optimizer"]["max_workers"] = args.max_workers
        if args.no_resume:
            config["optimizer"]["resume"] = False

        pipeline = CacheOptimizationPipeline(config)
        errors = pipeline.validate()
        if errors:
            raise PipelineError("Configuration validation failed:\n- " + "\n- ".join(errors))
        for warning in pipeline.warnings:
            print("Warning: {}".format(warning), file=sys.stderr)
        if args.validate_only:
            print("Configuration is valid.")
            print("Applications: {}".format(len(config["applications"])))
            print("Output directory: {}".format(config["paths"]["output_dir"]))
            return 0

        result = pipeline.run()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ConfigError, PipelineError) as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        return 2
    except (accelsim.AccelSimError, nscache.NSCacheError, OSError) as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted. Completed PPA/simulation results remain resumable.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
