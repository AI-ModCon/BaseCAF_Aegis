"""CLI entry point for Aegis."""

import argparse
import sys

from .config import AegisConfig, load_config, merge_cli_args, _normalize_models
from .launcher import launch_instances, stage_conda_env, stage_weights
from .scheduler import generate_pbs_script, submit_job


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add config flags shared by both subcommands."""
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model", type=str, help="HuggingFace model name")
    parser.add_argument("--instances", type=int, help="Number of vLLM instances to launch")
    parser.add_argument(
        "--tensor-parallel-size", type=int, dest="tensor_parallel_size",
        help="Number of GPUs per instance",
    )
    parser.add_argument(
        "--port-start", type=int, dest="port_start",
        help="Base port for each node (incremented for additional instances on the same node)",
    )
    parser.add_argument("--hf-home", type=str, dest="hf_home", help="Path to model weights")
    parser.add_argument("--hf-token", type=str, dest="hf_token", help="HuggingFace token")
    parser.add_argument(
        "--model-source", type=str, dest="model_source",
        help="Source path for bcast weight staging",
    )
    parser.add_argument("--walltime", type=str, help="PBS walltime")
    parser.add_argument("--queue", type=str, help="PBS queue name")
    parser.add_argument("--account", type=str, help="PBS account/project")
    parser.add_argument("--filesystems", type=str, help="PBS filesystem directive")
    parser.add_argument(
        "--extra-vllm-args", nargs="*", dest="extra_vllm_args",
        help="Additional arguments passed to vllm serve",
    )
    parser.add_argument(
        "--redis-port", type=int, dest="redis_port",
        help="Port for the Redis service registry (default: 6379)",
    )
    parser.add_argument(
        "--conda-env", type=str, dest="conda_env",
        help="Path to a conda-pack tarball to distribute and activate on all nodes",
    )


def _build_config(args) -> AegisConfig:
    """Build an AegisConfig from a config file + CLI overrides."""
    if args.config:
        config = load_config(args.config)
    else:
        config = AegisConfig()
    merge_cli_args(config, args)
    _normalize_models(config)
    return config


def cmd_submit(args) -> None:
    """Generate and submit a PBS batch job."""
    config = _build_config(args)

    if not config.models:
        print("Error: --model is required (or provide a 'models' list in the config file).", file=sys.stderr)
        sys.exit(1)
    if not config.account:
        print("Error: --account is required.", file=sys.stderr)
        sys.exit(1)

    script = generate_pbs_script(config)

    if args.dry_run:
        print(script)
        return

    submit_job(script)


def cmd_launch(args) -> None:
    """Run inside an existing allocation: stage weights and launch vLLM instances."""
    config = _build_config(args)

    if not config.models:
        print("Error: --model is required (or provide a 'models' list in the config file).", file=sys.stderr)
        sys.exit(1)

    if not args.skip_staging:
        stage_conda_env(config)
        stage_weights(config)
    else:
        print("Skipping conda env and weight staging (--skip-staging)", file=sys.stderr)
    launch_instances(config)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="aegis",
        description="Aegis: HPC LLM Instance Launcher",
    )
    subparsers = parser.add_subparsers(dest="command")

    # submit
    submit_parser = subparsers.add_parser(
        "submit", help="Generate and submit a PBS batch job",
    )
    _add_common_args(submit_parser)
    submit_parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Print the generated PBS script without submitting",
    )
    submit_parser.set_defaults(func=cmd_submit)

    # launch
    launch_parser = subparsers.add_parser(
        "launch", help="Launch vLLM instances inside an existing allocation",
    )
    _add_common_args(launch_parser)
    launch_parser.add_argument(
        "--skip-staging", action="store_true", dest="skip_staging",
        help="Skip conda env and weight staging (use when already staged)",
    )
    launch_parser.set_defaults(func=cmd_launch)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)
