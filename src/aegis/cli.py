"""CLI entry point for Aegis."""

import argparse
import json
import os
import sys

from .config import AegisConfig, load_config, merge_cli_args, _normalize_models
from .launcher import launch_instances, stage_conda_env, stage_weights
from .registry import ServiceRegistryClient, ServiceStatus
from .scheduler import (
    SSHConnection,
    generate_pbs_script,
    submit_job,
    submit_job_remote,
    wait_for_endpoints,
)


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
        "--download-weights", action="store_true", dest="download_weights",
        default=None,
        help="Download model weights from HuggingFace Hub before staging",
    )
    parser.add_argument(
        "--extra-vllm-args", nargs="*", dest="extra_vllm_args",
        help="Additional arguments passed to vllm serve",
    )
    parser.add_argument(
        "--registry-port", type=int, dest="registry_port",
        help="Port for the service registry HTTP API (default: 8471)",
    )
    parser.add_argument(
        "--conda-env", type=str, dest="conda_env",
        help="Path to a conda-pack tarball to distribute and activate on all nodes",
    )
    parser.add_argument(
        "--startup-timeout", type=int, dest="startup_timeout",
        help="Seconds to wait for instances to become healthy (default: 600)",
    )
    parser.add_argument(
        "--endpoints-file", type=str, dest="endpoints_file",
        help="Output path for the endpoints file (default: aegis_endpoints.txt)",
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


def _resolve_hf_token(config: AegisConfig) -> str | None:
    """Return the HF token from config or environment."""
    return config.hf_token or os.environ.get("HF_TOKEN")


def _check_gated_models(config: AegisConfig) -> None:
    """Check whether any models require authentication and error early if
    an HF token is needed but not available."""
    token = _resolve_hf_token(config)
    if token:
        return  # token is available, nothing to check

    try:
        from huggingface_hub import model_info
    except ImportError:
        return  # can't check without the library; let it fail later

    for m in config.models:
        try:
            info = model_info(m.model)
            if getattr(info, "gated", False):
                print(
                    f"Error: Model '{m.model}' is gated and requires authentication.\n"
                    "Set HF_TOKEN in your environment or pass --hf-token.",
                    file=sys.stderr,
                )
                sys.exit(1)
        except Exception:
            # Network error or model not found — skip the check and let
            # vllm surface the real error at serve time.
            pass


def cmd_submit(args) -> None:
    """Generate and submit a PBS batch job."""
    config = _build_config(args)

    if not config.models:
        print("Error: --model is required (or provide a 'models' list in the config file).", file=sys.stderr)
        sys.exit(1)
    if not config.account:
        print("Error: --account is required.", file=sys.stderr)
        sys.exit(1)

    _check_gated_models(config)

    script = generate_pbs_script(config)

    if args.dry_run:
        print(script)
        return

    ssh = None
    if args.remote:
        ssh = SSHConnection(args.remote)
        ssh.connect()

    try:
        if ssh:
            job_id = submit_job_remote(script, ssh)
        else:
            job_id = submit_job(script)

        if args.wait:
            wait_for_endpoints(config.endpoints_file, job_id, ssh=ssh)
    finally:
        if ssh:
            ssh.close()


def cmd_launch(args) -> None:
    """Run inside an existing allocation: stage weights and launch vLLM instances."""
    config = _build_config(args)

    if not config.models:
        print("Error: --model is required (or provide a 'models' list in the config file).", file=sys.stderr)
        sys.exit(1)

    _check_gated_models(config)

    if not args.skip_staging:
        stage_conda_env(config)
        stage_weights(config)
    else:
        print("Skipping conda env and weight staging (--skip-staging)", file=sys.stderr)
    launch_instances(config)


# ---------------------------------------------------------------------------
# aegis registry subcommand
# ---------------------------------------------------------------------------

def _format_services(services, fmt: str) -> str:
    """Format a list of ServiceInfo objects for output."""
    if fmt == "json":
        return json.dumps([s.to_dict() for s in services], indent=2)
    lines = []
    for s in services:
        lines.append(f"{s.service_id}  {s.host}:{s.port}  {s.status}  last_seen={s.last_seen:.1f}")
    return "\n".join(lines) if lines else "(no services)"


def cmd_registry_list(args) -> None:
    client = ServiceRegistryClient(host=args.registry_host, port=args.registry_port)
    services = client.list_services(
        service_type=args.type,
        status_filter=args.status,
    )
    print(_format_services(services, args.format))


def cmd_registry_get(args) -> None:
    client = ServiceRegistryClient(host=args.registry_host, port=args.registry_port)
    service = client.get_service(args.service_id)
    if service is None:
        print(f"Service '{args.service_id}' not found.", file=sys.stderr)
        sys.exit(1)
    if args.format == "json":
        print(json.dumps(service.to_dict(), indent=2))
    else:
        print(f"{service.service_id}  {service.host}:{service.port}  {service.status}  last_seen={service.last_seen:.1f}")


def cmd_registry_list_healthy(args) -> None:
    client = ServiceRegistryClient(host=args.registry_host, port=args.registry_port)
    services = client.get_healthy_services(
        service_type=args.type,
        timeout_seconds=args.timeout,
    )
    print(_format_services(services, args.format))


def cmd_registry_count(args) -> None:
    client = ServiceRegistryClient(host=args.registry_host, port=args.registry_port)
    count = client.get_service_count(service_type=args.type)
    print(count)


def _add_registry_args(parser: argparse.ArgumentParser) -> None:
    """Add --registry-host and --registry-port to a registry sub-parser."""
    parser.add_argument(
        "--registry-host", type=str, default="localhost",
        help="Registry server host (default: localhost)",
    )
    parser.add_argument(
        "--registry-port", type=int, default=8471,
        help="Registry server port (default: 8471)",
    )
    parser.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format (default: text)",
    )


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
    submit_parser.add_argument(
        "--aegis-env", type=str, dest="aegis_env",
        help="Path to a conda environment containing the aegis package",
    )
    submit_parser.add_argument(
        "--wait", action="store_true",
        help="Block until instances are healthy and the endpoints file is written",
    )
    submit_parser.add_argument(
        "--remote", type=str, metavar="USER@HOST",
        help="Submit via SSH to a remote login node (e.g., user@aurora.alcf.anl.gov)",
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

    # registry
    registry_parser = subparsers.add_parser(
        "registry", help="Query the service registry",
    )
    registry_sub = registry_parser.add_subparsers(dest="registry_command")

    # registry list
    reg_list = registry_sub.add_parser("list", help="List all registered services")
    _add_registry_args(reg_list)
    reg_list.add_argument("--type", type=str, default=None, help="Filter by service type")
    reg_list.add_argument("--status", type=str, default=None, help="Filter by status")
    reg_list.set_defaults(func=cmd_registry_list)

    # registry get
    reg_get = registry_sub.add_parser("get", help="Get a single service by ID")
    _add_registry_args(reg_get)
    reg_get.add_argument("service_id", type=str, help="Service identifier")
    reg_get.set_defaults(func=cmd_registry_get)

    # registry list-healthy
    reg_healthy = registry_sub.add_parser("list-healthy", help="List healthy services")
    _add_registry_args(reg_healthy)
    reg_healthy.add_argument("--type", type=str, default=None, help="Filter by service type")
    reg_healthy.add_argument("--timeout", type=int, default=30, help="Heartbeat timeout in seconds")
    reg_healthy.set_defaults(func=cmd_registry_list_healthy)

    # registry count
    reg_count = registry_sub.add_parser("count", help="Count registered services")
    _add_registry_args(reg_count)
    reg_count.add_argument("--type", type=str, default=None, help="Filter by service type")
    reg_count.set_defaults(func=cmd_registry_count)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "registry" and not args.registry_command:
        registry_parser.print_help()
        sys.exit(1)

    args.func(args)
