"""CLI entry point for Aegis."""

import argparse
import csv
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import AegisConfig, load_config, merge_cli_args, _normalize_models
from .launcher import launch_instances, stage_conda_env, stage_weights, set_verbose as _launcher_set_verbose
from .registry import ServiceRegistryClient, ServiceStatus
from .scheduler import (
    SSHConnection,
    generate_pbs_script,
    submit_job,
    submit_job_remote,
    wait_for_endpoints,
    set_verbose as _scheduler_set_verbose,
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
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=False,
        help="Print subprocess commands and other debug details",
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

    # Forward HF_TOKEN from the environment to the PBS job if not in config
    hf_token = None if config.hf_token else os.environ.get("HF_TOKEN")

    ssh = None
    if args.remote:
        ssh = SSHConnection(args.remote)
        ssh.connect()

    try:
        # Remove stale endpoints and registry files so --wait doesn't find them immediately
        if args.wait:
            registry_file = str(Path(config.endpoints_file).parent / "aegis_registry_url.txt")
            if ssh:
                ssh.run(f"rm -f {config.endpoints_file} {registry_file}")
            else:
                Path(config.endpoints_file).unlink(missing_ok=True)
                Path(registry_file).unlink(missing_ok=True)

        if ssh:
            job_id = submit_job_remote(script, ssh, hf_token=hf_token)
        else:
            job_id = submit_job(script, hf_token=hf_token)

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
        help="Hostname of the registry server, typically the first PBS node (default: localhost)",
    )
    parser.add_argument(
        "--registry-port", type=int, default=8471,
        help="Port of the registry HTTP API (default: 8471)",
    )
    parser.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format (default: text)",
    )


# ---------------------------------------------------------------------------
# aegis bench subcommand
# ---------------------------------------------------------------------------

def _read_endpoints_file(path: str) -> list[str]:
    """Read endpoints from a file, one host:port per line."""
    endpoints = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                endpoints.append(line)
    return endpoints


_BENCH_METRICS = [
    "request_throughput", "output_throughput", "total_token_throughput",
    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
    "duration", "completed", "total_input_tokens", "total_output_tokens",
]

_ADDITIVE_METRICS = {
    "request_throughput", "output_throughput_tok_s", "total_token_throughput",
    "completed", "total_input_tokens", "total_output_tokens",
}


def _parse_bench_results(result_dir: str) -> list[dict]:
    """Parse JSON result files produced by vllm bench serve."""
    results = []
    for path in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        with open(path) as f:
            data = json.load(f)
        row: dict = {}
        # Identify endpoint from the base_url stored in the JSON
        base_url = data.get("base_url", "")
        if base_url:
            # Strip scheme and /v1 suffix to get host:port
            endpoint = base_url.replace("http://", "").replace("https://", "").rstrip("/")
            if endpoint.endswith("/v1"):
                endpoint = endpoint[:-3]
            row["endpoint"] = endpoint
        else:
            row["endpoint"] = Path(path).stem
        for key in _BENCH_METRICS:
            if key in data:
                row[key] = data[key]
        if "output_throughput" in row:
            row["output_throughput_tok_s"] = row.pop("output_throughput")
        results.append(row)
    return results


def _write_bench_csv(results: list[dict], output_path: str | None = None) -> None:
    """Write benchmark results as CSV with a summary row."""
    if not results:
        return
    columns = list(results[0].keys())
    # Ensure consistent column order across all rows
    for r in results[1:]:
        for k in r:
            if k not in columns:
                columns.append(k)

    numeric_cols = [c for c in columns if c != "endpoint"]

    # Compute TOTAL row (sum additive metrics only; leave latency columns blank)
    total: dict = {"endpoint": "TOTAL"}
    for col in numeric_cols:
        if col in _ADDITIVE_METRICS:
            vals = [r[col] for r in results if col in r and isinstance(r[col], (int, float))]
            if vals:
                total[col] = sum(vals)

    # Compute summary row
    summary: dict = {"endpoint": "SUMMARY"}
    for col in numeric_cols:
        vals = [r[col] for r in results if col in r and isinstance(r[col], (int, float))]
        if vals:
            mn, mx, avg = min(vals), max(vals), sum(vals) / len(vals)
            summary[col] = f"min={mn:.2f} max={mx:.2f} mean={avg:.2f}"

    if output_path:
        fh = open(output_path, "w", newline="")
    else:
        fh = sys.stdout

    try:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
        writer.writerow(total)
        writer.writerow(summary)
    finally:
        if output_path:
            fh.close()


def cmd_bench(args) -> None:
    """Benchmark launched vLLM instances via vllm bench serve."""
    # Resolve endpoints
    if args.registry_host != "localhost":
        client = ServiceRegistryClient(host=args.registry_host, port=args.registry_port)
        services = client.get_healthy_services()
        endpoints = [f"{s.host}:{s.port}" for s in services]
        if not endpoints:
            print("Error: no healthy endpoints found in registry.", file=sys.stderr)
            sys.exit(1)
    else:
        ep_path = args.endpoints_file
        if not os.path.exists(ep_path):
            print(f"Error: endpoints file '{ep_path}' not found.", file=sys.stderr)
            sys.exit(1)
        endpoints = _read_endpoints_file(ep_path)
        if not endpoints:
            print(f"Error: no endpoints found in '{ep_path}'.", file=sys.stderr)
            sys.exit(1)

    extra = args.extra_args if args.extra_args else []
    # Strip leading '--' separator that REMAINDER captures
    if extra and extra[0] == "--":
        extra = extra[1:]

    # Create temp directory for JSON result files
    result_dir = tempfile.mkdtemp(prefix="aegis_bench_")

    # Group endpoints by port — endpoints sharing a port get a single SPMD
    # segment; differing ports require separate MPMD segments.
    hosts_by_port: dict[str, list[str]] = {}
    for ep in endpoints:
        host, port = ep.rsplit(":", 1)
        hosts_by_port.setdefault(port, []).append(host)

    # Forward HF_TOKEN to mpiexec ranks if set in the environment
    hf_token = os.environ.get("HF_TOKEN")

    # Build mpiexec command: SPMD per port group, MPMD across groups
    mpi_cmd: list[str] = ["mpiexec"]
    first = True
    for port, hosts in hosts_by_port.items():
        if not first:
            mpi_cmd.append(":")
        first = False
        vllm_args = [
            "vllm", "bench", "serve",
            "--model", args.model,
            "--num-prompts", str(args.num_prompts),
            "--save-result",
            "--result-dir", result_dir,
            *extra,
        ]
        if args.conda_env:
            env_export = f"export HF_TOKEN={shlex.quote(hf_token)}" if hf_token else ""
            activate = f"source {args.conda_env}/bin/activate"
            parts = [p for p in [activate, env_export, shlex.join(vllm_args)] if p]
            rank_cmd = ["bash", "-c", " && ".join(parts)]
        else:
            env_export = f"export HF_TOKEN={shlex.quote(hf_token)}" if hf_token else ""
            parts = [p for p in ["module load frameworks", env_export, shlex.join(vllm_args)] if p]
            rank_cmd = ["bash", "-l", "-c", " && ".join(parts)]
        env_flags = ["-env", "HF_TOKEN", hf_token] if hf_token else []
        mpi_cmd.extend([
            "-n", str(len(hosts)),
            "-hosts", ",".join(hosts),
            *env_flags,
            *rank_cmd,
        ])

    print(f"Launching benchmarks on {len(endpoints)} endpoint(s) via mpiexec")
    for ep in endpoints:
        print(f"  {ep}")
    if args.verbose:
        print(f"  [bench] {shlex.join(mpi_cmd)}", file=sys.stderr)

    try:
        proc = subprocess.run(mpi_cmd)
        if proc.returncode != 0:
            print(f"\nmpiexec exited with code {proc.returncode}", file=sys.stderr)
            sys.exit(proc.returncode)

        # Post-process results into CSV
        results = _parse_bench_results(result_dir)
        if results:
            cumulative_tok_s = sum(
                r["output_throughput_tok_s"]
                for r in results
                if isinstance(r.get("output_throughput_tok_s"), (int, float))
            )
            print(f"\nCumulative output throughput: {cumulative_tok_s:.1f} tok/s ({len(results)} endpoint(s))")
            output_path = getattr(args, "output", None)
            _write_bench_csv(results, output_path)
            if output_path:
                print(f"Results written to {output_path}")
        else:
            print("Warning: no benchmark result files found.", file=sys.stderr)
    finally:
        shutil.rmtree(result_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# aegis shutdown subcommand
# ---------------------------------------------------------------------------

def cmd_shutdown(args) -> None:
    """Shut down launched vLLM instances and/or cancel PBS jobs."""
    endpoints_file = args.endpoints_file
    job_id = args.job_id
    has_endpoints = os.path.exists(endpoints_file)

    if not has_endpoints and not job_id:
        print(
            f"Error: endpoints file '{endpoints_file}' not found and no --job-id provided.\n"
            "Provide --endpoints-file pointing to a valid file and/or --job-id.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Kill vLLM processes on nodes listed in the endpoints file
    if has_endpoints:
        endpoints = _read_endpoints_file(endpoints_file)
        if not endpoints:
            print(f"Warning: no endpoints found in '{endpoints_file}', skipping process kill.", file=sys.stderr)
        else:
            # Extract unique hostnames from host:port lines
            hosts = list(dict.fromkeys(ep.rsplit(":", 1)[0] for ep in endpoints))
            hosts_str = ",".join(hosts)
            print(f"Killing vLLM processes on: {hosts_str}")
            pkill_cmd = ["mpiexec", "-hosts", hosts_str, "-ppn", "1", "pkill", "-f", "vllm serve"]
            if args.verbose:
                print(f"  [shutdown] {shlex.join(pkill_cmd)}", file=sys.stderr)
            result = subprocess.run(pkill_cmd)
            if result.returncode == 0:
                print("vLLM processes terminated successfully.")
            else:
                print(f"pkill exited with code {result.returncode}", file=sys.stderr)

    # Cancel the PBS job if requested
    if job_id:
        ssh = None
        if args.remote:
            ssh = SSHConnection(args.remote)
            ssh.connect()
        try:
            if ssh:
                print(f"Cancelling PBS job {job_id} via {args.remote}")
                ssh.run(f"qdel {job_id}")
            else:
                print(f"Cancelling PBS job {job_id}")
                if args.verbose:
                    print(f"  [qdel] qdel {job_id}", file=sys.stderr)
                subprocess.run(["qdel", job_id], check=True)
            print(f"Job {job_id} cancelled.")
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            print(f"Error cancelling job {job_id}: {exc}", file=sys.stderr)
            sys.exit(1)
        finally:
            if ssh:
                ssh.close()


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
        "--bench", action="store_true", default=None,
        help="Run aegis bench after instances are healthy and exit the job",
    )
    submit_parser.add_argument(
        "--bench-num-prompts", type=int, dest="bench_num_prompts", default=None,
        help="Number of prompts per endpoint for the benchmark (default: 100)",
    )
    submit_parser.add_argument(
        "--bench-output", type=str, dest="bench_output", default=None,
        help="Path to write benchmark CSV results",
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

    # bench
    bench_parser = subparsers.add_parser("bench", help="Benchmark launched vLLM instances")
    bench_parser.add_argument("--model", type=str, required=True, help="Model name for the benchmark")
    bench_parser.add_argument(
        "--num-prompts", type=int, default=100, dest="num_prompts",
        help="Number of prompts per endpoint (default: 100)",
    )
    bench_parser.add_argument(
        "--endpoints-file", type=str, default="aegis_endpoints.txt", dest="endpoints_file",
        help="Path to endpoints file (default: aegis_endpoints.txt)",
    )
    bench_parser.add_argument(
        "--output", type=str, default=None,
        help="Path to write CSV results (default: print to stdout)",
    )
    bench_parser.add_argument(
        "--conda-env", type=str, default=None, dest="conda_env",
        help="Path to staged conda environment directory (default: /tmp/conda_env)",
    )
    _add_registry_args(bench_parser)
    bench_parser.add_argument(
        "--verbose", "-v", action="store_true", default=False,
        help="Print subprocess commands and other debug details",
    )
    bench_parser.add_argument(
        "extra_args", nargs=argparse.REMAINDER,
        help="Extra arguments passed through to vllm bench serve (put after --)",
    )
    bench_parser.set_defaults(func=cmd_bench)

    # shutdown
    shutdown_parser = subparsers.add_parser("shutdown", help="Shut down launched vLLM instances")
    shutdown_parser.add_argument(
        "--endpoints-file", type=str, default="aegis_endpoints.txt", dest="endpoints_file",
        help="Path to endpoints file (default: aegis_endpoints.txt)",
    )
    shutdown_parser.add_argument(
        "--job-id", type=str, dest="job_id",
        help="PBS job ID to cancel via qdel",
    )
    shutdown_parser.add_argument(
        "--remote", type=str, metavar="USER@HOST",
        help="Run qdel via SSH on a remote login node",
    )
    shutdown_parser.add_argument(
        "--verbose", "-v", action="store_true", default=False,
        help="Print subprocess commands and other debug details",
    )
    shutdown_parser.set_defaults(func=cmd_shutdown)

    args = parser.parse_args(argv)

    _verbose = getattr(args, "verbose", False)
    _launcher_set_verbose(_verbose)
    _scheduler_set_verbose(_verbose)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "registry" and not args.registry_command:
        registry_parser.print_help()
        sys.exit(1)

    args.func(args)
