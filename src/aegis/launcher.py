"""Core orchestration: stage weights, launch vLLM instances."""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path

from jinja2 import Environment, PackageLoader

from .config import AegisConfig, GPUS_PER_NODE, ModelConfig
from .registry import ServiceRegistry, ServiceInfo, ServiceStatus


def _project_root() -> Path:
    """Find the project root by searching upward for pyproject.toml."""
    anchor = Path(__file__).resolve().parent
    for directory in (anchor, *anchor.parents):
        if (directory / "pyproject.toml").exists():
            return directory
    raise FileNotFoundError(
        "Could not find project root (no pyproject.toml in any parent of "
        f"{anchor})"
    )


def _get_template_env() -> Environment:
    return Environment(
        loader=PackageLoader("aegis", "templates"),
        keep_trailing_newline=True,
    )


def _get_allocated_nodes() -> list[str]:
    """Read the list of unique nodes from PBS_NODEFILE."""
    nodefile = os.environ.get("PBS_NODEFILE")
    if not nodefile:
        print("Error: PBS_NODEFILE not set. Are you inside a PBS allocation?", file=sys.stderr)
        sys.exit(1)
    with open(nodefile) as f:
        nodes = list(dict.fromkeys(line.strip() for line in f if line.strip()))
    return nodes


def stage_conda_env(config: AegisConfig) -> None:
    """Compile bcast (if needed) and broadcast a conda-pack tarball to all nodes."""
    if not config.conda_env:
        return

    tools_dir = _project_root() / "tools"
    bcast_bin = tools_dir / "bcast"

    if not bcast_bin.exists():
        print("Compiling bcast...", file=sys.stderr)
        result = subprocess.run(["make", "bcast"], cwd=tools_dir)
        if result.returncode != 0:
            print("Failed to compile bcast.", file=sys.stderr)
            sys.exit(1)

    tarball = config.conda_env
    print(f"Staging conda env: {tarball} -> /tmp", file=sys.stderr)

    env = os.environ.copy()
    env["MPIR_CVAR_CH4_OFI_ENABLE_MULTI_NIC_STRIPING"] = "1"
    env["MPIR_CVAR_CH4_OFI_MAX_NICS"] = "4"

    result = subprocess.run(
        ["mpiexec", "-ppn", "1", "--cpu-bind", "numa", str(bcast_bin), tarball, "/tmp"],
        env=env,
    )
    if result.returncode != 0:
        print("Conda env staging failed.", file=sys.stderr)
        sys.exit(1)

    print("Conda env staging complete.", file=sys.stderr)


def stage_weights(config: AegisConfig) -> None:
    """Compile bcast (if needed) and broadcast model weights to local storage."""
    sources = [m.model_source for m in config.models if m.model_source]
    if not sources:
        print("No model_source specified, skipping weight staging.", file=sys.stderr)
        return

    tools_dir = _project_root() / "tools"
    bcast_bin = tools_dir / "bcast"

    if not bcast_bin.exists():
        print("Compiling bcast...", file=sys.stderr)
        result = subprocess.run(["make", "bcast"], cwd=tools_dir)
        if result.returncode != 0:
            print("Failed to compile bcast.", file=sys.stderr)
            sys.exit(1)

    dest = f"{config.hf_home}/hub"
    env = os.environ.copy()
    env["MPIR_CVAR_CH4_OFI_ENABLE_MULTI_NIC_STRIPING"] = "1"
    env["MPIR_CVAR_CH4_OFI_MAX_NICS"] = "4"

    for source in sources:
        print(f"Staging weights: {source} -> {dest}", file=sys.stderr)
        result = subprocess.run(
            ["mpiexec", "-ppn", "1", "--cpu-bind", "numa", str(bcast_bin), source, dest],
            env=env,
        )
        if result.returncode != 0:
            print(f"Weight staging failed for {source}.", file=sys.stderr)
            sys.exit(1)

    print("Weight staging complete.", file=sys.stderr)


def _wait_for_instances(
    endpoints: list[tuple[str, int]],
    poll_interval: int = 10,
    timeout: int = 1200,
) -> None:
    """Poll /health on each instance until all respond 200 or timeout expires."""
    # Bypass http_proxy env vars — health checks target internal compute nodes.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    ready = set()
    deadline = time.monotonic() + timeout
    cycle = 0

    while time.monotonic() < deadline:
        cycle += 1
        for node, port in endpoints:
            if (node, port) in ready:
                continue
            url = f"http://{node}:{port}/health"
            try:
                with opener.open(url, timeout=5) as resp:
                    if resp.status == 200:
                        ready.add((node, port))
                        print(
                            f"Instance {node}:{port} is ready "
                            f"({len(ready)}/{len(endpoints)})",
                            file=sys.stderr,
                        )
            except (urllib.error.URLError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                print(
                    f"[poll {cycle}] {node}:{port} not ready: {reason}",
                    file=sys.stderr,
                )

        if len(ready) == len(endpoints):
            return

        time.sleep(poll_interval)

    not_ready = [(n, p) for n, p in endpoints if (n, p) not in ready]
    print(
        f"Error: Timed out after {timeout}s waiting for instances: "
        + ", ".join(f"{n}:{p}" for n, p in not_ready),
        file=sys.stderr,
    )
    sys.exit(1)


def _check_redis_server() -> None:
    """Verify that a redis-server binary is available."""
    redis_stable = os.environ.get("REDIS_STABLE", "")
    if redis_stable and os.path.isfile(f"{redis_stable}/src/redis-server"):
        return
    if shutil.which("redis-server"):
        return
    print(
        "Error: redis-server not found.\n"
        "Install Redis and ensure redis-server is on your PATH,\n"
        "or set the REDIS_STABLE environment variable to a Redis build tree.",
        file=sys.stderr,
    )
    sys.exit(1)


def _start_redis(port: int) -> str:
    """Start a Redis server on the head node and return its bind address."""
    _check_redis_server()

    tools_dir = _project_root() / "tools"
    script = tools_dir / "start_redis.sh"

    result = subprocess.run(
        [str(script), "", str(port)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Failed to start Redis:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Parse bind address from script output (line: "...  Bind address: <addr>")
    redis_host = None
    for line in result.stdout.splitlines():
        if "Bind address:" in line:
            redis_host = line.split("Bind address:")[-1].strip()
            break

    if not redis_host:
        print("Error: could not determine Redis bind address from start script output.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Redis started on {redis_host}:{port}", file=sys.stderr)
    return redis_host


def _stop_redis() -> None:
    """Stop the Redis server using the stop script."""
    tools_dir = _project_root() / "tools"
    script = tools_dir / "stop_redis.sh"

    result = subprocess.run([str(script)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: Redis shutdown returned non-zero: {result.stderr}",
              file=sys.stderr)
    else:
        print("Redis stopped.", file=sys.stderr)


def _register_instances(
    endpoints: list[tuple[str, int]],
    config: AegisConfig,
    redis_host: str,
    redis_port: int,
    status: str = ServiceStatus.STARTING.value,
) -> None:
    """Register vLLM instances with the Redis service registry."""
    registry = ServiceRegistry(redis_host=redis_host, redis_port=redis_port)

    # Build a lookup from (node, port) -> ModelConfig so we can attach metadata
    model_lookup: dict[tuple[str, int], ModelConfig] = {}
    node_port_counter: dict[str, int] = {}
    node_offset = 0
    nodes = _get_allocated_nodes()
    for model_cfg in config.models:
        for _ in range(model_cfg.instances):
            npi = model_cfg.nodes_per_instance
            head_node = nodes[node_offset]
            port = config.port_start + node_port_counter.get(head_node, 0)
            node_port_counter[head_node] = node_port_counter.get(head_node, 0) + 1
            model_lookup[(head_node, port)] = model_cfg
            node_offset += npi

    for node, port in endpoints:
        model_cfg = model_lookup.get((node, port))
        metadata = {}
        if model_cfg:
            metadata["model"] = model_cfg.model
            metadata["tensor_parallel_size"] = model_cfg.tensor_parallel_size

        service = ServiceInfo(
            service_id=f"vllm-{node}-{port}",
            host=node,
            port=port,
            service_type="vllm",
            status=status,
            metadata=metadata,
        )
        registry.register_service(service)
        print(f"Registered {service.service_id} ({status}) with registry", file=sys.stderr)


def _update_instances_status(
    endpoints: list[tuple[str, int]],
    redis_host: str,
    redis_port: int,
    status: ServiceStatus,
) -> None:
    """Update the status of previously registered instances."""
    registry = ServiceRegistry(redis_host=redis_host, redis_port=redis_port)

    for node, port in endpoints:
        service_id = f"vllm-{node}-{port}"
        registry.update_health(service_id, status)
        print(f"Updated {service_id} -> {status.value}", file=sys.stderr)


def launch_instances(config: AegisConfig) -> None:
    """Launch vLLM instances across allocated nodes."""
    nodes = _get_allocated_nodes()
    total_nodes_needed = config.nodes_needed

    if len(nodes) < total_nodes_needed:
        print(
            f"Error: Need {total_nodes_needed} nodes but only {len(nodes)} allocated.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Start Redis service registry on the head node
    redis_host = _start_redis(config.redis_port)

    # Write temp files to the shared filesystem so remote nodes can access them.
    # PBS sets TMPDIR to a node-local path, so we use PBS_O_WORKDIR instead.
    shared_tmpdir = os.environ.get("PBS_O_WORKDIR", None)

    env = _get_template_env()
    template = env.get_template("instance.sh.j2")

    processes = []
    endpoints = []
    tmp_files = []
    node_port_counter: dict[str, int] = {}
    instance_idx = 0
    node_offset = 0

    for model_cfg in config.models:
        for j in range(model_cfg.instances):
            npi = model_cfg.nodes_per_instance
            instance_nodes = nodes[node_offset : node_offset + npi]
            head_node = instance_nodes[0]
            port = config.port_start + node_port_counter.get(head_node, 0)
            node_port_counter[head_node] = node_port_counter.get(head_node, 0) + 1
            endpoints.append((head_node, port))

            # Render the per-instance script
            script_content = template.render(
                model=model_cfg.model,
                tensor_parallel_size=model_cfg.tensor_parallel_size,
                port=port,
                hf_home=config.hf_home,
                extra_vllm_args=model_cfg.extra_vllm_args,
                conda_env=config.conda_env,
            )

            # Write to a temp file on the shared filesystem
            script_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".sh", prefix=f"aegis_instance_{instance_idx}_",
                dir=shared_tmpdir, delete=False,
            )
            script_file.write(script_content)
            script_file.close()
            os.chmod(script_file.name, 0o755)
            tmp_files.append(script_file.name)

            # Build hostfile for this instance's nodes
            hostfile = tempfile.NamedTemporaryFile(
                mode="w", suffix=".hosts", prefix=f"aegis_hosts_{instance_idx}_",
                dir=shared_tmpdir, delete=False,
            )
            for node in instance_nodes:
                hostfile.write(f"{node}\n")
            hostfile.close()
            tmp_files.append(hostfile.name)

            print(
                f"Launching instance {instance_idx} ({model_cfg.model}): "
                f"nodes={instance_nodes}, port={port}",
                file=sys.stderr,
            )

            proc = subprocess.Popen(
                [
                    "mpiexec",
                    "-ppn", "1",
                    "--hostfile", hostfile.name,
                    "-o", f"{instance_idx}/%h/out.%R",
                    script_file.name,
                ],
                env=os.environ,
            )
            processes.append(proc)
            instance_idx += 1
            node_offset += npi

    total_instances = instance_idx
    print(f"All {total_instances} instance(s) launched. Waiting for health checks...", file=sys.stderr)

    # Register instances as "starting" while they initialize
    _register_instances(endpoints, config, redis_host, config.redis_port,
                        status=ServiceStatus.STARTING.value)

    _wait_for_instances(endpoints)

    for path in tmp_files:
        os.unlink(path)

    # Update instances to "healthy" now that health checks passed
    _update_instances_status(endpoints, redis_host, config.redis_port,
                             ServiceStatus.HEALTHY)

    # Spawn a single heartbeat monitor on the head node that watches all
    # instances.  This replaces the previous per-node SSH approach and avoids
    # N SSH calls / N long-running remote processes.
    heartbeat_args = [
        sys.executable, "-m", "aegis.heartbeat",
        "--all", redis_host, str(config.redis_port),
    ]
    for node, port in endpoints:
        heartbeat_args.append(f"vllm-{node}-{port}:{node}:{port}")

    subprocess.Popen(
        heartbeat_args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"Started heartbeat monitor for {len(endpoints)} instance(s)", file=sys.stderr)

    print(f"All {total_instances} instance(s) are healthy.", file=sys.stderr)
    print(f"Redis service registry: {redis_host}:{config.redis_port}", file=sys.stderr)
