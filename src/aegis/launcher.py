"""Core orchestration: stage weights, launch vLLM instances."""

import math
import os
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path

from jinja2 import Environment, PackageLoader

from .config import AegisConfig, GPUS_PER_NODE, ModelConfig

_verbose = False


def set_verbose(flag: bool) -> None:
    global _verbose
    _verbose = flag


def _vlog(*args) -> None:
    if _verbose:
        print(*args, file=sys.stderr)


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


def _ensure_bcast() -> Path:
    """Compile bcast if the binary is missing or the source is newer."""
    tools_dir = _project_root() / "tools"
    bcast_bin = tools_dir / "bcast"
    bcast_src = tools_dir / "bcast.c"

    needs_build = (
        not bcast_bin.exists()
        or (bcast_src.exists() and bcast_src.stat().st_mtime > bcast_bin.stat().st_mtime)
    )

    if needs_build:
        print("Compiling bcast...", file=sys.stderr)
        _vlog(f"  [bcast] make bcast (cwd={tools_dir})")
        result = subprocess.run(["make", "bcast"], cwd=tools_dir)
        if result.returncode != 0:
            print("Failed to compile bcast.", file=sys.stderr)
            sys.exit(1)

    return bcast_bin


def stage_conda_env(config: AegisConfig) -> float | None:
    """Compile bcast (if needed) and broadcast a conda-pack tarball to all nodes.

    Returns the elapsed time in seconds, or None if staging was skipped.
    """
    if not config.conda_env:
        return None

    bcast_bin = _ensure_bcast()

    tarball = config.conda_env
    print(f"Staging conda env: {tarball} -> /tmp", file=sys.stderr)

    env = os.environ.copy()
    env["MPIR_CVAR_CH4_OFI_ENABLE_MULTI_NIC_STRIPING"] = "1"
    env["MPIR_CVAR_CH4_OFI_MAX_NICS"] = "4"

    cmd = ["mpiexec", "-ppn", "1", "--cpu-bind", "numa", str(bcast_bin), tarball, "/tmp"]
    _vlog(f"  [conda-stage] {shlex.join(cmd)}")
    t0 = time.monotonic()
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print("Conda env staging failed.", file=sys.stderr)
        sys.exit(1)
    elapsed = time.monotonic() - t0
    print(f"Conda env staging complete ({elapsed:.1f}s).", file=sys.stderr)
    return elapsed


def stage_apptainer_image(config: AegisConfig) -> float | None:
    """Compile bcast (if needed) and broadcast an Apptainer image to all nodes.

    Returns the elapsed time in seconds, or None if staging was skipped.
    """
    if not config.apptainer_image:
        return None

    bcast_bin = _ensure_bcast()
    image = config.apptainer_image
    print(f"Staging apptainer image: {image} -> /tmp", file=sys.stderr)

    env = os.environ.copy()
    env["MPIR_CVAR_CH4_OFI_ENABLE_MULTI_NIC_STRIPING"] = "1"
    env["MPIR_CVAR_CH4_OFI_MAX_NICS"] = "4"

    cmd = ["mpiexec", "-ppn", "1", "--cpu-bind", "numa", str(bcast_bin), image, "/tmp"]
    _vlog(f"  [apptainer-stage] {shlex.join(cmd)}")
    t0 = time.monotonic()
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print("Apptainer image staging failed.", file=sys.stderr)
        sys.exit(1)
    elapsed = time.monotonic() - t0
    print(f"Apptainer image staging complete ({elapsed:.1f}s).", file=sys.stderr)
    return elapsed


def _download_hf_weights(config: AegisConfig) -> None:
    """Download model weights from HuggingFace Hub.

    Two cases trigger a download:
    - model_source is set but the path does not yet exist on disk: weights are
      downloaded directly into model_source so the same config can be reused on
      future runs without re-downloading.
    - download_weights is True and no model_source is set: weights are
      downloaded into hf_home and model_source is set to the resulting path.
    """
    needs_download = [
        m for m in config.models
        if (m.model_source and not Path(m.model_source).exists())
        or (m.download_weights and not m.model_source)
    ]
    if not needs_download:
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "Error: huggingface_hub is required for weight downloading.\n"
            "Install it with: pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)

    token = config.hf_token or os.environ.get("HF_TOKEN")

    for m in needs_download:
        if m.model_source and not Path(m.model_source).exists():
            print(
                f"model_source '{m.model_source}' not found. "
                f"Downloading {m.model} from HuggingFace Hub...",
                file=sys.stderr,
            )
            snapshot_download(m.model, local_dir=m.model_source, token=token)
            print(f"Downloaded {m.model} to {m.model_source}", file=sys.stderr)
        else:
            # download_weights=True, no model_source: download into hf_home cache
            print(f"Downloading weights from HuggingFace Hub: {m.model}", file=sys.stderr)
            downloaded_path = snapshot_download(m.model, cache_dir=config.hf_home, token=token)
            print(f"Downloaded {m.model} to {downloaded_path}", file=sys.stderr)
            m.model_source = downloaded_path


def stage_weights(config: AegisConfig) -> float | None:
    """Compile bcast (if needed) and broadcast model weights to local storage.

    Returns the total elapsed time in seconds, or None if staging was skipped.
    """
    _download_hf_weights(config)

    if not any(m.model_source for m in config.models):
        print("No model_source specified, skipping weight staging.", file=sys.stderr)
        return None

    bcast_bin = _ensure_bcast()

    dest = f"{config.hf_home}/hub"
    env = os.environ.copy()
    env["MPIR_CVAR_CH4_OFI_ENABLE_MULTI_NIC_STRIPING"] = "1"
    env["MPIR_CVAR_CH4_OFI_MAX_NICS"] = "4"

    t_total = time.monotonic()
    for m in config.models:
        if not m.model_source:
            continue
        source = m.model_source
        bcast_cmd = ["mpiexec", "-ppn", "1", "--cpu-bind", "numa", str(bcast_bin)]
        if m.download_weights:
            bcast_cmd.append("--no-root-write")
        bcast_cmd.extend([source, dest])
        print(f"Staging weights: {source} -> {dest}", file=sys.stderr)
        _vlog(f"  [weight-stage] {shlex.join(bcast_cmd)}")
        t0 = time.monotonic()
        result = subprocess.run(bcast_cmd, env=env)
        if result.returncode != 0:
            print(f"Weight staging failed for {source}.", file=sys.stderr)
            sys.exit(1)
        elapsed = time.monotonic() - t0
        print(f"Weight staging complete for {source} ({elapsed:.1f}s).", file=sys.stderr)

    total_elapsed = time.monotonic() - t_total
    print("All weight staging complete.", file=sys.stderr)
    return total_elapsed


def _wait_for_instances(
    endpoints: list[tuple[str, int]],
    poll_interval: int = 10,
    timeout: int = 600,
) -> list[tuple[str, int]]:
    """Poll /health on each instance until all respond 200 or timeout expires.

    Returns the list of endpoints that became healthy within the timeout.
    """
    # Bypass http_proxy env vars — health checks target internal compute nodes.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    ready = set()
    ready_times: dict[tuple[str, int], float] = {}
    start = time.monotonic()
    deadline = start + timeout
    cycle = 0

    while time.monotonic() < deadline:
        cycle += 1
        newly_ready = []
        for node, port in endpoints:
            if (node, port) in ready:
                continue
            url = f"http://{node}:{port}/health"
            try:
                with opener.open(url, timeout=5) as resp:
                    if resp.status == 200:
                        elapsed = time.monotonic() - start
                        ready.add((node, port))
                        ready_times[(node, port)] = elapsed
                        newly_ready.append((node, port, elapsed))
            except (urllib.error.URLError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                _vlog(f"  [poll {cycle}] {node}:{port} not ready: {reason}")

        for node, port, elapsed in newly_ready:
            print(
                f"Instance {node}:{port} ready in {elapsed:.1f}s "
                f"({len(ready)}/{len(endpoints)})",
                file=sys.stderr,
            )

        if len(ready) == len(endpoints):
            break

        pending = len(endpoints) - len(ready)
        elapsed_total = time.monotonic() - start
        print(
            f"[poll {cycle}] {len(ready)}/{len(endpoints)} ready, "
            f"{pending} pending ({elapsed_total:.0f}s elapsed)",
            file=sys.stderr,
        )
        time.sleep(poll_interval)

    not_ready = [(n, p) for n, p in endpoints if (n, p) not in ready]
    if not_ready:
        print(
            f"Warning: Timed out after {timeout}s waiting for instances: "
            + ", ".join(f"{n}:{p}" for n, p in not_ready),
            file=sys.stderr,
        )

    if ready_times:
        print("Instance startup times:", file=sys.stderr)
        for node, port in endpoints:
            if (node, port) in ready_times:
                print(f"  {node}:{port}  {ready_times[(node, port)]:.1f}s", file=sys.stderr)
            else:
                print(f"  {node}:{port}  timed out", file=sys.stderr)
        times = list(ready_times.values())
        avg = sum(times) / len(times)
        print(f"  Average: {avg:.1f}s (min {min(times):.1f}s, max {max(times):.1f}s)", file=sys.stderr)

    return [(n, p) for n, p in endpoints if (n, p) in ready]


def launch_instances(
    config: AegisConfig,
    staging_times: dict[str, float] | None = None,
) -> None:
    """Launch vLLM instances across allocated nodes."""
    nodes = _get_allocated_nodes()
    total_nodes_needed = config.nodes_needed

    if len(nodes) < total_nodes_needed:
        print(
            f"Error: Need {total_nodes_needed} nodes but only {len(nodes)} allocated.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create a log directory on the launch node under $TMPDIR.
    log_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "aegis-logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Write temp files to the shared filesystem so remote nodes can access them.
    # PBS sets TMPDIR to a node-local path, so we use PBS_O_WORKDIR instead.
    shared_tmpdir = os.environ.get("PBS_O_WORKDIR", None)

    env = _get_template_env()
    template = env.get_template("instance.sh.j2")

    endpoints = []
    endpoint_models: dict[tuple[str, int], str] = {}
    tmp_files = []
    instance_idx = 0
    node_offset = 0
    all_instance_nodes = []

    for model_cfg in config.models:
        port = config.port_start
        script_content = template.render(
            model=model_cfg.model,
            tensor_parallel_size=model_cfg.tensor_parallel_size,
            port=port,
            hf_home=config.hf_home,
            extra_vllm_args=model_cfg.extra_vllm_args,
            conda_env=config.conda_env,
            apptainer_image=config.apptainer_image,
            modelinfo_cache_script=str(
                _project_root() / "tools" / "vllm_build_all_modelinfo_caches.py"
            ),
        )
        script_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", prefix=f"aegis_model_{model_cfg.model.replace('/', '_')}_",
            dir=shared_tmpdir, delete=False,
        )
        script_file.write(script_content)
        script_file.close()
        os.chmod(script_file.name, 0o755)
        tmp_files.append(script_file.name)

        for j in range(model_cfg.instances):
            node = nodes[node_offset]
            endpoints.append((node, port))
            endpoint_models[(node, port)] = model_cfg.model
            all_instance_nodes.append((node, script_file.name, instance_idx))
            _vlog(f"  [instance {instance_idx}] {model_cfg.model} node={node} port={port}")
            instance_idx += 1
            node_offset += 1

    # One mpiexec per model config, each with its own hostfile.
    # This supports multiple models with different scripts running concurrently.
    model_nodes: dict[str, list[str]] = {}
    for node, script, _ in all_instance_nodes:
        model_nodes.setdefault(script, []).append(node)

    for script, nodes_for_model in model_nodes.items():
        hostfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".hosts", prefix="aegis_hosts_",
            dir=shared_tmpdir, delete=False,
        )
        for node in nodes_for_model:
            hostfile.write(f"{node}\n")
        hostfile.close()
        tmp_files.append(hostfile.name)

        mpi_launch_cmd = [
            "mpiexec", "-ppn", "1",
            "--hostfile", hostfile.name,
            "-o", f"{log_dir}/%r/%h/out.%R",
            script,
        ]
        _vlog(f"  [mpiexec] {shlex.join(mpi_launch_cmd)}")
        subprocess.Popen(mpi_launch_cmd, env=os.environ)

    total_instances = instance_idx
    t_wait_start = time.monotonic()
    print(f"Launched {total_instances} instance(s), waiting for health checks...", file=sys.stderr)

    # Spawn the heartbeat process which creates the in-memory registry,
    # registers all instances as STARTING, and serves the HTTP query API.
    head_node = nodes[0]
    heartbeat_args = [
        sys.executable, "-m", "aegis.heartbeat",
        "--registry-port", str(config.registry_port),
    ]
    for node, port in endpoints:
        model = endpoint_models[(node, port)]
        heartbeat_args.append(f"vllm-{node}-{port}:{node}:{port}:{model}")

    heartbeat_log = open(log_dir / "heartbeat.log", "w")
    _vlog(f"  [heartbeat] {shlex.join(heartbeat_args)}")
    subprocess.Popen(
        heartbeat_args,
        stdin=subprocess.DEVNULL,
        stdout=heartbeat_log,
        stderr=heartbeat_log,
    )
    print(
        f"Heartbeat monitor started. Registry: http://{head_node}:{config.registry_port}  Logs: {log_dir}",
        file=sys.stderr,
    )

    # Write the registry URL sidecar file early so the submit side can find it
    # as soon as the registry is available, even before instances are healthy.
    registry_url = f"http://{head_node}:{config.registry_port}"
    registry_file = Path(config.endpoints_file).parent / "aegis_registry_url.txt"
    with open(registry_file, "w") as f:
        f.write(registry_url + "\n")

    healthy = _wait_for_instances(endpoints, timeout=config.startup_timeout)

    for path in tmp_files:
        os.unlink(path)

    if not healthy:
        print("Error: No instances became healthy.", file=sys.stderr)
        sys.exit(1)

    # The heartbeat loop will naturally transition instances from STARTING to
    # HEALTHY once their /health endpoints respond 200 — no separate update needed.

    # Write an endpoints file with only the healthy instances.
    endpoints_file = Path(config.endpoints_file)
    with open(endpoints_file, "w") as f:
        for node, port in healthy:
            f.write(f"{node}:{port}\n")

    total_wait = time.monotonic() - t_wait_start
    total_staging = sum(staging_times.values()) if staging_times else 0.0

    if staging_times:
        print("Staging times:", file=sys.stderr)
        labels = {"conda_env": "conda env", "apptainer_image": "apptainer image", "weights": "weights"}
        for key, label in labels.items():
            t = staging_times.get(key)
            if t is not None:
                print(f"  {label + ':':<20} {t:.1f}s", file=sys.stderr)
        print(f"  {'total:':<20} {total_staging:.1f}s", file=sys.stderr)

    print(
        f"{len(healthy)}/{total_instances} instance(s) are healthy "
        f"(total wait: {total_wait + total_staging:.1f}s).",
        file=sys.stderr,
    )
    print(f"Endpoints written to {endpoints_file.resolve()}", file=sys.stderr)
