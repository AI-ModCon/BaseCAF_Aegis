"""Core orchestration: stage weights, launch vLLM instances."""

import os
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path

from jinja2 import Environment, PackageLoader

from .config import AegisConfig, GPUS_PER_NODE, ModelConfig


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


def _download_hf_weights(config: AegisConfig) -> None:
    """Download model weights from HuggingFace Hub for models that request it."""
    models_to_download = [
        m for m in config.models if m.download_weights and not m.model_source
    ]
    if not models_to_download:
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "Error: huggingface_hub is required for --download-weights.\n"
            "Install it with: pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)

    for m in models_to_download:
        print(f"Downloading weights from HuggingFace Hub: {m.model}", file=sys.stderr)
        downloaded_path = snapshot_download(m.model, cache_dir=config.hf_home)
        print(f"Downloaded {m.model} to {downloaded_path}", file=sys.stderr)
        m.model_source = downloaded_path


def stage_weights(config: AegisConfig) -> None:
    """Compile bcast (if needed) and broadcast model weights to local storage."""
    _download_hf_weights(config)

    if not any(m.model_source for m in config.models):
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

    for m in config.models:
        if not m.model_source:
            continue
        source = m.model_source
        bcast_cmd = ["mpiexec", "-ppn", "1", "--cpu-bind", "numa", str(bcast_bin)]
        if m.download_weights:
            bcast_cmd.append("--no-root-write")
        bcast_cmd.extend([source, dest])
        print(f"Staging weights: {source} -> {dest}", file=sys.stderr)
        result = subprocess.run(bcast_cmd, env=env)
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

    # Spawn the heartbeat process which creates the in-memory registry,
    # registers all instances as STARTING, and serves the HTTP query API.
    head_node = nodes[0]
    heartbeat_args = [
        sys.executable, "-m", "aegis.heartbeat",
        "--registry-port", str(config.registry_port),
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

    _wait_for_instances(endpoints)

    for path in tmp_files:
        os.unlink(path)

    # The heartbeat loop will naturally transition instances from STARTING to
    # HEALTHY once their /health endpoints respond 200 — no separate update needed.

    # Write an endpoints file so users/scripts can discover healthy instances.
    endpoints_file = Path("aegis_endpoints.txt")
    with open(endpoints_file, "w") as f:
        for node, port in endpoints:
            f.write(f"{node}:{port}\n")

    print(f"All {total_instances} instance(s) are healthy.", file=sys.stderr)
    print(f"Endpoints written to {endpoints_file.resolve()}", file=sys.stderr)
    print(f"Service registry: http://{head_node}:{config.registry_port}", file=sys.stderr)
