"""Core orchestration: stage weights, launch vLLM instances."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from jinja2 import Environment, PackageLoader

from .config import AegisConfig, GPUS_PER_NODE


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


def stage_weights(config: AegisConfig) -> None:
    """Compile bcast (if needed) and broadcast model weights to local storage."""
    if not config.model_source:
        print("No model_source specified, skipping weight staging.", file=sys.stderr)
        return

    # Find the tools directory relative to the working directory
    tools_dir = Path(os.environ.get("PBS_O_WORKDIR", ".")) / "tools"
    bcast_bin = tools_dir / "bcast"

    if not bcast_bin.exists():
        print("Compiling bcast...", file=sys.stderr)
        result = subprocess.run(["make", "bcast"], cwd=tools_dir)
        if result.returncode != 0:
            print("Failed to compile bcast.", file=sys.stderr)
            sys.exit(1)

    dest = f"{config.hf_home}/hub"
    print(f"Staging weights: {config.model_source} -> {dest}", file=sys.stderr)

    env = os.environ.copy()
    env["MPIR_CVAR_CH4_OFI_ENABLE_MULTI_NIC_STRIPING"] = "1"
    env["MPIR_CVAR_CH4_OFI_MAX_NICS"] = "4"

    result = subprocess.run(
        ["mpiexec", "-ppn", "1", "--cpu-bind", "numa", str(bcast_bin), config.model_source, dest],
        env=env,
    )
    if result.returncode != 0:
        print("Weight staging failed.", file=sys.stderr)
        sys.exit(1)

    print("Weight staging complete.", file=sys.stderr)


def launch_instances(config: AegisConfig) -> None:
    """Launch vLLM instances across allocated nodes."""
    nodes = _get_allocated_nodes()
    nodes_per_instance = config.nodes_per_instance

    if len(nodes) < config.instances * nodes_per_instance:
        print(
            f"Error: Need {config.instances * nodes_per_instance} nodes but only {len(nodes)} allocated.",
            file=sys.stderr,
        )
        sys.exit(1)

    env = _get_template_env()
    template = env.get_template("instance.sh.j2")

    processes = []
    for i in range(config.instances):
        port = config.port_start + i
        start_node = i * nodes_per_instance
        instance_nodes = nodes[start_node : start_node + nodes_per_instance]

        # Render the per-instance script
        script_content = template.render(
            model=config.model,
            tensor_parallel_size=config.tensor_parallel_size,
            port=port,
            hf_home=config.hf_home,
            extra_vllm_args=config.extra_vllm_args,
        )

        # Write to a temp file
        script_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", prefix=f"aegis_instance_{i}_", delete=False
        )
        script_file.write(script_content)
        script_file.close()
        os.chmod(script_file.name, 0o755)

        # Build hostfile for this instance's nodes
        hostfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".hosts", prefix=f"aegis_hosts_{i}_", delete=False
        )
        for node in instance_nodes:
            hostfile.write(f"{node}\n")
        hostfile.close()

        print(
            f"Launching instance {i}: nodes={instance_nodes}, port={port}",
            file=sys.stderr,
        )

        proc = subprocess.Popen(
            [
                "mpiexec",
                "-ppn", "1",
                "--hostfile", hostfile.name,
                "-o", f"{i}/%h/out.%R",
                script_file.name,
            ],
            env=os.environ,
        )
        processes.append(proc)

    print(f"All {config.instances} instance(s) launched. Waiting...", file=sys.stderr)

    for proc in processes:
        proc.wait()
