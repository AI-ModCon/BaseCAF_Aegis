"""Broadcast pattern using MPI for co-located HTTP requests.

Each MPI rank runs on the same node as its vLLM server and makes a single
HTTP request to localhost:{port}.  MPI handles distribution and aggregation
— no remote connections needed.  This avoids the file-descriptor and
aiohttp connector limits that cap the async broadcast at large scale.

USAGE EXAMPLES:
---------------

Broadcast to all healthy instances:
    python examples/broadcast_mpi.py --registry-host <host>

Custom prompt and model:
    python examples/broadcast_mpi.py --registry-host <host> \\
        --prompt "Answer yes or no: is the sky blue?" \\
        --model meta-llama/Llama-3.1-70B-Instruct

Show failures:
    python examples/broadcast_mpi.py --registry-host <host> --show-failures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(msg: str) -> None:
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Worker mode — launched by mpiexec, one rank per node
# ---------------------------------------------------------------------------

def run_worker(args: argparse.Namespace) -> None:
    """SPMD worker: POST to localhost, gather results at rank 0, print on rank 0."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    url = f"http://localhost:{args.port}/v1/chat/completions"
    payload = json.dumps({
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
    }).encode()

    result: dict
    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            data = json.loads(resp.read().decode())
        text = data["choices"][0]["message"]["content"]
        result = {"success": True, "text": text, "error": None}
    except Exception as exc:
        result = {"success": False, "text": "", "error": str(exc)}

    # All ranks must call gather; only rank 0 receives results.
    all_results = comm.gather(result, root=0)

    if rank != 0:
        return

    # Build Response objects and run aggregators on rank 0.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Aurora-Swarm"))
    from aurora_swarm.aggregators import concat, failure_report, majority_vote
    from aurora_swarm.pool import Response

    responses = [
        Response(
            success=r["success"],
            text=r["text"],
            error=r["error"],
            agent_index=i,
        )
        for i, r in enumerate(all_results)
    ]

    winner, confidence = majority_vote(responses)
    print("\nMajority vote:")
    print(f"  Winner: {winner!r}")
    print(f"  Confidence: {confidence:.2f}")

    combined = concat(responses, separator=" | ")
    if len(combined) > args.max_concat_chars:
        combined = combined[: args.max_concat_chars] + "..."
    print("\nAll responses (concat):")
    print(f"  {combined}")

    if args.show_failures:
        report = failure_report(responses)
        _ts(
            f"Failure report — total: {report['total']}, "
            f"success: {report['success_count']}, "
            f"failures: {report['failure_count']}"
        )
        for f in report["failures"]:
            _ts(f"  Agent {f['agent_index']}: {f['error']}")


# ---------------------------------------------------------------------------
# Controller mode — run directly; queries registry and launches mpiexec
# ---------------------------------------------------------------------------

def run_controller(args: argparse.Namespace) -> int:
    """Query registry, write hostfile, launch mpiexec workers."""
    import shlex
    import subprocess

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from aegis.registry.service_registry import ServiceRegistryClient

    _ts("=" * 60)
    _ts("Broadcast MPI Example")
    _ts("=" * 60)

    client = ServiceRegistryClient(host=args.registry_host, port=args.registry_port)
    services = client.get_healthy_services(timeout_seconds=args.healthy_timeout)
    if not services:
        print(
            f"Error: no healthy services found in registry at "
            f"{args.registry_host}:{args.registry_port}",
            file=sys.stderr,
        )
        return 1

    _ts(
        f"Found {len(services)} healthy instance(s) from registry at "
        f"{args.registry_host}:{args.registry_port}"
    )

    bench_base = os.environ.get("PBS_O_WORKDIR", ".")
    hf = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="aegis_broadcast_mpi_hosts_",
        suffix=".txt",
        dir=bench_base,
        delete=False,
    )
    try:
        hf.write("\n".join(s.host for s in services) + "\n")
        hf.flush()
        hf.close()

        worker_cmd = [
            sys.executable, os.path.abspath(__file__),
            "--worker",
            "--port", str(args.port),
            "--prompt", args.prompt,
            "--model", args.model,
            "--max-tokens", str(args.max_tokens),
            "--timeout", str(args.timeout),
            "--max-concat-chars", str(args.max_concat_chars),
        ]
        if args.show_failures:
            worker_cmd.append("--show-failures")

        mpi_cmd = [
            "mpiexec",
            "-n", str(len(services)),
            "--hostfile", hf.name,
        ] + worker_cmd

        _ts(f"Launching: {shlex.join(mpi_cmd[:6])} ... ({len(services)} ranks)")
        proc = subprocess.run(mpi_cmd)
        return proc.returncode
    finally:
        try:
            os.unlink(hf.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    default_prompt = "Answer with one word: yes or no. Is the sky blue?"
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Registry (controller only)
    parser.add_argument(
        "--registry-host",
        default="localhost",
        help="Hostname of the Aegis registry (default: localhost)",
    )
    parser.add_argument(
        "--registry-port",
        type=int,
        default=8471,
        help="Port of the Aegis registry (default: 8471)",
    )
    parser.add_argument(
        "--healthy-timeout",
        type=int,
        default=30,
        help="Seconds since last heartbeat to consider a service healthy (default: 30)",
    )

    # Shared (controller passes these through to workers)
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port on each node (default: 8000)",
    )
    parser.add_argument(
        "--prompt",
        default=default_prompt,
        help=f"Prompt to broadcast (default: {default_prompt!r})",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-70B-Instruct",
        help="Model name (default: meta-llama/Llama-3.1-70B-Instruct)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens per response (default: 1024)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="Print failure report to stderr",
    )
    parser.add_argument(
        "--max-concat-chars",
        type=int,
        default=2000,
        help="Max characters to print for concat output (default: 2000)",
    )

    # Hidden flag — set by the controller when launching workers via mpiexec
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)

    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.worker:
        run_worker(args)
        return 0
    return run_controller(args)


if __name__ == "__main__":
    sys.exit(main())
