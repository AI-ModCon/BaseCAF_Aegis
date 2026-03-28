"""Broadcast pattern example with aggregation strategies.

This script queries the Aegis service registry for healthy vLLM instances,
broadcasts the same prompt to all of them except one (configurable by index),
then demonstrates two aggregation strategies: majority_vote (categorical
consensus) and concat (combined text).

USAGE EXAMPLES:
---------------

Broadcast to all healthy instances (exclude index 0 by default):
    python examples/broadcast_aggregators.py --registry-host <host>

Override the registry port:
    python examples/broadcast_aggregators.py --registry-host <host> --registry-port 9000

Exclude a specific index with a custom prompt:
    python examples/broadcast_aggregators.py --registry-host <host> \\
        --exclude-index 2 --prompt "Reply with one word: left or right."
"""

import argparse
import asyncio
import sys
from datetime import datetime

from aegis.registry import ServiceRegistryClient

from aurora_swarm import AgentEndpoint, VLLMPool
from aurora_swarm.aggregators import concat, failure_report, majority_vote
from aurora_swarm.patterns.broadcast import broadcast


def print_with_timestamp(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr)


async def main() -> int:
    default_prompt = "Answer with one word: yes or no. Is the sky blue?"
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    parser.add_argument(
        "--exclude-index",
        type=int,
        default=0,
        help="Agent index to exclude from the broadcast (default: 0)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=default_prompt,
        help=f"Prompt to broadcast (default: {default_prompt!r})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens per response (default: 1024)",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-70B-Instruct",
        help="Model name (default: meta-llama/Llama-3.1-70B-Instruct)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Maximum concurrent requests (default: 64)",
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
    args = parser.parse_args()

    print_with_timestamp("=" * 60)
    print_with_timestamp("Broadcast + Aggregators Example")
    print_with_timestamp("=" * 60)

    # Discover healthy endpoints from the Aegis registry.
    client = ServiceRegistryClient(host=args.registry_host, port=args.registry_port)
    services = client.get_healthy_services(timeout_seconds=args.healthy_timeout)
    if not services:
        print(
            f"Error: no healthy services found in registry at "
            f"{args.registry_host}:{args.registry_port}",
            file=sys.stderr,
        )
        return 1

    endpoints = [AgentEndpoint(host=s.host, port=s.port) for s in services]
    print_with_timestamp(
        f"Found {len(endpoints)} healthy endpoint(s) from registry at "
        f"{args.registry_host}:{args.registry_port}"
    )

    if args.exclude_index < 0 or args.exclude_index >= len(endpoints):
        print(
            f"Error: --exclude-index must be in [0, {len(endpoints) - 1}]",
            file=sys.stderr,
        )
        return 1

    indices = [i for i in range(len(endpoints)) if i != args.exclude_index]
    pool = VLLMPool(
        endpoints,
        model=args.model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        concurrency=args.concurrency,
    )
    sub_pool = None
    try:
        sub_pool = pool.select(indices)
        print_with_timestamp(
            f"Broadcasting to {sub_pool.size} agents (excluding index {args.exclude_index})"
        )
        prompt_preview = args.prompt[:80] + "..." if len(args.prompt) > 80 else args.prompt
        print_with_timestamp(f"Prompt: {prompt_preview}")
        responses = await broadcast(sub_pool, args.prompt)
    finally:
        if sub_pool is not None:
            await sub_pool.close()
        await pool.close()
        await asyncio.sleep(0)  # yield so aiohttp cleanup can run

    print_with_timestamp(f"Received {len(responses)} responses")

    # Aggregation strategy 1: majority vote
    winner, confidence = majority_vote(responses)
    print("\nMajority vote:")
    print(f"  Winner: {winner!r}")
    print(f"  Confidence: {confidence:.2f}")

    report = failure_report(responses)
    if report["success_count"] == 0 and report["failure_count"] > 0:
        print_with_timestamp(
            f"  (No successful responses; {report['failure_count']} failure(s). "
            "Use --show-failures for details.)"
        )

    # Aggregation strategy 2: concat
    combined = concat(responses, separator=" | ")
    if len(combined) > args.max_concat_chars:
        combined_display = combined[: args.max_concat_chars] + "..."
    else:
        combined_display = combined
    print("\nAll responses (concat):")
    print(f"  {combined_display}")

    if args.show_failures:
        print_with_timestamp("\nFailure report:")
        print_with_timestamp(
            f"  Total: {report['total']}, Success: {report['success_count']}, "
            f"Failures: {report['failure_count']}"
        )
        for f in report["failures"]:
            print_with_timestamp(f"  Agent {f['agent_index']}: {f['error']}")

    print_with_timestamp("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
