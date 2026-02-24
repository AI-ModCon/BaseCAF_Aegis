# Aegis: HPC LLM Instance Launcher

Aegis automates launching configurable numbers of vLLM inference instances on HPC clusters. It handles PBS job generation, model weight staging via MPI broadcast, and per-instance orchestration.

Currently targets **Aurora** (PBS). Frontier/Slurm support is planned.

## Quick Install

```bash
pip install .
```

For development:

```bash
pip install -e .
```

## How It Works

1. **`aegis submit`** renders a PBS batch script from a Jinja2 template and submits it via `qsub`. The generated job script calls `aegis launch` inside the allocation.
2. **`aegis launch`** runs inside a PBS allocation. It optionally stages model weights to local storage using MPI broadcast, then launches one `vllm serve` process per instance on the assigned nodes.
3. **`aegis bench`** benchmarks running instances in parallel via `mpiexec` and aggregates results into CSV.
4. **`aegis shutdown`** tears down instances by killing vLLM processes on nodes and/or cancelling PBS jobs.

## Architecture

```
src/aegis/
├── cli.py              # CLI entry point (argparse)
├── config.py           # Config file loading + merging with CLI args
├── scheduler.py        # PBS job generation and submission
├── launcher.py         # Core orchestration: stage weights, launch instances
└── templates/
    ├── pbs_job.sh.j2   # Jinja2 template for PBS batch script
    └── instance.sh.j2  # Jinja2 template for per-node vLLM launch script
```

## Next Steps

- [Getting Started](getting-started.md) — Installation and first launch walkthrough
- [CLI Reference](cli.md) — All commands and flags
- [Configuration](configuration.md) — YAML config format and all options
- [Platforms](platforms/aurora.md) — Platform-specific setup guides
