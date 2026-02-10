# Aegis: HPC LLM Instance Launcher

Aegis automates launching configurable numbers of vLLM inference instances on HPC clusters. It handles PBS job generation, model weight staging via MPI broadcast, and per-instance orchestration.

Currently targets **Aurora** (PBS). Frontier/Slurm support planned.

## Installation

```bash
pip install .
```

For development:

```bash
pip install -e .
```

## Usage

### Submit a PBS job

Generate and submit a batch job to the PBS queue:

```bash
aegis submit --config config.yaml
```

Preview the generated PBS script without submitting:

```bash
aegis submit --config config.yaml --dry-run
```

### Launch inside an existing allocation

If you already have a PBS allocation (e.g., via `qsub -I`), launch instances directly:

```bash
aegis launch --config config.yaml
```

### CLI flags

All config values can be overridden via CLI flags. CLI flags take precedence over the config file.

```bash
aegis submit \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --instances 2 \
    --tensor-parallel-size 6 \
    --account MyProject \
    --walltime 01:00:00 \
    --model-source /flare/datasets/model-weights/hub/models--meta-llama--Llama-3.3-70B-Instruct \
    --dry-run
```

## Example Config (YAML)

```yaml
model: meta-llama/Llama-3.3-70B-Instruct
instances: 2
tensor_parallel_size: 6
port_start: 8000
hf_home: /tmp/hf_home
model_source: /flare/datasets/model-weights/hub/models--meta-llama--Llama-3.3-70B-Instruct
walltime: "01:00:00"
account: MyProject
filesystems: flare:home
extra_vllm_args:
  - --max-model-len
  - "32768"
```

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

### How it works

1. **`aegis submit`** renders a PBS batch script from a Jinja2 template and submits it via `qsub`. The generated job script calls `aegis launch` inside the allocation.
2. **`aegis launch`** runs inside a PBS allocation. It optionally stages model weights to local storage using the MPI broadcast tool (`tools/bcast.c`), then launches one `vllm serve` process per instance on the assigned nodes.

## Platform Documentation

Reference documentation for manual setup on each platform:

- [Aurora](Aurora/README.md) — Intel Data Center GPU Max, PBS
- [Frontier](Frontier/README.md) — AMD Instinct MI250X, Slurm

## Tools

- **`tools/bcast.c`** — MPI-based tool for efficiently broadcasting model weights to all compute nodes via `tar` streaming.
