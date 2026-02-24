# Getting Started

## Prerequisites

- Python 3.9+
- Access to an HPC cluster with PBS (Aurora) or Slurm (Frontier, planned)
- vLLM (can be installed as part of a conda environment staged by Aegis via the `--conda-env` option, or available on compute nodes via `module load frameworks` on Aurora)

## Installation

Install from the repository root:

```bash
pip install .
```

For development (editable install):

```bash
pip install -e .
```

Verify the installation:

```bash
aegis --help
```

### Installing in a conda environment

Create a standalone conda environment for Aegis:

```bash
conda create -n aegis python=3.11 -y
conda activate aegis
pip install .
```

Aegis is a launcher — it does not require vLLM itself. vLLM only needs to be available on the compute nodes, either via a system module (e.g., `module load frameworks` on Aurora) or a separate conda environment distributed with `--conda-env`. See [Staging a Conda Environment](#staging-a-conda-environment) below.

## First Launch

### 1. Create a config file

Create a file called `config.yaml`:

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

### 2. Preview the generated PBS script

Use `--dry-run` to see what would be submitted without actually submitting:

```bash
aegis submit --config config.yaml --dry-run
```

### 3. Submit the job

```bash
aegis submit --config config.yaml
```

This generates a PBS batch script and submits it via `qsub`. The job will call `aegis launch` inside the allocation to stage weights and start vLLM instances.

### 4. Launch inside an existing allocation

If you already have a PBS allocation (e.g., via `qsub -I`), launch instances directly:

```bash
aegis launch --config config.yaml
```

### 5. Query running instances

Once instances are running, use the service registry to discover them:

```bash
aegis registry list
aegis registry list-healthy
```

### 6. Benchmark running instances

Once instances are running, benchmark them:

```bash
aegis bench --model meta-llama/Llama-3.3-70B-Instruct
```

This runs `vllm bench serve` in parallel across all endpoints listed in `aegis_endpoints.txt` and prints a CSV summary. Write results to a file with `--output`:

```bash
aegis bench --model meta-llama/Llama-3.3-70B-Instruct --output results.csv
```

### 7. Shut down instances

When you're done, shut down the vLLM processes:

```bash
aegis shutdown
```

This reads `aegis_endpoints.txt` to find the nodes and kills vLLM processes on each one. To also cancel the PBS job:

```bash
aegis shutdown --job-id 12345.aurora-pbs-0001
```

If you submitted remotely, cancel the job via SSH:

```bash
aegis shutdown --job-id 12345.aurora-pbs-0001 --remote user@aurora.alcf.anl.gov
```

## Staging a Conda Environment

If your compute nodes don't have vLLM pre-installed, you can distribute a custom conda environment to all nodes using [conda-pack](https://conda.github.io/conda-pack/). First, create a tarball from an existing conda environment:

```bash
conda pack -n my_vllm_env -o my_vllm_env.tar.gz
```

Then reference it in your config file:

```yaml
model: meta-llama/Llama-3.3-70B-Instruct
instances: 2
tensor_parallel_size: 6
conda_env: /path/to/my_vllm_env.tar.gz
walltime: "01:00:00"
account: MyProject
filesystems: flare:home
```

Or pass it as a CLI flag:

```bash
aegis submit --config config.yaml --conda-env /path/to/my_vllm_env.tar.gz
```

Aegis will broadcast the tarball to all compute nodes and activate the environment before launching vLLM instances.

## Overriding Config via CLI

All config values can be overridden with CLI flags. CLI flags take precedence over the config file:

```bash
aegis submit \
    --config config.yaml \
    --instances 4 \
    --walltime 02:00:00 \
    --dry-run
```

See [CLI Reference](cli.md) for all available flags and [Configuration](configuration.md) for the full config format.
