# CLI Reference

Aegis provides five subcommands: `submit`, `launch`, `registry`, `bench`, and `shutdown`.

## `aegis submit`

Generate and submit a PBS batch job.

```bash
aegis submit --config config.yaml
```

### Submit-specific flags

| Flag | Type | Description |
|------|------|-------------|
| `--dry-run` | flag | Print the generated PBS script without submitting |
| `--aegis-env` | `str` | Path to a conda environment containing the aegis package |
| `--wait` | flag | Block until instances are healthy and the endpoints file is written |
| `--remote` | `str` | Submit via SSH to a remote login node (e.g., `user@aurora.alcf.anl.gov`) |

### `--wait`

By default `aegis submit` exits immediately after `qsub` succeeds. Pass `--wait` to block until the endpoints file appears:

```bash
aegis submit --config config.yaml --wait
```

The command polls the PBS job with `qstat` and watches for the configured `endpoints_file`. Once the file is found and non-empty the endpoints are printed to stdout and the command exits. If the job terminates before the file appears, an error is printed and the command exits with code 1.

### `--remote`

Run `aegis submit` from a machine that does not have `qsub` available (e.g., your laptop) by tunnelling commands over SSH:

```bash
aegis submit --config config.yaml --remote user@aurora.alcf.anl.gov
```

ALCF login nodes require a one-time password (OTP). Aegis opens an SSH `ControlMaster` session that prompts once for the OTP and reuses the connection for all subsequent operations (file copy, `qsub`, polling). The control socket is cleaned up on exit.

Combine with `--wait` to submit remotely and block until endpoints are ready. The endpoints file is automatically copied to the current working directory:

```bash
aegis submit --config config.yaml --remote user@aurora.alcf.anl.gov --wait
```

### Common flags (shared with `launch`)

| Flag | Type | Description |
|------|------|-------------|
| `--config` | `str` | Path to YAML config file |
| `--model` | `str` | HuggingFace model name |
| `--instances` | `int` | Number of vLLM instances to launch |
| `--tensor-parallel-size` | `int` | Number of GPUs per instance |
| `--port-start` | `int` | Base port for each node (incremented for additional instances on the same node) |
| `--hf-home` | `str` | Path to model weights |
| `--hf-token` | `str` | HuggingFace token |
| `--model-source` | `str` | Source path for bcast weight staging |
| `--walltime` | `str` | PBS walltime |
| `--queue` | `str` | PBS queue name |
| `--account` | `str` | PBS account/project |
| `--filesystems` | `str` | PBS filesystem directive |
| `--download-weights` | flag | Download model weights from HuggingFace Hub before staging |
| `--extra-vllm-args` | `str...` | Additional arguments passed to `vllm serve` |
| `--registry-port` | `int` | Port for the service registry HTTP API (default: 8471) |
| `--conda-env` | `str` | Path to a conda-pack tarball to distribute and activate on all nodes |
| `--startup-timeout` | `int` | Seconds to wait for instances to become healthy (default: 600) |
| `--endpoints-file` | `str` | Output path for the endpoints file (default: `aegis_endpoints.txt`) |

## `aegis launch`

Launch vLLM instances inside an existing PBS allocation. Stages model weights and starts `vllm serve` processes on assigned nodes.

```bash
aegis launch --config config.yaml
```

### Launch-specific flags

| Flag | Description |
|------|-------------|
| `--skip-staging` | Skip conda env and weight staging (use when already staged) |

All [common flags](#common-flags-shared-with-launch) listed above are also available.

## `aegis registry`

Query the service registry to discover running vLLM instances.

### Common registry flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--registry-host` | `str` | `localhost` | Registry server host |
| `--registry-port` | `int` | `8471` | Registry server port |
| `--format` | `text\|json` | `text` | Output format |

### `aegis registry list`

List all registered services.

```bash
aegis registry list [--type TYPE] [--status STATUS]
```

| Flag | Type | Description |
|------|------|-------------|
| `--type` | `str` | Filter by service type |
| `--status` | `str` | Filter by status |

### `aegis registry get`

Get a single service by ID.

```bash
aegis registry get SERVICE_ID
```

### `aegis registry list-healthy`

List services that are currently healthy (recent heartbeat).

```bash
aegis registry list-healthy [--type TYPE] [--timeout SECONDS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--type` | `str` | | Filter by service type |
| `--timeout` | `int` | `30` | Heartbeat timeout in seconds |

### `aegis registry count`

Count registered services.

```bash
aegis registry count [--type TYPE]
```

| Flag | Type | Description |
|------|------|-------------|
| `--type` | `str` | Filter by service type |

## `aegis bench`

Benchmark launched vLLM instances using `vllm bench serve`. Aegis runs benchmarks in parallel across all endpoints via `mpiexec`, then aggregates results into a CSV summary.

```bash
aegis bench --model meta-llama/Llama-3.3-70B-Instruct
```

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | `str` | *required* | Model name for the benchmark |
| `--num-prompts` | `int` | `100` | Number of prompts per endpoint |
| `--endpoints-file` | `str` | `aegis_endpoints.txt` | Path to endpoints file |
| `--output` | `str` | stdout | Path to write CSV results |
| `--conda-env` | `str` | | Path to staged conda environment directory |
| `--registry-host` | `str` | `localhost` | Registry server host (use to discover endpoints from registry instead of file) |
| `--registry-port` | `int` | `8471` | Registry server port |
| `--format` | `text\|json` | `text` | Output format |

Extra arguments after `--` are passed through to `vllm bench serve`:

```bash
aegis bench --model meta-llama/Llama-3.3-70B-Instruct -- --dataset-name random --random-input-len 512 --random-output-len 128
```

### Endpoint discovery

By default, `aegis bench` reads endpoints from the file specified by `--endpoints-file`. If `--registry-host` is set to something other than `localhost`, it queries the service registry for healthy endpoints instead.

### Conda environment

If your compute nodes use a staged conda environment for vLLM, pass `--conda-env` so that each benchmark rank activates the environment before running:

```bash
aegis bench --model meta-llama/Llama-3.3-70B-Instruct --conda-env /tmp/conda_env
```

### Output

Results are printed as CSV to stdout (or to a file with `--output`). Each row corresponds to one endpoint, with a `SUMMARY` row showing min/max/mean across all endpoints. Metrics include request throughput, token throughput, TTFT, TPOT, ITL, and more.

## `aegis shutdown`

Shut down launched vLLM instances and/or cancel PBS jobs. Supports two modes that can be used independently or combined.

```bash
aegis shutdown
```

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--endpoints-file` | `str` | `aegis_endpoints.txt` | Path to endpoints file |
| `--job-id` | `str` | | PBS job ID to cancel via `qdel` |
| `--remote` | `str` | | Run `qdel` via SSH on a remote login node (e.g., `user@aurora.alcf.anl.gov`) |

### Process kill (default)

When the endpoints file exists, `aegis shutdown` reads it to discover which nodes are running vLLM instances, then uses `mpiexec` to run `pkill -f "vllm serve"` on each node:

```bash
aegis shutdown --endpoints-file aegis_endpoints.txt
```

### PBS job cancel

Use `--job-id` to cancel a PBS job via `qdel`, which kills all child processes:

```bash
aegis shutdown --job-id 12345.aurora-pbs-0001
```

### Combined

Kill processes on nodes first, then cancel the PBS job:

```bash
aegis shutdown --endpoints-file aegis_endpoints.txt --job-id 12345.aurora-pbs-0001
```

### Remote cancel

If `qdel` is not available locally, use `--remote` to run it via SSH:

```bash
aegis shutdown --job-id 12345.aurora-pbs-0001 --remote user@aurora.alcf.anl.gov
```
