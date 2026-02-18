# CLI Reference

Aegis provides three subcommands: `submit`, `launch`, and `registry`.

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

Combine with `--wait` to submit remotely and block until endpoints are ready:

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
