# Configuration

Aegis uses YAML configuration files. All config values can also be overridden via CLI flags, which take precedence over the config file.

## Single Model Config

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

## Multi-Model Config

Launch different models within a single job allocation. Each model can have its own instance count, tensor-parallel size, weight source, and vLLM arguments. Ports are assigned per node starting from `port_start`, incrementing only for additional instances on the same node.

```yaml
port_start: 8000
hf_home: /tmp/hf_home
walltime: "01:00:00"
account: MyProject
filesystems: flare:home

models:
  - model: meta-llama/Llama-3.3-70B-Instruct
    instances: 2
    tensor_parallel_size: 6
    model_source: /flare/datasets/model-weights/hub/models--meta-llama--Llama-3.3-70B-Instruct
    extra_vllm_args:
      - --max-model-len
      - "32768"
  - model: meta-llama/Llama-3.1-8B-Instruct
    instances: 1
    tensor_parallel_size: 1
```

## Global Settings (AegisConfig)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `port_start` | `int` | `8000` | Base port for each node. Incremented for additional instances on the same node. |
| `hf_home` | `str` | `/tmp/hf_home` | Path used as `HF_HOME` for model weights. |
| `hf_token` | `str` | `None` | HuggingFace token for gated model access. |
| `walltime` | `str` | `01:00:00` | PBS job walltime. |
| `queue` | `str` | `None` | PBS queue name. |
| `account` | `str` | `""` | PBS account/project. Required for `submit`. |
| `filesystems` | `str` | `flare:home` | PBS filesystem directive. |
| `conda_env` | `str` | `None` | Path to a conda-pack tarball to distribute and activate on all nodes. |
| `registry_port` | `int` | `8471` | Port for the in-process service registry HTTP API. |
| `startup_timeout` | `int` | `600` | Seconds to wait for instances to become healthy. |
| `aegis_env` | `str` | `None` | Path to a conda environment containing the aegis package. Used by `submit` to activate the environment in the PBS script. If not set, `aegis submit` automatically detects the active conda environment it is running in. |
| `endpoints_file` | `str` | `aegis_endpoints.txt` | Output path for the file listing healthy instance endpoints. |
| `models` | `list` | `[]` | List of per-model configurations (see below). |

## Per-Model Settings (ModelConfig)

These fields can appear at the top level (single-model mode) or within entries in the `models` list (multi-model mode).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `""` | HuggingFace model name (e.g., `meta-llama/Llama-3.3-70B-Instruct`). |
| `instances` | `int` | `1` | Number of vLLM instances to launch for this model. |
| `tensor_parallel_size` | `int` | `1` | Number of GPUs per instance. |
| `model_source` | `str` | `None` | Source path for MPI broadcast weight staging. |
| `download_weights` | `bool` | `false` | Download model weights from HuggingFace Hub before staging. |
| `extra_vllm_args` | `list[str]` | `[]` | Additional arguments passed to `vllm serve`. |

### Computed Properties

- **`nodes_per_instance`** — Number of nodes each instance spans, calculated as `ceil(tensor_parallel_size / 12)` (12 GPUs per node on Aurora).
- **`nodes_needed`** — Total nodes needed across all models: `sum(instances * nodes_per_instance)`.

## Precedence

1. CLI flags (highest priority)
2. YAML config file
3. Default values (lowest priority)
