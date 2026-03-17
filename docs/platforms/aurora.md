# Aurora

Aurora is an Intel Data Center GPU Max (Ponte Vecchio) system at the Argonne Leadership Computing Facility, managed with PBS.

## Example Config

Serve Llama 3.3 70B on Aurora with tensor parallelism across 8 GPU tiles:

```yaml
model: meta-llama/Llama-3.3-70B-Instruct
instances: 2
tensor_parallel_size: 8
model_source: /flare/datasets/model-weights/hub/models--meta-llama--Llama-3.3-70B-Instruct
walltime: "01:00:00"
account: MyProject
filesystems: flare:home
extra_vllm_args:
  - --max-model-len
  - "32768"
```

## Submitting Jobs

Submit from an Aurora login node:

```bash
aegis submit --config config.yaml
```

Submit from your laptop via SSH:

```bash
aegis submit --config config.yaml --remote user@aurora.alcf.anl.gov
```

Submit and wait for endpoints to be ready:

```bash
aegis submit --config config.yaml --wait
```

These flags can be combined — see [CLI Reference](../cli.md) for the full list.

## vLLM Availability

vLLM is pre-installed on Aurora compute nodes via `module load frameworks`. Alternatively, distribute a custom environment with the `--conda-env` option (see [Getting Started](../getting-started.md#staging-a-conda-environment)).

## Frameworks Module Workaround

The `frameworks` module on Aurora has a bug that crashes vLLM during model inspection for certain architectures. Aegis automatically runs `tools/vllm_build_all_modelinfo_caches.py` on every node before launching a model to pre-populate vLLM's model-info caches and avoid this codepath. No user action is required.

## MPI Broadcast Example

`examples/broadcast_mpi.py` demonstrates large-scale inference using MPI for co-located HTTP requests. Each MPI rank runs on the same node as its vLLM server and makes a single request to `localhost:{port}`, avoiding the file-descriptor and connection limits of async approaches:

```bash
python examples/broadcast_mpi.py \
    --registry-url http://node01:8471 \
    --prompt "Answer yes or no: is the sky blue?" \
    --model meta-llama/Llama-3.1-70B-Instruct
```

The script launches one MPI rank per healthy instance discovered from the registry, gathers all responses, and prints majority-vote and concatenated results.
