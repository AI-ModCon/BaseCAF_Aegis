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
