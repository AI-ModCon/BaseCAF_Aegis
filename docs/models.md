# Tested Models

The following models have been tested with Aegis on Aurora.

## Summary

| Model | Parameters | TP Size | Instances Tested | Max Nodes | Platform |
|-------|-----------|---------|-----------------|-----------|----------|
| Llama 3.1 8B Instruct | 8B | 1 | up to 1024 | 1024 | Aurora |
| Llama 3.3 70B Instruct | 70B | 8 | up to 1024 | 1024 | Aurora |
| OpenAI gpt-oss-120b | 120B | 8 | up to 1024 | 1024 | Aurora |

## Model Details

### Llama 3.1 8B Instruct

- **Model**: `meta-llama/Llama-3.1-8B-Instruct`
- **Tensor parallel size**: 1 (fits on a single GPU tile)
- **Nodes per instance**: 1
- **Tested at scale**: Up to 1024 instances on 1024 nodes

```yaml
model: meta-llama/Llama-3.1-8B-Instruct
instances: 1024
tensor_parallel_size: 1
```

### Llama 3.3 70B Instruct

- **Model**: `meta-llama/Llama-3.3-70B-Instruct`
- **Tensor parallel size**: 8
- **Nodes per instance**: 1
- **Tested at scale**: Up to 1024 instances on 1024 nodes

```yaml
model: meta-llama/Llama-3.3-70B-Instruct
instances: 1024
tensor_parallel_size: 8
model_source: /flare/datasets/model-weights/hub/models--meta-llama--Llama-3.3-70B-Instruct
extra_vllm_args:
  - --max-model-len
  - "32768"
```

### OpenAI gpt-oss-120b

- **Model**: `OpenAI/gpt-oss-120b`
- **Tensor parallel size**: 8
- **Nodes per instance**: 1
- **Tested at scale**: Up to 1024 instances on 1024 nodes

```yaml
model: OpenAI/gpt-oss-120b
instances: 1024
tensor_parallel_size: 8
```

## Scalability

All three models have been tested on Aurora at scales up to **1024 nodes**. Aegis handles the orchestration of weight staging and instance launch across all nodes automatically via MPI broadcast and PBS job management.

Key observations:

- Weight staging via MPI broadcast scales efficiently to 1024 nodes
- The service registry tracks instance health across all nodes
- Port assignment is handled automatically per node, with `port_start` incrementing for additional instances on the same node
