## Overview

This repository provides guides and tools for deploying large language model inference on high-performance computing systems at ALCF and OLCF. The documentation covers multiple supercomputing platforms with their specific hardware configurations and software stacks.

## Supported Systems

### 🌟 Aurora (Intel)
- **Architecture:** Intel Xeon CPUs with Intel Data Center GPU Max Series
- **Framework:** vLLM with XPU backend
- **Key Features:** 12 GPU tiles per node, 64GB memory per tile
- **Documentation:** [Aurora/README.md](Aurora/README.md)

### 🔥 Frontier (AMD)
- **Architecture:** AMD EPYC CPUs with AMD Instinct GPUs
- **Framework:** vLLM with ROCm backend
- **Key Features:** 8 GPU tiles per node, 64GB memory per tile
- **Documentation:** [Frontier/README.md](Frontier/README.md)

## Quick Start

### Aurora Setup
```bash
# Load frameworks module with XPU-enabled vLLM
module load frameworks
export NUMEXPR_MAX_THREADS=208
export CCL_PROCESS_LAUNCHER=torchrun

# Configure model weights location
export HF_HOME="/flare/datasets/model-weights"
export HF_DATASETS_CACHE="/flare/datasets/model-weights"
export HF_TOKEN="YOUR_HF_TOKEN"

# Serve a small model (single tile)
vllm serve meta-llama/Llama-2-7b-chat-hf
```

### Frontier Setup
```bash
# Build ROCm vLLM container
apptainer build rocm-vllm.sif docker://rocm/vllm-dev:nightly

# Configure proxy settings
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export HF_TOKEN="YOUR_HF_TOKEN"

# Run in container
./rocm-vllm.sif vllm serve meta-llama/Llama-2-7b-chat-hf
```

## Model Categories

### Small Models (< 7B parameters)
- **Memory:** Fits in single tile (64GB)
- **Setup:** No distributed configuration needed
- **Examples:** Llama-2-7B, Mistral-7B

### Medium Models (7B-70B parameters)
- **Memory:** Requires multiple tiles on single node
- **Setup:** Tensor parallelism across tiles
- **Examples:** Llama-3.3-70B, Mixtral-8x7B

### Large Models (> 70B parameters)
- **Memory:** Requires multiple nodes
- **Setup:** Multi-node distributed inference
- **Status:** Coming soon...

## Tools and Utilities

### Data Staging
- **`tools/bcast.c`**: MPI-based tool for efficient data distribution across compute nodes
- **Usage**: Optimized for staging large model weights to local storage

### Job Scripts
- **`Aurora/start_instance.sh`**: Example script for launching vLLM instances on Aurora
- **`Aurora/vllm.pbs`**: PBS job configuration for batch submissions

## Hardware Specifications

| System | GPU Type | GPUs/Node | Memory/GPU | Interconnect |
|--------|-----------|------------|------------|--------------|
| Aurora | Intel Data Center GPU Max | 12 | 64GB | Slingshot |
| Frontier | AMD Instinct MI250X | 8 | 64GB | Slingshot |

## Performance Considerations

- **Tensor Parallelism**: Distribute model layers across multiple GPUs
- **Memory Management**: Utilize pre-cached model weights where available
- **Network Configuration**: Proper Ray cluster setup for multi-GPU inference
- **Environment Variables**: System-specific optimizations for each platform

## Contributing

This repository is actively maintained. Contributions welcome for:
- Additional system configurations
- Performance optimization tips
- New model serving examples
- Bug fixes and documentation improvements

## Support

For questions or issues:
1. Check system-specific documentation
2. Review example outputs in documentation
3. Contact LCF support teams for platform-specific issues

---

**Note**: Access to some models may require Hugging Face authentication tokens. Ensure proper token configuration before attempting to serve gated models.
