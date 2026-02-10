# Inference with vLLM on Frontier

Until there are proven working instructions for running vLLM natively on Frontier nodes, we recommend using the ROCm vLLM Docker images published in the vLLM DockerHub organization.

## Build and run ROCm vLLM image
```bash
apptainer build vllm-openai-rocm.sif docker://vllm/vllm-openai-rocm
apptainer shell vllm-openai-rocm.sif
vllm --version
```
## Example output from Frontier compute node

```console
Apptainer> vllm --version
0.11.1rc6.dev141+g0b8e871e5.rocm700
```

## Access Model Weights

To my knowledge there are no common models stored on the Frontier filesystem for community use. It may be a good idea to do so for whatever ModCon allocation we receive. In any case, we recommend using Frontier's Burst Buffer storage for staging model weights.

When downloading models from Hugging Face you will need to set your http proxy settings for outbound access. Be aware that some gated models also require additional authentication. To access these gated models, you will need a [Hugging Face authentication token](https://huggingface.co/docs/hub/en/security-tokens).

```bash
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'
export HF_TOKEN=<your_token>
```

## Serve Small Models

For small models that fit within a single GPU's memory (64 GB), no additional configuration is required to serve the model. The defaultconfiguration ensures the model is run on a single tile without the need for distributed setup. Models with fewer than 7 billion parameters typically fit within a single tile.

#### Using Single Tile

The following command serves `meta-llama/Llama-2-7b-chat-hf` on a single tile of a single node:
```bash linenums="1"
vllm serve meta-llama/Llama-2-7b-chat-hf
```

<details>
<summary>Click for example output</summary>

```console
<tbd>
```

</details>

#### Using Multiple Tiles

To utilize multiple tiles for larger models (`TP>1`), a more advanced setup is necessary. First, configure a Ray cluster.
```bash linenums="1"
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
unset ROCM_VISIBLE_DEVICES # vLLM throws an error if it sees this envvar
ray start --head --node-ip-address=$VLLM_HOST_IP --num-cpus=128 --num-gpus=8 &
```

The following script demonstrates how to serve the `meta-llama/Llama-2-7b-chat-hf` model across 8 tiles on a single node:

```bash linenums="1"
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
unset ROCM_VISIBLE_DEVICES
ray start --head --node-ip-address=$VLLM_HOST_IP --num-cpus=128 --num-gpus=8 &
vllm serve meta-llama/Llama-2-7b-chat-hf --port 8000 --tensor-parallel-size 8 --trust-remote-code
```

## Serve Medium Models

#### Using Single Node

The following script demonstrates how to serve `meta-llama/Llama-3.3-70B-Instruct` on 8 tiles on a single node. Models with up to 70 billion parameters can usually fit within a single node, utilizing multiple tiles.

```bash linenums="1"
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
unset ROCM_VISIBLE_DEVICES # vLLM throws an error if it sees this envvar
ray start --head --node-ip-address=$VLLM_HOST_IP --num-cpus=128 --num-gpus=8 &
vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 8 --trust-remote-code --max-model-len 32768
```

## Serve Large Models

#### Using Multiple Nodes

coming soon...
