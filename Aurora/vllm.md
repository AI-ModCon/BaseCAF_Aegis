# Inference with vLLM on Aurora

## Load frameworks module with XPU-enabled vLLM installation

```bash linenums="1"
module load frameworks
export NUMEXPR_MAX_THREADS=208 # number of CPU threads on Aurora node
export CCL_PROCESS_LAUNCHER=torchrun
vllm --version
```

## Example output from Aurora compute node

```console
raffenet@x4311c1s4b0n0:~> module load frameworks
(/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0) raffenet@x4311c1s4b0n0:~> vllm --version
[W1106 16:55:04.833350173 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
    registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/aten/src/ATen/VmapModeRegistrations.cpp:37
       new kernel: registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/ipex_2.8.10_xpu_rel_08_18_2025/intel-extension-for-pytorch/build/Release/csrc/gpu/csrc/gpu/xpu/ATen/RegisterXPU_0.cpp:172 (function operator())
[2025-11-06 16:55:04,586] [INFO] [real_accelerator.py:260:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2025-11-06 16:55:06,888] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
INFO 11-06 16:55:07 [__init__.py:241] Automatically detected platform xpu.
0.10.1rc2.dev189+ge2db1164a.xpu
```

## Access Model Weights

Model weights for commonly used open-weight models are downloaded and available in the `/flare/datasets/model-weights/hub` directory on Aurora. To ensure your workflows utilize the preloaded model weights and datasets, update the following environment variables in your session. Some models hosted on Hugging Face may be gated, requiring additional authentication. To access these gated models, you will need a [Hugging Face authentication token](https://huggingface.co/docs/hub/en/security-tokens).
```bash linenums="1"
export HF_HOME="/flare/datasets/model-weights"
export HF_DATASETS_CACHE="/flare/datasets/model-weights"
export HF_MODULES_CACHE="/flare/datasets/model-weights"
export HF_TOKEN="YOUR_HF_TOKEN"
export RAY_TMPDIR="/tmp"
export TMPDIR="/tmp"
```

## Serve Small Models

For small models that fit within a single tile's memory (64 GB), no additional configuration is required to serve the model. Simply set `TP=1` (Tensor Parallelism). This configuration ensures the model is run on a single tile without the need for distributed setup. Models with fewer than 7 billion parameters typically fit within a single tile.

#### Using Single Tile

The following command serves `meta-llama/Llama-2-7b-chat-hf` on a single tile of a single node:
```bash linenums="1"
vllm serve meta-llama/Llama-2-7b-chat-hf
```

<details>
<summary>Click for example output</summary>

```console
(/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0) raffenet@x4516c6s1b0n0:~> vllm serve meta-llama/Llama-2-7b-chat-hf

[W1106 18:40:40.729878143 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
    registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/aten/src/ATen/VmapModeRegistrations.cpp:37
       new kernel: registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/ipex_2.8.10_xpu_rel_08_18_2025/intel-extension-for-pytorch/build/Release/csrc/gpu/csrc/gpu/xpu/ATen/RegisterXPU_0.cpp:172 (function operator())
[2025-11-06 18:40:40,292] [INFO] [real_accelerator.py:260:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2025-11-06 18:40:42,596] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
INFO 11-06 18:40:42 [__init__.py:241] Automatically detected platform xpu.
(APIServer pid=206609) INFO 11-06 18:40:44 [api_server.py:1873] vLLM API server version 0.10.1rc2.dev189+ge2db1164a
(APIServer pid=206609) INFO 11-06 18:40:44 [utils.py:326] non-default args: {'model_tag': 'meta-llama/Llama-2-7b-chat-hf', 'model': 'meta-llama/Llama-2-7b-chat-hf'}
(APIServer pid=206609) Could not cache non-existence of file. Will ignore error and continue. Error: [Errno 13] Permission denied: '/flare/datasets/model-weights/hub/models--meta-llama--Llama-2-7b-chat-hf/.no_exist/f5db02db724555f92da89c216ac04704f23d4590/preprocessor_config.json'
(APIServer pid=206609) ERROR:huggingface_hub.file_download:Could not cache non-existence of file. Will ignore error and continue. Error: [Errno 13] Permission denied: '/flare/datasets/model-weights/hub/models--meta-llama--Llama-2-7b-chat-hf/.no_exist/f5db02db724555f92da89c216ac04704f23d4590/preprocessor_config.json'
(APIServer pid=206609) INFO 11-06 18:40:53 [__init__.py:742] Resolved architecture: LlamaForCausalLM
(APIServer pid=206609) `torch_dtype` is deprecated! Use `dtype` instead!
(APIServer pid=206609) INFO 11-06 18:40:53 [__init__.py:1786] Using max model len 4096
(APIServer pid=206609) INFO 11-06 18:40:53 [scheduler.py:222] Chunked prefill is enabled with max_num_batched_tokens=2048.
(APIServer pid=206609) INFO 11-06 18:40:53 [xpu.py:113] [XPU] CUDA graph is not supported on XPU, disabling cudagraphs.
[W1106 18:40:59.700780685 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
    registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/aten/src/ATen/VmapModeRegistrations.cpp:37
       new kernel: registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/ipex_2.8.10_xpu_rel_08_18_2025/intel-extension-for-pytorch/build/Release/csrc/gpu/csrc/gpu/xpu/ATen/RegisterXPU_0.cpp:172 (function operator())
[2025-11-06 18:40:59,263] [INFO] [real_accelerator.py:260:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2025-11-06 18:41:01,390] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
INFO 11-06 18:41:01 [__init__.py:241] Automatically detected platform xpu.
(EngineCore_0 pid=207574) INFO 11-06 18:41:02 [core.py:644] Waiting for init message from front-end.
(EngineCore_0 pid=207574) INFO 11-06 18:41:02 [core.py:74] Initializing a V1 LLM engine (v0.10.1rc2.dev189+ge2db1164a) with config: model='meta-llama/Llama-2-7b-chat-hf', speculative_config=None, tokenizer='meta-llama/Llama-2-7b-chat-hf', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=4096, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=True, quantization=None, enforce_eager=False, kv_cache_dtype=auto, device_config=xpu, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=meta-llama/Llama-2-7b-chat-hf, enable_prefix_caching=True, chunked_prefill_enabled=True, use_async_output_proc=True, pooler_config=None, compilation_config={"level":0,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":null,"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"cudagraph_mode":0,"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"pass_config":{},"max_capture_size":512,"local_cache_dir":null}
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
(EngineCore_0 pid=207574) INFO 11-06 18:41:03 [parallel_state.py:1134] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
2025:11:06-18:41:03:(207574) |CCL_WARN| value of CCL_ATL_TRANSPORT changed to be ofi (default:mpi)
2025:11:06-18:41:03:(207574) |CCL_WARN| value of CCL_OP_SYNC changed to be 1 (default:0)
2025:11:06-18:41:03:(207574) |CCL_WARN| value of CCL_PROCESS_LAUNCHER changed to be torchrun (default:hydra)
(EngineCore_0 pid=207574) INFO 11-06 18:41:03 [gpu_model_runner.py:1964] Starting to load model meta-llama/Llama-2-7b-chat-hf...
(EngineCore_0 pid=207574) INFO 11-06 18:41:03 [gpu_model_runner.py:1996] Loading model from scratch...
(EngineCore_0 pid=207574) INFO 11-06 18:41:03 [xpu.py:45] Using Flash Attention backend on V1 engine.
(EngineCore_0 pid=207574) INFO 11-06 18:41:04 [weight_utils.py:294] Using model weights format ['*.safetensors']
(EngineCore_0 pid=207574) Ignored error while writing commit hash to /flare/datasets/model-weights/hub/models--meta-llama--Llama-2-7b-chat-hf/refs/main: [Errno 13] Permission denied: '/flare/datasets/model-weights/hub/models--meta-llama--Llama-2-7b-chat-hf/refs/main'.
(EngineCore_0 pid=207574) WARNING:huggingface_hub._snapshot_download:Ignored error while writing commit hash to /flare/datasets/model-weights/hub/models--meta-llama--Llama-2-7b-chat-hf/refs/main: [Errno 13] Permission denied: '/flare/datasets/model-weights/hub/models--meta-llama--Llama-2-7b-chat-hf/refs/main'.
Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.01s/it]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:04<00:00,  2.23s/it]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:04<00:00,  2.05s/it]
(EngineCore_0 pid=207574)
(EngineCore_0 pid=207574) INFO 11-06 18:41:08 [default_loader.py:267] Loading weights took 4.35 seconds
(EngineCore_0 pid=207574) INFO 11-06 18:41:08 [gpu_model_runner.py:2018] Model loading took 12.5533 GiB and 5.050575 seconds
(EngineCore_0 pid=207574) INFO 11-06 18:41:09 [xpu_worker.py:104] Before memory profiling run, total GPU memory: 65520.00 MB, model load takes 12870.74 MB, free gpu memory is 52435.44 MB.
(EngineCore_0 pid=207574) INFO 11-06 18:41:10 [xpu_worker.py:139] After memory profiling run, peak memory usage is 13552.69 MB,torch mem is 12870.74 MB, non-torch mem is 426.94 MB, free gpu memory is 51896.32 MB.
(EngineCore_0 pid=207574) INFO 11-06 18:41:10 [kv_cache_utils.py:849] GPU KV cache size: 90,816 tokens
(EngineCore_0 pid=207574) INFO 11-06 18:41:10 [kv_cache_utils.py:853] Maximum concurrency for 4,096 tokens per request: 22.17x
(EngineCore_0 pid=207574) WARNING 11-06 18:41:10 [_logger.py:68] Skipping CUDA graph capture. To turn on CUDA graph capture, ensure `cudagraph_mode` was not manually set to `NONE`
(EngineCore_0 pid=207574) INFO 11-06 18:41:10 [core.py:215] init engine (profile, create kv cache, warmup model) took 1.44 seconds
(APIServer pid=206609) INFO 11-06 18:41:10 [loggers.py:142] Engine 000: vllm cache_config_info with initialization after num_gpu_blocks is: 1419
(APIServer pid=206609) INFO 11-06 18:41:10 [async_llm.py:165] Torch profiler disabled. AsyncLLM CPU traces will not be collected.
(APIServer pid=206609) INFO 11-06 18:41:10 [api_server.py:1679] Supported_tasks: ['generate']
(APIServer pid=206609) WARNING 11-06 18:41:10 [logger.py:71] Default sampling parameters have been overridden by the model's Hugging Face generation config recommended from the model creator. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
(APIServer pid=206609) INFO 11-06 18:41:10 [serving_responses.py:124] Using default chat sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
(APIServer pid=206609) INFO 11-06 18:41:10 [serving_chat.py:135] Using default chat sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
(APIServer pid=206609) INFO 11-06 18:41:10 [serving_completion.py:77] Using default completion sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
(APIServer pid=206609) INFO 11-06 18:41:10 [api_server.py:1948] Starting vLLM API server 0 on http://0.0.0.0:8000
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:36] Available routes are:
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /openapi.json, Methods: GET, HEAD
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /docs, Methods: GET, HEAD
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /docs/oauth2-redirect, Methods: GET, HEAD
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /redoc, Methods: GET, HEAD
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /health, Methods: GET
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /load, Methods: GET
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /ping, Methods: POST
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /ping, Methods: GET
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /tokenize, Methods: POST
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /detokenize, Methods: POST
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /v1/models, Methods: GET
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /version, Methods: GET
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /v1/responses, Methods: POST
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /v1/chat/completions, Methods: POST
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /v1/completions, Methods: POST
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /v1/embeddings, Methods: POST
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /pooling, Methods: POST
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /classify, Methods: POST
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /score, Methods: POST
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /v1/score, Methods: POST
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /v1/audio/transcriptions, Methods: POST
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /v1/audio/translations, Methods: POST
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /rerank, Methods: POST
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /v1/rerank, Methods: POST
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /v2/rerank, Methods: POST
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /invocations, Methods: POST
(APIServer pid=206609) INFO 11-06 18:41:10 [launcher.py:44] Route: /metrics, Methods: GET
(APIServer pid=206609) INFO:     Started server process [206609]
(APIServer pid=206609) INFO:     Waiting for application startup.
(APIServer pid=206609) INFO:     Application startup complete.
```

</details>

#### Using Multiple Tiles

To utilize multiple tiles for larger models (`TP>1`), a more advanced setup is necessary. First, configure a Ray cluster.
```bash linenums="1"
export VLLM_HOST_IP=$(getent hosts $(hostname).hsn.cm.aurora.alcf.anl.gov | awk '{ print $1 }' | tr ' ' '\n' | sort | head -n 1)
unset ONEAPI_DEVICE_SELECTOR # allow Ray to access all 12 GPU tiles
ray start --head --node-ip-address=$VLLM_HOST_IP --num-cpus=96 --num-gpus=12 &
export no_proxy="localhost,127.0.0.1" # Set no_proxy for the client to interact with the locally hosted model
```

The following script demonstrates how to serve the `meta-llama/Llama-2-7b-chat-hf` model across 8 tiles on a single node:

```bash linenums="1"
export VLLM_HOST_IP=$(getent hosts $(hostname).hsn.cm.aurora.alcf.anl.gov | awk '{ print $1 }' | tr ' ' '\n' | sort | head -n 1)
vllm serve meta-llama/Llama-2-7b-chat-hf --port 8000 --tensor-parallel-size 8 --trust-remote-code
```

## Serve Medium Models

#### Using Single Node

The following script demonstrates how to serve `meta-llama/Llama-3.3-70B-Instruct` on 8 tiles on a single node. Models with up to 70 billion parameters can usually fit within a single node, utilizing multiple tiles.

```bash linenums="1"
export VLLM_HOST_IP=$(getent hosts $(hostname).hsn.cm.aurora.alcf.anl.gov | awk '{ print $1 }' | tr ' ' '\n' | sort | head -n 1)
unset ONEAPI_DEVICE_SELECTOR # allow Ray to access all 12 GPU tiles
ray start --head --node-ip-address=$VLLM_HOST_IP --num-cpus=96 --num-gpus=12 &
vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 8 --trust-remote-code --max-model-len 32768
```

## Serve Large Models

#### Using Multiple Nodes

coming soon...
