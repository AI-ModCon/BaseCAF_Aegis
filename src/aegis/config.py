"""Configuration loading and merging for Aegis."""

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import yaml


GPUS_PER_NODE = 12  # Aurora


@dataclass
class ModelConfig:
    """Per-model settings for a single model within a job."""
    model: str = ""
    instances: int = 1
    tensor_parallel_size: int = 1
    model_source: Optional[str] = None
    download_weights: bool = False
    extra_vllm_args: list[str] = field(default_factory=list)

    @property
    def nodes_per_instance(self) -> int:
        """Number of nodes each instance of this model spans."""
        return math.ceil(self.tensor_parallel_size / GPUS_PER_NODE)


@dataclass
class AegisConfig:
    # Per-model fields kept at top level for backward compat (single-model CLI usage)
    model: str = ""
    instances: int = 1
    tensor_parallel_size: int = 1
    model_source: Optional[str] = None
    download_weights: bool = False
    extra_vllm_args: list[str] = field(default_factory=list)

    # Global settings
    port_start: int = 8000
    hf_home: str = "/tmp/hf_home"
    hf_token: Optional[str] = None
    walltime: str = "01:00:00"
    queue: Optional[str] = None
    account: str = ""
    filesystems: str = "flare:home"
    conda_env: Optional[str] = None

    # In-process service registry (HTTP API port)
    registry_port: int = 8471

    # Startup timeout (seconds) for instances to become healthy
    startup_timeout: int = 600

    # Path to a conda environment containing the aegis package (for submit)
    aegis_env: Optional[str] = None

    # Output path for the endpoints file
    endpoints_file: str = "aegis_endpoints.txt"

    # Benchmark settings (used when --bench is passed to aegis submit)
    bench: bool = False
    bench_num_prompts: int = 100
    bench_output: Optional[str] = None

    # Multi-model list
    models: list[ModelConfig] = field(default_factory=list)

    @property
    def nodes_needed(self) -> int:
        """Calculate the number of nodes needed across all models."""
        return sum(
            m.instances * m.nodes_per_instance for m in self.models
        )


# Fields that live on ModelConfig (used for backward-compat conversion)
_MODEL_FIELDS = {"model", "instances", "tensor_parallel_size", "model_source", "download_weights", "extra_vllm_args"}


def _normalize_models(config: AegisConfig) -> None:
    """Ensure config.models is populated.

    If models is already set, leave it alone.
    Otherwise, convert top-level single-model fields into a one-entry models list.
    """
    if config.models:
        return

    if config.model:
        config.models = [
            ModelConfig(
                model=config.model,
                instances=config.instances,
                tensor_parallel_size=config.tensor_parallel_size,
                model_source=config.model_source,
                download_weights=config.download_weights,
                extra_vllm_args=list(config.extra_vllm_args),
            )
        ]


def load_config(path: str | Path) -> AegisConfig:
    """Load an AegisConfig from a YAML file."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    # Extract and parse models list separately
    raw_models = data.pop("models", [])
    model_configs = []
    for entry in raw_models:
        mc_fields = {k: v for k, v in entry.items() if k in {f.name for f in fields(ModelConfig)}}
        model_configs.append(ModelConfig(**mc_fields))

    # Map remaining YAML keys to AegisConfig fields (excluding 'models' which we handle above)
    valid_fields = {f.name for f in fields(AegisConfig)} - {"models"}
    filtered = {k: v for k, v in data.items() if k in valid_fields}

    config = AegisConfig(**filtered, models=model_configs)
    _normalize_models(config)
    return config


def merge_cli_args(config: AegisConfig, args) -> AegisConfig:
    """Overlay CLI arguments onto an existing config. CLI values take precedence."""
    for f in fields(AegisConfig):
        if f.name == "models":
            continue
        cli_val = getattr(args, f.name, None)
        if cli_val is not None:
            setattr(config, f.name, cli_val)
    return config


def config_to_yaml(config: AegisConfig) -> str:
    """Serialize an AegisConfig to YAML for embedding in PBS scripts."""
    data: dict = {}

    # Global settings (non-model fields)
    data["port_start"] = config.port_start
    data["hf_home"] = config.hf_home
    if config.hf_token:
        data["hf_token"] = config.hf_token
    data["walltime"] = config.walltime
    if config.queue:
        data["queue"] = config.queue
    data["account"] = config.account
    data["filesystems"] = config.filesystems
    if config.conda_env:
        data["conda_env"] = config.conda_env

    # Models list
    models_list = []
    for m in config.models:
        entry: dict = {"model": m.model}
        if m.instances != 1:
            entry["instances"] = m.instances
        if m.tensor_parallel_size != 1:
            entry["tensor_parallel_size"] = m.tensor_parallel_size
        if m.model_source:
            entry["model_source"] = m.model_source
        if m.download_weights:
            entry["download_weights"] = m.download_weights
        if m.extra_vllm_args:
            entry["extra_vllm_args"] = m.extra_vllm_args
        models_list.append(entry)
    data["models"] = models_list

    return yaml.dump(data, default_flow_style=False, sort_keys=False)
