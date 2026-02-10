"""Configuration loading and merging for Aegis."""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import yaml


GPUS_PER_NODE = 12  # Aurora


@dataclass
class AegisConfig:
    model: str = ""
    instances: int = 1
    tensor_parallel_size: int = 1
    port_start: int = 8000
    hf_home: str = "/tmp/hf_home"
    hf_token: Optional[str] = None
    model_source: Optional[str] = None
    walltime: str = "01:00:00"
    queue: Optional[str] = None
    account: str = ""
    filesystems: str = "flare:home"
    extra_vllm_args: list[str] = field(default_factory=list)

    @property
    def nodes_needed(self) -> int:
        """Calculate the number of nodes needed for all instances."""
        import math
        gpus_total = self.instances * self.tensor_parallel_size
        return math.ceil(gpus_total / GPUS_PER_NODE)

    @property
    def gpus_per_instance(self) -> int:
        return self.tensor_parallel_size

    @property
    def nodes_per_instance(self) -> int:
        """Number of nodes each instance spans."""
        import math
        return math.ceil(self.tensor_parallel_size / GPUS_PER_NODE)


def load_config(path: str | Path) -> AegisConfig:
    """Load an AegisConfig from a YAML file."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    # Map YAML keys to dataclass fields
    valid_fields = {f.name for f in fields(AegisConfig)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return AegisConfig(**filtered)


def merge_cli_args(config: AegisConfig, args) -> AegisConfig:
    """Overlay CLI arguments onto an existing config. CLI values take precedence."""
    for f in fields(AegisConfig):
        cli_val = getattr(args, f.name, None)
        if cli_val is not None:
            setattr(config, f.name, cli_val)
    return config
