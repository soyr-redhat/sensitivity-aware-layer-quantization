"""Configuration management for the project."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model loading configuration."""
    name: str = "mistralai/Mixtral-8x7B-v0.1"
    cache_dir: Optional[str] = None
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    # Quantization options
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    llm_int8_enable_fp32_cpu_offload: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_name: str = "openwebtext"
    num_samples: int = 1000
    max_length: int = 512
    seed: int = 42


@dataclass
class RoutingAnalysisConfig:
    """Routing analysis configuration."""
    output_dir: str = "outputs/routing_logs"
    save_frequency: int = 100


@dataclass
class AnalysisConfig:
    """General analysis configuration."""
    output_dir: str = "outputs/analysis"
    plot_dir: str = "outputs/plots"


@dataclass
class Config:
    """Main configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    routing_analysis: RoutingAnalysisConfig = field(default_factory=RoutingAnalysisConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            routing_analysis=RoutingAnalysisConfig(**config_dict.get('routing_analysis', {})),
            analysis=AnalysisConfig(**config_dict.get('analysis', {}))
        )

    def save_yaml(self, path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'routing_analysis': self.routing_analysis.__dict__,
            'analysis': self.analysis.__dict__
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
