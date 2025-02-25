from dataclasses import dataclass, field
from typing import List, Dict
from omegaconf import OmegaConf as om

class BaseConfig:
    @classmethod
    def load(cls, path: str):
        conf = om.load(path)
        return cls(**conf)

    def save(self, path: str):
        om.save(config=self, f=path)

    def asdict(self):
        return asdict(self)

@dataclass
class EINetParameters(BaseConfig):
    input_dim: int = 784
    excitatory_layer_sizes: List[int] = field(default_factory=lambda: [10])
    inhibitory_layer_sizes: List[int] = field(default_factory=lambda: [20])
    excitatory_branch_factors: List[int] = field(default_factory=lambda: [2, 2])
    inhibitory_branch_factors: List[int] = field(default_factory=lambda: [])
    ee_synapses_per_branch_per_layer: List[int] = field(default_factory=lambda: [24])
    ei_synapses_per_branch_per_layer: List[int] = field(default_factory=lambda: [200])
    ie_synapses_per_branch_per_layer: List[int] = field(default_factory=lambda: [1])
    ii_synapses_per_branch_per_layer: List[int] = field(default_factory=lambda: [])
    reactivate: bool = True
    somatic_synapses: bool = True
    topk_init_method: str = "xavier_normal"
    use_shunting: bool = True
    reactivation_type: str = "param_tanh"
    reactivation_strategy: str = "inverse"
    blocklinear_strategy: str = "block_conductance_dynamic"
    enable_branch_outputs: bool = True
    output_layer: bool = False
    output_dim: int = 10

@dataclass
class FeedForwardParameters(BaseConfig):
    input_dim: int = 784
    hidden_layer_sizes: List[int] = field(default_factory=lambda: [128, 64])
    activation: str = "relu"
    dropout: float = 0.2
    output_dim: int = 10

@dataclass
class ModelNetworkConfig(BaseConfig):
    type: str = "EINet"
    parameters: EINetParameters = field(default_factory=EINetParameters)

@dataclass
class ModelConfig(BaseConfig):
    task: str = "classification"
    probabilistic: bool = True
    network: ModelNetworkConfig = field(default_factory=ModelNetworkConfig)

@dataclass
class TrainConfig(BaseConfig):
    learning_strategy: str = "mle"
    run_name: str = "mnist_einet"
    seed: int = 0
    train_valid_split: float = 0.8
    lr: float = 0.01
    weight_decay_rate: float = 0.1
    epochs: int = 50
    batch_size: int = 128
    grad_clip_value: float = 5.0
    epochs_per_layer: int = 50
    pretrain_epochs: int = 50
    
@dataclass
class VisualizationConfig(BaseConfig):
    enabled: bool = True
    save_path: str = ""
    visualize_epochs: List[int] = field(default_factory=lambda: [])
    plots: List[str] = field(default_factory=lambda: ["branch_info", "ablation"])

@dataclass
class WandbConfig(BaseConfig):
    entity: str = ""
    project: str = ""
    group: str = ""
    tags: List[str] = field(default_factory=list)
    use_wandb: bool = False

@dataclass
class TaskConfig(BaseConfig):
    dataset: str = "mnist"
    data_path: str = ""
    parameters: Dict = field(default_factory=dict)

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    task: TaskConfig = field(default_factory=TaskConfig)

def load_config(path: str):
    base = Config()
    if not path:
        return base
    loaded = om.load(path)
    if "model" in loaded:
        if "task" in loaded["model"]:
            base.model.task = loaded["model"]["task"]
        if "probabilistic" in loaded["model"]:
            base.model.probabilistic = loaded["model"]["probabilistic"]
        if "network" in loaded["model"]:
            net_type = loaded["model"]["network"].get("type", "EINet")
            base.model.network.type = net_type
            if "parameters" in loaded["model"]["network"]:
                params = loaded["model"]["network"]["parameters"]
                for k, v in params.items():
                    setattr(base.model.network.parameters, k, v)
    if "train" in loaded:
        for k, v in loaded["train"].items():
            if hasattr(base.train, k):
                setattr(base.train, k, v)
    if "wandb" in loaded:
        for k, v in loaded["wandb"].items():
            if hasattr(base.wandb, k):
                setattr(base.wandb, k, v)
    if "visualization" in loaded:
        for k, v in loaded["visualization"].items():
            if hasattr(base.visualization, k):
                setattr(base.visualization, k, v)
    if "task" in loaded:
        if "dataset" in loaded["task"]:
            base.task.dataset = loaded["task"]["dataset"]
        if "parameters" in loaded["task"]:
            base.task.parameters = loaded["task"]["parameters"]
    return base