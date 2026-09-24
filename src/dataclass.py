"""Training configuration, context, and output dataclasses."""


from collections.abc import Callable
from dataclasses import dataclass
from typing import List, Tuple

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from .utils import save_callback


@dataclass
class CFG:
    device: int = 0
    threshold_value: float = 0.5
    epoch_size: int = 15
    epoch_period: int = 1
    learning_rate: float = 0.001
    threshold: float = 0.5

    model: nn.Module | None = None
    criterion: nn.Module | None = None
    optimizer: Optimizer | None = None
    scheduler: StepLR | None = None
    metric_func: Callable | None = None

    save_dir: str = "./runs"
    name: str = "baseline"
    callbacks: Tuple[Callable, ...] = (save_callback,)
    train_dataloader: DataLoader | None = None
    valid_dataloader: DataLoader | None = None
    test_dataloader: DataLoader | None = None

@dataclass
class TrainingContext:
    train_losses: List[float]
    valid_losses: List[float]
    metric_values: List[float]
    cfg: CFG | None = None
    epoch_id: int = None

@dataclass
class TrainOutput:
    train_loss: List[float]
    valid_loss: List[float]
    metrics: List[float]
    test_micro_f1: float = 0.0
    test_loss: float = 0.0
