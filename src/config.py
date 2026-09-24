"""Baseline experiment configuration."""


import torch
import torch.nn as nn
import torchvision
from torcheval.metrics.functional import binary_f1_score
from torch.optim.lr_scheduler import StepLR

from .dataclass import CFG


def create_baseline_cfg(
    train_dataloader,
    valid_dataloader,
    test_dataloader,
    name="baseline",
):
    cfg = CFG()

    cfg.name=name

    cfg.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    cfg.model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    )
    cfg.model.fc = nn.Linear(cfg.model.fc.in_features, 1)

    cfg.criterion = torch.nn.BCEWithLogitsLoss()

    cfg.optimizer = torch.optim.Adam(
        cfg.model.parameters(),
        lr=cfg.learning_rate,
    )

    cfg.scheduler = StepLR(
        cfg.optimizer,
        step_size=5,
        gamma=0.1,
    )

    cfg.metric_func = binary_f1_score

    cfg.train_dataloader = train_dataloader
    cfg.valid_dataloader = valid_dataloader
    cfg.test_dataloader = test_dataloader

    return cfg
