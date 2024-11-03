from .projection_layer import ProjectionLayer

import torch
import torch.nn as nn

from pathlib import Path
from omegaconf import OmegaConf


class ConditionedImageEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, encoder_config: OmegaConf):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


if __name__ == '__main__':
    pass
