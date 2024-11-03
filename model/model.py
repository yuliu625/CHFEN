from embedding import TotalEncoder
# from .feature_level import
# from .decision_level import
from .projection_layer import ProjectionLayer

import torch
import torch.nn as nn

from pathlib import Path
from omegaconf import OmegaConf


class CHFEN(nn.Module):
    def __init__(self):
        super().__init__()
        self.total_encoder = TotalEncoder()

    def forward(self, x):
        pass


if __name__ == '__main__':
    pass
