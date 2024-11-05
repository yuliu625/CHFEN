import torch
import torch.nn as nn

import math

from pathlib import Path
from omegaconf import OmegaConf


class TypeEncoding(nn.Module):
    """
    每个种类的具体embedding。
    实例化后是不同的。
    """
    def __init__(self, embedding_dim=768):
        super().__init__()
        self.type_encoding = nn.Parameter(torch.randn(1, embedding_dim))

    def forward(self):
        return self.type_encoding


if __name__ == '__main__':
    pass
