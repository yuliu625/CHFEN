import torch
import torch.nn as nn

import math

from pathlib import Path
from omegaconf import OmegaConf


class PositionalEncoding(nn.Module):
    """sequence部分的位置编码。"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

    def forward(self, x):
        pass


if __name__ == '__main__':
    pass
