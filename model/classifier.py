import torch
import torch.nn as nn

from pathlib import Path
from omegaconf import OmegaConf


class EmotionClassifier(nn.Module):
    """
    下游任务。
    这里默认定义的是8分类任务。
    """
    def __init__(self, input_dim=768, num_classes=8):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out


if __name__ == '__main__':
    pass
