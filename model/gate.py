import torch
import torch.nn as nn

from pathlib import Path
from omegaconf import OmegaConf


class GatedLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 定义一个线性层用于生成 gate
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()  # 将输出限制在0到1之间
        )

    def forward(self, x):
        # 通过 gate 层生成的权重控制 x 的传递
        gate = self.gate(x)   # (batch_size, input_dim)，值在 0 和 1 之间
        gated_output = gate * x  # 输入元素逐元素乘以 gate，选择性传递
        return gated_output


if __name__ == '__main__':
    pass
