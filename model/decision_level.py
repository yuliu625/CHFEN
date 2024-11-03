import torch
import torch.nn as nn

from pathlib import Path
from omegaconf import OmegaConf


class DecisionModule(nn.Module):
    """
    输入：
        feature-level已经处理过的各个模态的embedding。
    处理：
        给各个模态的embedding加入各自类别的embedding。
        使用self attention加ffn提取信息。
        使用pooling聚合语义信息。
        使用classifier进行分类。
    输出：
        classifier的输出，即softmax的输出。
    """
    def __init__(self):
        super().__init__()

        self.self_attention_layer = nn.MultiheadAttention()

    def forward(self, x):
        pass


if __name__ == '__main__':
    pass
