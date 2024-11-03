import torch
import torch.nn as nn

import math

from pathlib import Path
from omegaconf import OmegaConf


class ListPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(ListPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = len(x)
        # 将位置编码与输入embedding相加
        x = torch.stack(x) + self.pe[:seq_len]
        return x


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: (max_len, 1, d_model)

        # Register pe as a buffer to avoid updating it in backpropagation
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add position encoding to input tensor
        x = x + self.pe[:x.size(0), :]
        return x


if __name__ == '__main__':
    # 一般的位置编码使用方法
    # d_model = 512  # embedding的维度
    # pos_encoder = PositionalEncoding(d_model)  # 实例化位置编码模块
    #
    # x = torch.randn(50, 32, d_model)  # 示例输入 (seq_len=50, batch_size=32, embedding_dim=d_model)
    # x_pos = pos_encoder(x)  # 将位置编码加到embedding上
    # print(x_pos.shape)

    # list的位置编码使用
    # 假设embedding_dim为512
    embedding_dim = 512
    pos_encoder = ListPositionalEncoding(embedding_dim)

    # 示例输入：嵌套列表形式的embedding
    x = [torch.randn(1, embedding_dim) for _ in range(10)]  # 假设序列长度为10
    x_pos = pos_encoder(x)
    print(x_pos.shape)
    print(x_pos)
