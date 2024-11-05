import torch
import torch.nn as nn

from pathlib import Path
from omegaconf import OmegaConf


class ImageFusionLayer(nn.Module):
    """
    以cross attention实现的图片信息聚合，看起来最合理。
    query是由num_faces扩展得到的tensor。
    这个模块似乎不需要位置编码。
    """
    def __init__(self, embedding_dim=768, num_heads=8):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)

    def forward(self, num_faces: int, face_embedding, scene_embedding):
        query = torch.full_like(face_embedding, fill_value=num_faces)
        key = torch.cat([face_embedding, scene_embedding], dim=0)
        value = key
        attention_output, attention_output_weights = self.cross_attention(query, key, value)
        return attention_output


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


class GatedFusion(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        # 定义gate的权重参数，用于控制两个embedding的组合
        self.gate_fc = nn.Linear(embedding_dim * 2, 1)  # 输入为两个embedding拼接后的维度
        self.sigmoid = nn.Sigmoid()

    def forward(self, embedding_a, embedding_b):
        """
        :param embedding_a: 模态A的embedding, shape: (batch_size, embedding_dim)
        :param embedding_b: 模态B的embedding, shape: (batch_size, embedding_dim)
        :return: 融合后的embedding, shape: (batch_size, embedding_dim)
        """
        # 将两个embedding拼接在一起
        combined = torch.cat((embedding_a, embedding_b), dim=1)  # shape: (batch_size, embedding_dim * 2)

        # 通过gate网络计算门控值
        gate = self.sigmoid(self.gate_fc(combined))  # shape: (batch_size, 1)

        # 基于gate值对两个embedding加权求和
        gated_embedding = gate * embedding_a + (1 - gate) * embedding_b  # shape: (batch_size, embedding_dim)

        return gated_embedding, gate


if __name__ == '__main__':
    pass
