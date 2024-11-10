from untils import get_sequence_length_from_num_faces_vector

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
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)

    def forward(self, num_faces, face_embeddings, scene_embeddings):
        # 下面的是不用矩阵，纯for loop。
        # if num_faces == 0:
        #     # 对于没有人脸的情况，那就是场景。
        #     return scene_embedding
        # # 对于有人脸的情况，正常进行融合。
        # query = torch.full_like(face_embedding, fill_value=num_faces)
        # key = torch.cat([face_embedding, scene_embedding], dim=0)
        # value = key
        # attention_output, attention_output_weights = self.cross_attention(query, key, value)

        # 以下这段不要删除，这是单条数据的情况下的正确处理，是可以正常运行的。
        # # 根据原本的序列长度，还原原本的sequence
        # sequence_length = get_sequence_length_from_num_faces_vector(num_faces)
        # num_faces = num_faces[:sequence_length]
        # face_embeddings = face_embeddings[:sequence_length]
        # scene_embeddings = scene_embeddings[:sequence_length]
        # # 先替换0即没有人的情况，然后扩展至查询方式。
        # num_faces = torch.where(num_faces == 0, torch.tensor( 1e-6 ), num_faces)
        # num_faces = num_faces.unsqueeze(2).expand_as(scene_embeddings)

        # 因为batch进行的改造。不去计算序列长度，而使用掩码。
        num_faces = torch.where(num_faces == 0, torch.tensor(1e-6, device=scene_embeddings.device), num_faces)
        num_faces = num_faces.unsqueeze(2).expand_as(scene_embeddings)
        num_faces = torch.where(num_faces == -1, torch.tensor(0, device=scene_embeddings.device), num_faces)

        # print(type(num_faces))  # <class 'torch.Tensor'>
        # print(num_faces.shape)  # torch.Size([2, 19, 768])
        # num_faces = torch.where(num_faces == -1, torch.tensor(0, device=num_faces.device), num_faces)

        # 还是不用掩码了，好麻烦。
        # face_mask = (num_faces != -1).float()  # [batch_size, seq_len, embedding_dim]
        # face_mask = face_mask.max(dim=2).values  # [batch_size, seq_len], max over embedding_dim

        attention_output, _ = self.cross_attention(num_faces, face_embeddings, scene_embeddings)
        # print(type(attention_output))  # <class 'torch.Tensor'>
        # print(attention_output.shape)  # torch.Size([2, 19, 768])

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
