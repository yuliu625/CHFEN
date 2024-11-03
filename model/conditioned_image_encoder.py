from model.projection_layer import ProjectionLayer
from model.gate import ImageFusionLayer

import torch
import torch.nn as nn

from pathlib import Path
from omegaconf import OmegaConf


class ConditionedImageEncoder(nn.Module):
    """
    根据全局query，对于当前的image_embedding_list中的各个embedding进行查询。
    实现的操作：
        - 聚合face和scene的语义信息。
        - 提取每张图片的语义信息。
    需要输入：
        image_embeddings_input: dict{'scene_embedding_list', 'face_embedding_list'}
    返回：
        一个相同sequence_length的embedding_list。
    """
    def __init__(self, embedding_dim=768, num_heads=8):
        super().__init__()
        self.image_fusion_layer = ImageFusionLayer()

        # 这个是conditioned query的实现。
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)

    def forward(self, conditioned_query_embedding, image_embeddings_input):
        scene_embedding_list = image_embeddings_input['scene_embedding_list']
        face_embedding_list = image_embeddings_input['face_embedding_list']

        sequence_length = len(scene_embedding_list)

        result_embedding_list = []
        for i in range(sequence_length):
            scene_embedding = scene_embedding_list[i]
            num_faces, face_embedding = face_embedding_list[i]
            image_embedding = self.image_fusion_layer(num_faces, scene_embedding, face_embedding)

            attention_output, _ = self.cross_attention(conditioned_query_embedding, image_embedding, image_embedding)
            result_embedding_list.append(attention_output)

        return result_embedding_list


if __name__ == '__main__':
    pass
