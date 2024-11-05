from model.projection_layer import ProjectionLayer
from model.positional_encoding import positional_encoding
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

        self.positional_encoding = positional_encoding

        # 这个是conditioned query的实现。
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)

    # def forward(self, conditioned_query_embedding, image_embeddings_input):
    #     scene_embedding_list = image_embeddings_input['scene_embedding_list']
    #     face_embedding_list = image_embeddings_input['face_embedding_list']
    #
    #     sequence_length = len(scene_embedding_list)
    #
    #     result_embedding_list = []
    #     for i in range(sequence_length):
    #         scene_embedding = scene_embedding_list[i]
    #         num_faces, face_embedding = face_embedding_list[i]
    #         image_embedding = self.image_fusion_layer(num_faces, scene_embedding, face_embedding)
    #
    #         attention_output, _ = self.cross_attention(conditioned_query_embedding, image_embedding, image_embedding)
    #         result_embedding_list.append(attention_output)
    #
    #     return result_embedding_list

    # def forward(self, conditioned_query_embedding, image_embeddings_input):
    #     scene_embedding_list = image_embeddings_input['scene_embedding_list']
    #     face_embedding_list = image_embeddings_input['face_embedding_list']
    #
    #     num_faces_list = [face_embedding[0] for face_embedding in face_embedding_list]
    #     face_embedding_list_ = [face_embedding[1] for face_embedding in face_embedding_list]
    #
    #     scene_embedding_sequence = torch.cat(scene_embedding_list, dim=1)
    #     face_embedding_sequence = torch.cat(face_embedding_list_, dim=0)
    #
    #
    #     result_embedding_list = []
    #     for i in range(sequence_length):
    #         scene_embedding = scene_embedding_list[i]
    #         num_faces, face_embedding = face_embedding_list[i]
    #         image_embedding = self.image_fusion_layer(num_faces, scene_embedding, face_embedding)
    #
    #         attention_output, _ = self.cross_attention(conditioned_query_embedding, image_embedding, image_embedding)
    #         result_embedding_list.append(attention_output)
    #
    #     return result_embedding_list

    def forward(self, conditioned_query_embedding, image_embeddings_input):
        scene_embedding_list = image_embeddings_input['scene_embedding_list']
        face_embedding_list = image_embeddings_input['face_embedding_list']

        # face和scene融合，这不进行位置编码。
        sequence_length = len(scene_embedding_list)
        image_result_embedding_list = []
        for i in range(sequence_length):
            scene_embedding = scene_embedding_list[i]
            num_faces, face_embedding = face_embedding_list[i]
            image_embedding = self.image_fusion_layer(num_faces, scene_embedding, face_embedding)
            image_result_embedding_list.append(image_embedding)

        # 矩阵化计算，加入位置编码。
        image_embedding_sequence = torch.cat(image_result_embedding_list, dim=0)
        image_embedding_sequence += positional_encoding(image_embedding_sequence.shape[0], image_embedding_sequence.shape[1])
        conditioned_query_embedding = conditioned_query_embedding.expand_as(image_embedding_sequence)

        # 计算结果。
        attention_output, _ = self.cross_attention(conditioned_query_embedding, image_embedding_sequence, image_embedding_sequence)

        return attention_output


if __name__ == '__main__':
    pass
