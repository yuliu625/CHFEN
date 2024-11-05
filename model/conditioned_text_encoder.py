from model.projection_layer import ProjectionLayer
from model.positional_encoding import positional_encoding

import torch
import torch.nn as nn

from pathlib import Path
from omegaconf import OmegaConf


class ConditionedTextEncoder(nn.Module):
    """
        根据全局query，对于当前的text_embedding_list中的各个embedding进行查询。
        需要输入：
            text_embeddings_input: dict{'text_embedding_list'}
        返回：
            一个相同sequence_length的embedding_list。
        """
    def __init__(self, embedding_dim=768, num_heads=8):
        super().__init__()

        # 这个是conditioned query的实现。
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)

        self.positional_encoding = positional_encoding

    def forward(self, conditioned_query_embedding, text_embeddings_input):
        text_embedding_list = text_embeddings_input['text_embedding_list']

        # 按照list进行处理的做法。
        # result_embedding_list = []
        # for text_embedding in text_embedding_list:
        #     attention_output, _ = self.cross_attention(conditioned_query_embedding, text_embedding, text_embedding)
        #     result_embedding_list.append(attention_output)
        #
        # return result_embedding_list

        # 矩阵化处理方法, 加入位置编码。。
        text_embedding_sequence = torch.cat(text_embedding_list, dim=0)
        text_embedding_sequence += self.positional_encoding(text_embedding_sequence.shape[0], text_embedding_sequence.shape[1])
        conditioned_query_embedding = conditioned_query_embedding.expand_as(text_embedding_sequence)

        # 计算结果。
        attention_output, _ = self.cross_attention(conditioned_query_embedding, text_embedding_sequence, text_embedding_sequence)

        return attention_output


if __name__ == '__main__':
    pass
