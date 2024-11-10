from model.conditioned_image_encoder import ConditionedImageEncoder
from model.conditioned_text_encoder import ConditionedTextEncoder
from model.positional_encoding import positional_encoding, PositionalEncoding, ListPositionalEncoding

import torch
import torch.nn as nn

from pathlib import Path
from omegaconf import OmegaConf


class LearnableQuery(nn.Module):
    """
    这是conditional的具体实现的一部分。
    为了更好的效果，以及实现无标题时候的查询。
    """
    def __init__(self, query_dim=768):
        """这里默认embedding_dim是768，是因为roberta的embedding_dim是768。"""
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, query_dim))

    def forward(self):
        return self.query


class FeatureModule(nn.Module):
    """
    输入：
        dataset中的各个模态的embedding。
    处理：
        title：使用learnable_query成为更好的query。
        image&text sequence：
            输入2个conditional模块。
            需要同时输入query，需要进行位置编码。
            得到embedding sequence。
        audio：直接输出。
    输出：
        title：原始embedding。
        image&text：embedding sequence。
        audion：原始embedding。
    """
    def __init__(self, is_need_positional_encoding=True, ):
        super().__init__()
        self.is_need_positional_encoding = is_need_positional_encoding

        # self.learnable_query = LearnableQuery()
        self.learnable_query = nn.Parameter(torch.randn(1, 768))

        # self.positional_encoding = ListPositionalEncoding(d_model=768)  # 处理效率太低了
        # self.positional_encoding = PositionalEncoding(d_model=768)  # 方法不对，shape改变了
        self.positional_encoding = PositionalEncoding()

        self.conditional_image_encoder = ConditionedImageEncoder()
        self.conditional_text_encoder = ConditionedTextEncoder()

    def forward(self, embedding_dict):
        title_embedding = embedding_dict['title_embedding']
        image_embedding_dict = {
            'scene_embeddings': embedding_dict['scene_embeddings'],
            # 'scene_mask': embedding_dict['scene_mask'],
            'num_faces': embedding_dict['num_faces'],
            'face_embeddings': embedding_dict['face_embeddings'],
            # 'face_mask': embedding_dict['face_mask'],
        }
        text_embedding_dict = {
            'text_embeddings': embedding_dict['text_embeddings'],
            # 'text_mask': embedding_dict['text_mask'],
        }
        audio_embedding = embedding_dict['audio_embedding']

        # 这里的全局query或许需要将标题和音频都加上。
        conditioned_query_embedding = self.learnable_query + title_embedding + audio_embedding

        # 执行条件查询。
        image_embedding_sequence = self.conditional_image_encoder(conditioned_query_embedding, image_embedding_dict)
        text_embedding_sequence = self.conditional_text_encoder(conditioned_query_embedding, text_embedding_dict)

        result = {
            'title_embedding': title_embedding,
            'image_embedding_sequence': image_embedding_sequence,
            'text_embedding_sequence': text_embedding_sequence,
            'audio_embedding': audio_embedding,
        }

        return result


if __name__ == '__main__':
    pass
