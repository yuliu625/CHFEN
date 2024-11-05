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
    def __init__(self, is_need_audio=True, is_need_positional_encoding=True, ):
        super().__init__()
        self.is_need_audio = is_need_audio
        self.is_need_positional_encoding = is_need_positional_encoding

        self.learnable_query = LearnableQuery()

        # self.positional_encoding = ListPositionalEncoding(d_model=768)  # 处理效率太低了
        # self.positional_encoding = PositionalEncoding(d_model=768)  # 方法不对，shape改变了
        self.positional_encoding = positional_encoding

        self.conditional_image_encoder = ConditionedImageEncoder()
        self.conditional_text_encoder = ConditionedTextEncoder()

    # def forward(self, embedding_dict):
    #     title_embedding = embedding_dict['title_embedding']
    #     image_embedding_dict = {
    #         'scene_embedding_list': embedding_dict['scene_embedding_list'],
    #         'face_embedding_list': embedding_dict['face_embedding_list'],
    #     }
    #     text_embedding_dict = {
    #         'text_embedding_list': embedding_dict['text_embedding_list'],
    #     }
    #
    #     conditioned_query_embedding = title_embedding + self.learnable_query()
    #     image_embedding_list = self.conditional_image_encoder(conditioned_query_embedding, image_embedding_dict)
    #     text_embedding_list = self.conditional_text_encoder(conditioned_query_embedding, text_embedding_dict)
    #
    #     image_embedding_sequence = torch.cat(image_embedding_list, dim=0)
    #     text_embedding_sequence = torch.cat(text_embedding_list, dim=0)
    #
    #     if self.is_need_positional_encoding:
    #         # 如何进行了位置编码，这里的形状会发生变化。(sequence_length, embedding_dim)
    #         # image_embedding_sequence = self.positional_encoding(image_embedding_sequence)
    #         # text_embedding_sequence = self.positional_encoding(text_embedding_sequence)
    #         image_embedding_sequence += self.positional_encoding(image_embedding_sequence.shape[0], image_embedding_sequence.shape[1])
    #         text_embedding_sequence += self.positional_encoding(text_embedding_sequence.shape[0], text_embedding_sequence.shape[1])
    #
    #     result = {
    #         'title_embedding': title_embedding,
    #         # 'image_embedding_list': image_embedding_list,
    #         # 'text_embedding_list': text_embedding_list,
    #         'image_embedding_sequence': image_embedding_sequence,
    #         'text_embedding_sequence': text_embedding_sequence,
    #     }
    #     if self.is_need_audio:
    #         result['audio_embedding'] = embedding_dict['audio_embedding']
    #
    #     return result

    def forward(self, embedding_dict):
        title_embedding = embedding_dict['title_embedding']
        image_embedding_dict = {
            'scene_embedding_list': embedding_dict['scene_embedding_list'],
            'face_embedding_list': embedding_dict['face_embedding_list'],
        }
        text_embedding_dict = {
            'text_embedding_list': embedding_dict['text_embedding_list'],
            # 'text_embedding_sequence': torch.cat(embedding_dict['text_embedding_list'], dim=0),
        }

        conditioned_query_embedding = title_embedding + self.learnable_query()
        image_embedding_sequence = self.conditional_image_encoder(conditioned_query_embedding, image_embedding_dict)
        text_embedding_sequence = self.conditional_text_encoder(conditioned_query_embedding, text_embedding_dict)

        # image_embedding_sequence = torch.cat(image_embedding_list, dim=0)
        # text_embedding_sequence = torch.cat(text_embedding_list, dim=0)

        # 没必要判断，这里是可能需要位置编码的。
        # if self.is_need_positional_encoding:
        #     # 如何进行了位置编码，这里的形状会发生变化。(sequence_length, embedding_dim)
        #     # image_embedding_sequence = self.positional_encoding(image_embedding_sequence)
        #     # text_embedding_sequence = self.positional_encoding(text_embedding_sequence)
        #     image_embedding_sequence += self.positional_encoding(image_embedding_sequence.shape[0],
        #                                                          image_embedding_sequence.shape[1])
        #     text_embedding_sequence += self.positional_encoding(text_embedding_sequence.shape[0],
        #                                                         text_embedding_sequence.shape[1])

        result = {
            'title_embedding': title_embedding,
            # 'image_embedding_list': image_embedding_list,
            # 'text_embedding_list': text_embedding_list,
            'image_embedding_sequence': image_embedding_sequence,
            'text_embedding_sequence': text_embedding_sequence,
        }
        if self.is_need_audio:
            result['audio_embedding'] = embedding_dict['audio_embedding']

        return result


if __name__ == '__main__':
    pass
