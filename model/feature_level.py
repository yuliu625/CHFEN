# from .conditioned_image_encoder import
# from .conditioned_text_encoder import
from .positional_encoding import PositionalEncoding

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


# class Gate


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
    def __init__(self, config):
        super().__init__()

    def forward(self, embeddings_dict):
        pass


if __name__ == '__main__':
    pass
