# from .conditioned_image_encoder import
# from .conditioned_text_encoder import

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


if __name__ == '__main__':
    pass
