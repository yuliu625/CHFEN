from .projection_layer import ProjectionLayer

import torch
import torch.nn as nn

from pathlib import Path
from omegaconf import OmegaConf


class ConditionedTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, conditioned_query_embedding, text_embeddings_input):
        pass


if __name__ == '__main__':
    pass
