from embedding import TotalEncoder
from model.feature_level import FeatureModule
from model.decision_level import DecisionModule
from model.classifier import EmotionClassifier
from model.projection_layer import ProjectionLayer

import torch
import torch.nn as nn

from pathlib import Path
from omegaconf import OmegaConf


class CHFEN(nn.Module):
    def __init__(self):
        super().__init__()
        self.total_encoder = TotalEncoder()
        self.feature_module = FeatureModule()
        # self.decision_module = DecisionModule()

    def forward(self, data):
        out = self.total_encoder(data)
        out = self.feature_module(out)
        return out


if __name__ == '__main__':
    pass
