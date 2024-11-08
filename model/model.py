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
    """
    模型整体。
    主要的模型结构：
        - total_encoder: 将data编码成embedding。
        - feature_module: 进行特征级融合。
        - decision_module: 进行决策级融合。
        - classifier: 当前具体任务定义的分类器。
    特性：
        - total_encoder是来自transformers的，所有情况都会被冻结。
        - 数据传递是通过dict，每个模块内已经写好了处理方法。
    """
    def __init__(self, config=None):
        super().__init__()
        # self.total_encoder = TotalEncoder(encoder_config_path_str=config['model']['encoder_config_path'])
        self.feature_module = FeatureModule()
        self.decision_module = DecisionModule()
        self.classifier = EmotionClassifier()

    def forward(self, data):
        # with torch.no_grad():
        #     out = self.total_encoder(data)
        out = self.feature_module(data)
        out = self.decision_module(out)
        out = self.classifier(out)
        return out

    # def freeze_total_encoder(self):
    #     """在将模型实例化之后，这个方法就需要被调用。"""
    #     for param in self.total_encoder.parameters():
    #         param.requires_grad = False


if __name__ == '__main__':
    pass
