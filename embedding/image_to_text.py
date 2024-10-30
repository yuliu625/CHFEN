import torch

from transformers import BlipProcessor, BlipForConditionalGeneration

import numpy as np
from PIL import Image

from pathlib import Path
from omegaconf import OmegaConf


class Captioner:
    """caption生成器。"""
    def __init__(self, model='blip', encoder_config_path_str='../configs/encoder.yaml'):
        # 导入配置。
        self.config = OmegaConf.load(encoder_config_path_str)

        # encoder_path 并选择模型。
        self.encoder_path = Path(self.config['text'][model]['path'])


if __name__ == '__main__':
    pass
