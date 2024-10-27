import torch
from transformers import RobertaTokenizer, RobertaModel

from pathlib import Path
from omegaconf import OmegaConf


class TextEncoder:
    def __init__(self, model='roberta', encoder_config_path_str='../config/encoder.yaml'):
        # 导入配置。
        self.config = OmegaConf.load(encoder_config_path_str)

        # encoder_path 并选择模型。
        self.encoder_path = Path(self.config['text'][model]['path'])


if __name__ == '__main__':
    pass
