import torch

from transformers import ViTModel, ViTFeatureExtractor

from pathlib import Path
from omegaconf import OmegaConf


class ImageEncoder:
    def __init__(self, model='vit', encoder_config_path_str='../configs/encoder.yaml'):
        # 导入配置。
        self.config = OmegaConf.load(encoder_config_path_str)

        # encoder_path 并选择模型。
        self.encoder_path = Path(self.config['image'][model]['path'])
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.encoder_path)
        self.model = ViTModel.from_pretrained(self.encoder_path)

    def encode(self, image):
        # 使用特征提取器处理图片
        inputs = self.feature_extractor(images=image, return_tensors="pt")

        # 将图片输入模型并获取输出
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 提取最后一层的CLS token embedding作为图片的embedding
        image_embedding = outputs.last_hidden_state[:, 0, :]


if __name__ == '__main__':
    pass
