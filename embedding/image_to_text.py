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
        self.generator_path = Path(self.config['text'][model]['path'])
        self.processor = BlipProcessor.from_pretrained(self.generator_path)
        self.model = BlipForConditionalGeneration.from_pretrained(self.generator_path)

    def generate(self, np_array_image):
        pil_image = self.trans_np_array_to_image(np_array_image)

        inputs = self.processor(pil_image, return_tensors='pt')

        pixel_values = inputs['pixel_values']  # 这里直接使用model(**inputs)有问题。

        outputs = self.model.generate(pixel_values)
        captions = self.processor.decode(outputs[0], skip_special_tokens=True)

        return captions

    def trans_np_array_to_image(self, np_array_image):
        """由于moviepy中的帧图片是ndarray，这里需要转换之后才能输入blip captioner。"""
        pil_img = Image.fromarray(np_array_image)
        return pil_img


if __name__ == '__main__':
    pass
