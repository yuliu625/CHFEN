from .original_dataset import OriginalMultimodalDataset

from embedding import TextEncoder, Captioner, AudioEncoder

import torch
import torchvision.transforms as transforms


class ProcessedMultimodalDataset(OriginalMultimodalDataset):
    def __init__(self, image_transform, is_need_audio=True, path_config_path_str='../configs/path.yaml'):
        super().__init__(is_need_audio, path_config_path_str)
        self.image_transform = image_transform

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return data

    def transform_image(self):
        """将图片进行处理转换。主要是resize。"""
        pass

    def get_facial_image(self):
        """通过scene得到人脸。"""
        pass


def build_transform():
    pass


if __name__ == '__main__':
    pass
